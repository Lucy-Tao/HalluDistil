import pathlib

p = pathlib.Path("gen_llama.sh")
s = p.read_text()

# The gate installed earlier: aborts when the allocation is bad.
old_start = "# --- cross-GPU copy gate ---"
old_end = "P2PEOF\nfi\n"

i = s.find(old_start)
if i == -1:
    raise SystemExit("existing gate not found; nothing replaced")
j = s.find(old_end, i)
if j == -1:
    raise SystemExit("end of existing gate not found")
j += len(old_end)

new = '''# --- cross-GPU copy gate -------------------------------------
# Some nodes on this cluster silently return all-zero tensors on
# GPU-to-GPU copies (seen on oat10, oat16, oat17). A 70B model split
# across a bad pair produces zero hidden states from the first layer on
# the second device and generates garbage, with no error raised. On
# oat11 the failure was per-pair rather than per-node: copies INTO GPU 0
# or 1 were fine, copies into 2 or 3 were not. So rather than aborting
# on a bad allocation, search for a pair that works in both directions
# and restrict the job to it.
#
# Request 4 GPUs for 70B runs so there is something to choose from; the
# model still only needs 2. CUDA_VISIBLE_DEVICES set here is interpreted
# relative to the devices SLURM already exposed, so the indices below
# are the right thing to export. Single-GPU runs skip the check.
if [ "$(nvidia-smi --list-gpus | wc -l)" -gt 1 ]; then
    GOOD_PAIR=$(python - <<'P2PEOF'
import sys, socket, torch

def copy_ok(src, dst):
    a = torch.arange(10000, device=f"cuda:{src}", dtype=torch.float32)
    b = a.to(f"cuda:{dst}")
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    # arange starts at 0, so exactly one legitimate zero is expected.
    return (b.cpu() == 0).sum().item() == 1

n = torch.cuda.device_count()
tested = []
for i in range(n):
    for j in range(i + 1, n):
        both = copy_ok(i, j) and copy_ok(j, i)
        tested.append(f"{i}<->{j}:{'ok' if both else 'BROKEN'}")
        if both:
            print(f"{i},{j}")
            print(" ".join(tested), file=sys.stderr)
            sys.exit(0)
print(socket.gethostname(), "no usable pair:", " ".join(tested), file=sys.stderr)
sys.exit(1)
P2PEOF
)
    if [ -z "${GOOD_PAIR}" ]; then
        echo "ABORT: no GPU pair on $(hostname) survives a round-trip copy."
        echo "Resubmit to get a different allocation."
        exit 1
    fi
    echo "===== using GPU pair ${GOOD_PAIR} on $(hostname) ====="
    export CUDA_VISIBLE_DEVICES="${GOOD_PAIR}"
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=p2p_gate | pair=${GOOD_PAIR} | node=$(hostname)" >> "${MANIFEST}"
fi

'''

s = s[:i] + new + s[j:]
p.write_text(s)
print("gate replaced")