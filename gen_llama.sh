#!/bin/bash
#SBATCH --job-name=gen_llama
#SBATCH --partition=msc
#SBATCH --nodes=1
#SBATCH --nodelist=oat14
#SBATCH --gres=gpu:a100:3
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# gen_llama.sh <teacher|student> <strict|fewshot>
#
# Llama line, phase 1 generation only. Mirrors gen_subset.sh but pinned to
# oat14 (only node with complete Llama weights) and forced offline (HF gate
# request was rejected, any network validation returns 403).
#
# Repo id note: teacher uses the OLD name, student uses the NEW name.
# Only those two cache dirs on oat14 hold real weights.
#
# Usage:
#   sbatch gen_llama.sh student strict
#   sbatch gen_llama.sh teacher strict   (3 GPUs, set in the header)

set -e
set -x

MODEL_ROLE="${1:?Usage: sbatch gen_llama.sh <teacher|student> <strict|fewshot>}"
PROMPT_STYLE="${2:?Usage: sbatch gen_llama.sh <teacher|student> <strict|fewshot>}"

if [[ "${MODEL_ROLE}" != "teacher" && "${MODEL_ROLE}" != "student" ]]; then
    echo "ERROR: model_role must be teacher or student, got '${MODEL_ROLE}'"
    exit 1
fi
if [[ "${PROMPT_STYLE}" != "strict" && "${PROMPT_STYLE}" != "fewshot" ]]; then
    echo "ERROR: prompt_style must be strict or fewshot, got '${PROMPT_STYLE}'"
    exit 1
fi

if [ "${MODEL_ROLE}" == "teacher" ]; then
    MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct"
    N_GPU=$(nvidia-smi --list-gpus | wc -l)
    # 70B in bf16 is about 140GB of weights, so two 80GB cards fit but
    # leave little room for KV cache. Three is comfortable. The old check
    # demanded four only so the P2P gate had a pair to choose from, and
    # that gate is now disabled.
    if [ "${N_GPU}" -lt 2 ]; then
        echo "ERROR: 70B needs at least 2 GPUs in bf16, got ${N_GPU}."
        exit 1
    fi
else
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
fi

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
N_HIGH_TEMP_SAMPLES=10
SUBSET_FILE="${PROJECT_DIR}/subset_500_seed44_question_indices.json"
OUTPUT_DIR="${PROJECT_DIR}/gen_data_seed44_llama"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

export HF_HOME=/scratch-ssd/oatml/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "===== [$(date)] host=$(hostname) ====="
echo "===== role=${MODEL_ROLE} style=${PROMPT_STYLE} model=${MODEL_NAME} ====="
nvidia-smi
# --- cross-GPU copy gate -------------------------------------
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
if false; then  # P2P gate disabled, let device_map=auto spread across all 4
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


echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_llama_${MODEL_ROLE}_${PROMPT_STYLE} | model=${MODEL_NAME} | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=3
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python -u generate_responses.py \
        --model_role "${MODEL_ROLE}" \
        --model_name "${MODEL_NAME}" \
        --dataset "${DATASET}" \
        --prompt_style "${PROMPT_STYLE}" \
        --question_indices_file "${SUBSET_FILE}" \
        --n_high_temp_samples "${N_HIGH_TEMP_SAMPLES}" \
        --output_dir "${OUTPUT_DIR}"
    exit_code=$?
    set -e
    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed, exit ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_llama_retry | attempt=${attempt} | exit=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts."
        exit "${exit_code}"
    fi
    sleep 60
done

echo "===== [$(date)] Done. Output in ${OUTPUT_DIR} ====="
