#!/bin/bash
#SBATCH --job-name=judge_se
#SBATCH --partition=msc
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/ms25yt/SimpleQA/logs/%x_%j.out
#SBATCH --error=/users/ms25yt/SimpleQA/logs/%x_%j.err
#
# judge_se.sh <gen_file> <filter|replace>
#
# Judging split out of distill_and_eval_se.sh. The three stages of that
# script need very different amounts of memory: distillation and
# generation fit in about 12G, judging needs roughly 66G for Qwen3-32B
# plus DeBERTa. SLURM accounts for GPUs but does not partition device
# memory, so a judging stage sharing a card with another job either
# hangs during model load or dies partway through. Keeping judging in
# its own serialised queue means a failure costs one job, not a whole
# pipeline run.
#
# Usage:
#   sbatch judge_se.sh <gen_file> filter

set -e
set -x

GEN_FILE="${1:?Usage: sbatch judge_se.sh <gen_file> <filter|replace>}"
# Second argument names the judged output directory. Accepts the short
# forms used by the SE runs, or a full directory name for anything else
# (noskip, raw_samples, and so on).
OUT_TAG="${2:?Usage: sbatch judge_se.sh <gen_file> <filter|replace|dirname>}"

case "${OUT_TAG}" in
    filter|replace) JUDGED_SUBDIR="judged_data_distilled_seed44_se_${OUT_TAG}" ;;
    *)              JUDGED_SUBDIR="${OUT_TAG}" ;;
esac

PROJECT_DIR=~/SimpleQA
JUDGED_OUTPUT_DIR="${PROJECT_DIR}/${JUDGED_SUBDIR}"
JUDGE_MODEL="Qwen/Qwen3-32B"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

# Judging loads Qwen3-32B (about 62G) plus DeBERTa (about 3.4G).
# Below this much free memory the load will hang rather than raise, so
# the check has to happen before any model touches the device.
MIN_FREE_MIB=72000

cd "${PROJECT_DIR}"
mkdir -p logs "${JUDGED_OUTPUT_DIR}"

set +x
source ~/.bashrc
conda activate haldist
set -x

# Prefer a node-local personal cache when present, else the group cache.
# Switch to the personal cache only when it actually holds the student
# weights. Testing that the directory exists is not enough: a node can
# have an empty hub/ left behind after a move, or hold an unrelated
# model, in which case switching hides a complete group cache.
if ls /scratch-ssd/ms25yt/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/*.safetensors >/dev/null 2>&1; then
    export HF_HOME=/scratch-ssd/ms25yt/hf
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "===== [$(date)] host=$(hostname) HF_HOME=${HF_HOME} ====="
echo "===== gen file: ${GEN_FILE} ====="

if [ ! -f "${GEN_FILE}" ]; then
    echo "ERROR: gen file not found: ${GEN_FILE}"
    exit 1
fi

GEN_LINES=$(wc -l < "${GEN_FILE}")
if [ "${GEN_LINES}" -lt 500 ]; then
    echo "ERROR: gen file has ${GEN_LINES} lines, expected 500. Regenerate first."
    exit 1
fi

# ── GPU health and capacity gate ─────────────────────────────
GPU_BUS=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader | head -1)
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
GPU_ERR=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | grep -c "ERR\|N/A" || true)
echo "GPU_GATE bus=${GPU_BUS} free=${GPU_FREE}MiB host=$(hostname)"

# A card whose NVML fields read ERR! can still be allocated by SLURM but
# cannot initialise CUDA. Seen on oat10 at bus CA:00.0, where 43 jobs
# died in sequence because each failure freed the slot for the next one.
if [ "${GPU_ERR}" -gt 0 ]; then
    echo "GPU_GATE_FAIL nvml reports ERR/NA on ${GPU_BUS}, card unhealthy"
    exit 1
fi
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.device_count() > 0 else 1)"; then
    echo "GPU_GATE_FAIL cuda init returned 0 devices on ${GPU_BUS}"
    exit 1
fi
if [ "${GPU_FREE}" -lt "${MIN_FREE_MIB}" ]; then
    echo "GPU_GATE_FAIL only ${GPU_FREE}MiB free on ${GPU_BUS}, need ${MIN_FREE_MIB}MiB"
    exit 1
fi

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_se_start | gen=${GEN_FILE} | node=$(hostname) | bus=${GPU_BUS} | free=${GPU_FREE}" >> "${MANIFEST}"

python -u judge_responses.py \
    --input "${GEN_FILE}" \
    --output_dir "${JUDGED_OUTPUT_DIR}" \
    --entailment_backend deberta \
    --grader qwen \
    --grader_model "${JUDGE_MODEL}" \
    --strict_entailment

echo "===== [$(date)] judging done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_se_done | gen=${GEN_FILE} | node=$(hostname)" >> "${MANIFEST}"