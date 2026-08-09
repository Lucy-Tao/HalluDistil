#!/bin/bash
#SBATCH --job-name=gen_qwen25
#SBATCH --partition=msc
#SBATCH --nodes=1
#SBATCH --nodelist=oat11,oat15
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/ms25yt/SimpleQA/logs/%x_%j.out
#SBATCH --error=/users/ms25yt/SimpleQA/logs/%x_%j.err
#
# gen_qwen25.sh <teacher|student> <strict|fewshot>
#
# Qwen2.5 control line. The Qwen3 lightweight models were produced by
# strong-to-weak distillation from Qwen3-32B and Qwen3-235B-A22B, so the
# 4B base student in the main line is already a distilled student of its
# own teacher. Qwen2.5 has no such published within-family distillation,
# which makes this an A/B on whether the base student was pre-distilled
# while holding organisation, architecture and tokenizer roughly fixed.
#
# Weights are in the group cache on oat11 and oat15 only.

set -e
set -x

MODEL_ROLE="${1:?Usage: sbatch gen_qwen25.sh <teacher|student> <strict|fewshot>}"
PROMPT_STYLE="${2:?Usage: sbatch gen_qwen25.sh <teacher|student> <strict|fewshot>}"

if [ "${MODEL_ROLE}" == "teacher" ]; then
    MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
elif [ "${MODEL_ROLE}" == "student" ]; then
    MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
else
    echo "ERROR: model_role must be teacher or student, got '${MODEL_ROLE}'"
    exit 1
fi
if [[ "${PROMPT_STYLE}" != "strict" && "${PROMPT_STYLE}" != "fewshot" ]]; then
    echo "ERROR: prompt_style must be strict or fewshot, got '${PROMPT_STYLE}'"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
SUBSET_FILE="${PROJECT_DIR}/subset_500_seed44_question_indices.json"
OUTPUT_DIR="${PROJECT_DIR}/gen_data_seed44_qwen25"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

set +x
source ~/.bashrc
conda activate haldist
set -x

# Qwen2.5 weights live in the group cache, not the personal one, so do
# not switch HF_HOME here the way the Qwen3 scripts do.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "===== [$(date)] host=$(hostname) HF_HOME=${HF_HOME} role=${MODEL_ROLE} style=${PROMPT_STYLE} model=${MODEL_NAME} ====="
nvidia-smi --query-gpu=gpu_bus_id,memory.free,temperature.gpu --format=csv

# A 32B teacher needs about 62G. SLURM accounts for GPUs but does not
# partition device memory, so another job on the same card can leave
# too little and the load then dies partway through. Failing here costs
# seconds instead of minutes and says plainly why.
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${MODEL_ROLE}" == "teacher" ] && [ "${GPU_FREE}" -lt 68000 ]; then
    echo "GPU_GATE_FAIL only ${GPU_FREE}MiB free, 32B needs about 65GiB"
    exit 1
fi

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_qwen25_${MODEL_ROLE}_${PROMPT_STYLE} | model=${MODEL_NAME} | node=$(hostname)" >> "${MANIFEST}"

python -u generate_responses.py \
    --model_role "${MODEL_ROLE}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --prompt_style "${PROMPT_STYLE}" \
    --question_indices_file "${SUBSET_FILE}" \
    --n_high_temp_samples 10 \
    --output_dir "${OUTPUT_DIR}"

echo "===== [$(date)] Done. Output in ${OUTPUT_DIR} ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_qwen25_done | model=${MODEL_NAME} | node=$(hostname)" >> "${MANIFEST}"
