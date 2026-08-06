#!/bin/bash
#SBATCH --job-name=gen_olmo
#SBATCH --partition=msc
#SBATCH --nodes=1
#SBATCH --nodelist=oat17
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/ms25yt/SimpleQA/logs/%x_%j.out
#SBATCH --error=/users/ms25yt/SimpleQA/logs/%x_%j.err
#
# gen_olmo.sh <teacher|student> <strict|fewshot>
#
# OLMo 2 cross-family line. Unlike Qwen3, Gemma, Ministral 3 and Llama
# 3.2, the OLMo 2 sizes were pretrained independently rather than
# distilled from the larger models, so the base student here has not
# already been through a strong-to-weak pass. Weights are in the group
# cache on oat17 only.

set -e
set -x

MODEL_ROLE="${1:?Usage: sbatch gen_olmo.sh <teacher|student> <strict|fewshot>}"
PROMPT_STYLE="${2:?Usage: sbatch gen_olmo.sh <teacher|student> <strict|fewshot>}"

if [ "${MODEL_ROLE}" == "teacher" ]; then
    MODEL_NAME="allenai/OLMo-2-0325-32B-Instruct"
elif [ "${MODEL_ROLE}" == "student" ]; then
    MODEL_NAME="allenai/OLMo-2-1124-7B-Instruct"
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
OUTPUT_DIR="${PROJECT_DIR}/gen_data_seed44_olmo"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

set +x
source ~/.bashrc
conda activate haldist
set -x

# OLMo weights live in the group cache, so do not switch HF_HOME here.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "===== [$(date)] host=$(hostname) HF_HOME=${HF_HOME} role=${MODEL_ROLE} style=${PROMPT_STYLE} model=${MODEL_NAME} ====="
nvidia-smi --query-gpu=gpu_bus_id,memory.free,temperature.gpu --format=csv

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_olmo_${MODEL_ROLE}_${PROMPT_STYLE} | model=${MODEL_NAME} | node=$(hostname)" >> "${MANIFEST}"

python -u generate_responses.py \
    --model_role "${MODEL_ROLE}" \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --prompt_style "${PROMPT_STYLE}" \
    --question_indices_file "${SUBSET_FILE}" \
    --n_high_temp_samples 10 \
    --output_dir "${OUTPUT_DIR}"

echo "===== [$(date)] Done. Output in ${OUTPUT_DIR} ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_olmo_done | model=${MODEL_NAME} | node=$(hostname)" >> "${MANIFEST}"
