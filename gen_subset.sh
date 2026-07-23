#!/bin/bash
#SBATCH --job-name=gen_responses
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# gen_subset.sh <model_role> <prompt_style> -- Phase 1 generation only
# (no judging) on the 500-question subset. Output goes to
# gen_data_subset500/ under $HOME, so no node pinning needed.
#
# Usage:
#   sbatch gen_subset.sh teacher strict
#   sbatch gen_subset.sh teacher fewshot
#   sbatch gen_subset.sh student strict
#   sbatch gen_subset.sh student fewshot

set -e
set -x

MODEL_ROLE="${1:?Usage: sbatch gen_subset.sh <teacher|student> <strict|fewshot>}"
PROMPT_STYLE="${2:?Usage: sbatch gen_subset.sh <teacher|student> <strict|fewshot>}"

if [[ "${MODEL_ROLE}" != "teacher" && "${MODEL_ROLE}" != "student" ]]; then
    echo "ERROR: model_role must be 'teacher' or 'student', got '${MODEL_ROLE}'"
    exit 1
fi
if [[ "${PROMPT_STYLE}" != "strict" && "${PROMPT_STYLE}" != "fewshot" ]]; then
    echo "ERROR: prompt_style must be 'strict' or 'fewshot', got '${PROMPT_STYLE}'"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
N_HIGH_TEMP_SAMPLES=10
SUBSET_FILE="${PROJECT_DIR}/subset_500_question_indices.json"
OUTPUT_DIR="${PROJECT_DIR}/gen_data_subset500"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
echo "===== model_role=${MODEL_ROLE} prompt_style=${PROMPT_STYLE} ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_${MODEL_ROLE}_${PROMPT_STYLE} | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python generate_responses.py \
        --model_role "${MODEL_ROLE}" \
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
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_${MODEL_ROLE}_${PROMPT_STYLE}_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_${MODEL_ROLE}_${PROMPT_STYLE}_done | node=$(hostname)" >> "${MANIFEST}"