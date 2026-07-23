#!/bin/bash
#SBATCH --job-name=gen_longform
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage:
#   sbatch --job-name=gen_longform_teacher gen_longform.sh teacher
#   sbatch --job-name=gen_longform_student gen_longform.sh student
#
# REMINDER: move any existing gen_factscore_bio_*.jsonl for this role out
# of OUTPUT_DIR before submitting. This script resumes from checkpoint and
# will silently skip entities that are already present.

set -e
set -x

export WANDB_MODE=disabled

MODEL_ROLE="$1"
if [ "${MODEL_ROLE}" != "teacher" ] && [ "${MODEL_ROLE}" != "student" ]; then
    echo "ERROR: first argument must be 'teacher' or 'student', got '${MODEL_ROLE}'."
    exit 1
fi

PROJECT_DIR=~/SimpleQA
OUTPUT_DIR="${PROJECT_DIR}/gen_longform_data"
SUBSET="${OUTPUT_DIR}/sampled_100_entities.jsonl"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"
STAGE="gen_longform_${MODEL_ROLE}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=${STAGE} | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python generate_longform_responses.py \
        --model_role "${MODEL_ROLE}" \
        --question_idx_subset "${SUBSET}" \
        --output_dir "${OUTPUT_DIR}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=${STAGE}_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=${STAGE}_done | node=$(hostname)" >> "${MANIFEST}"