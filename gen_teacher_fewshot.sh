#!/bin/bash
#SBATCH --job-name=gen_teacher_fewshot
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=54:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# gen_teacher_fewshot.sh -- Phase 1 generation only (no judging), teacher
# model, fewshot prompt, 4321-question pool.

set -e
set -x

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
N_SAMPLES=4321
N_HIGH_TEMP_SAMPLES=10
OUTPUT_DIR="${PROJECT_DIR}/gen_data"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_teacher_fewshot | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python generate_responses.py \
        --model_role teacher \
        --dataset "${DATASET}" \
        --prompt_style fewshot \
        --n_samples "${N_SAMPLES}" \
        --n_high_temp_samples "${N_HIGH_TEMP_SAMPLES}" \
        --output_dir "${OUTPUT_DIR}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_teacher_fewshot_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_teacher_fewshot_done | node=$(hostname)" >> "${MANIFEST}"