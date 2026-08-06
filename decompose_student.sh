#!/bin/bash
#SBATCH --job-name=decompose_student
#SBATCH --partition=msc
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# decompose_student.sh -- Phase 3, student side: decompose each student
# response into factual claims and verify each via web-search-grounded
# GPT-5-mini judging, through Oxford's Lagrange gateway.
#
# No --gres=gpu -- this is pure API/network I/O (like
# compute_abstention_rate.py's sbatch job), not GPU-bound. Submitted via
# sbatch (rather than just nohup on the login node) so it survives
# SSH/session drops over a potentially long run without needing to be
# watched.

set -e
set -x

PROJECT_DIR=~/SimpleQA
INPUT="${PROJECT_DIR}/gen_longform_data/answered_both.jsonl"
OUTPUT="${PROJECT_DIR}/gen_longform_data/claims_student.jsonl"
MODEL="gpt-5-mini"
BASE_URL="https://lagrange.uksouth.cloudapp.azure.com/openai"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=decompose_student | node=$(hostname)" >> "${MANIFEST}"

# NOTE: LAGRANGE_API_KEY must already be set as an environment variable
# (e.g. exported in ~/.bashrc) -- this script does not hardcode it, so
# it never ends up in the .out/.err log via `set -x`. If it's not set
# yet, export it in ~/.bashrc before submitting this job.
if [ -z "${LAGRANGE_API_KEY}" ]; then
    echo "ERROR: LAGRANGE_API_KEY is not set. Export it in ~/.bashrc and resubmit."
    exit 1
fi

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python decompose_and_verify.py \
        --input "${INPUT}" \
        --model_role student \
        --output "${OUTPUT}" \
        --model "${MODEL}" \
        --api_key "${LAGRANGE_API_KEY}" \
        --base_url "${BASE_URL}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=decompose_student_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 30
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=decompose_student_done | node=$(hostname)" >> "${MANIFEST}"