#!/bin/bash
#SBATCH --job-name=entropy
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage:
#   sbatch --job-name=entropy_Qwen3-32B compute_entropy.sh Qwen3-32B teacher
#   sbatch --job-name=entropy_student   compute_entropy.sh Qwen3-4B-Instruct-2507 student
#   sbatch --job-name=entropy_ep3       compute_entropy.sh distilled_ep3 student /scratch-ssd/ms25yt/models/factscore_bio_distilled_student_ep3
#
# Args:
#   1 MODEL_TAG  : the tag in factscore_<TAG>.jsonl (input) and entropy_<TAG>.jsonl (output)
#   2 MODEL_ROLE : teacher | student  (distilled -> student + arg 3)
#   3 OVERRIDE   : (optional) checkpoint path for a distilled student
#
# Pinned to oat16: has Qwen3-32B / Qwen3-4B / deberta cached, and the
# distilled checkpoints were trained there (node-local scratch-ssd). If a
# distilled checkpoint lives on another node, change --nodelist to match
# the DISTILLED_CHECKPOINT_NODE line in the manifest.

set -e
set -x

export WANDB_MODE=online

MODEL_TAG="$1"
MODEL_ROLE="$2"
OVERRIDE="$3"

if [ -z "${MODEL_TAG}" ] || [ -z "${MODEL_ROLE}" ]; then
    echo "ERROR: usage: compute_entropy.sh MODEL_TAG MODEL_ROLE [CHECKPOINT_OVERRIDE]"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
GEN_DATA_DIR="${PROJECT_DIR}/gen_longform_data"
INPUT="${GEN_DATA_DIR}/factscore_${MODEL_TAG}.jsonl"
OUTPUT="${GEN_DATA_DIR}/entropy_${MODEL_TAG}.jsonl"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

if [ ! -f "${INPUT}" ]; then
    echo "ERROR: input not found: ${INPUT}"
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p logs

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=entropy_${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

# Build the optional --model_name_override flag
OVERRIDE_FLAG=""
if [ -n "${OVERRIDE}" ]; then
    OVERRIDE_FLAG="--model_name_override ${OVERRIDE}"
fi

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python compute_claim_entropy.py \
        --input "${INPUT}" \
        --model_role "${MODEL_ROLE}" \
        --output "${OUTPUT}" \
        --judge_backend deberta \
        --wandb_project halludistil-longform \
        --wandb_run_name "entropy_${MODEL_TAG}" \
        ${OVERRIDE_FLAG}
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=entropy_${MODEL_TAG}_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=entropy_${MODEL_TAG}_done | node=$(hostname)" >> "${MANIFEST}"