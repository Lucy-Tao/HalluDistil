#!/bin/bash
#SBATCH --job-name=regen_distilled
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat15
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Patch: regenerate distilled-student bios over ALL 100 entities (not the
# 97 distill_targets), using the existing checkpoint on oat15. Does NOT
# retrain. One positional arg: EPOCHS.
#
#   sbatch regen_distilled_100.sh 5
#
# Pinned to oat15 because the checkpoint lives on its node-local scratch-ssd.

set -e
set -x
export WANDB_MODE=disabled

EPOCHS="$1"
if ! [[ "${EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: pass an integer epoch count."
    exit 1
fi

PROJECT_DIR=~/SimpleQA
GEN_DATA_DIR="${PROJECT_DIR}/gen_longform_data"
ALL_100="${GEN_DATA_DIR}/sampled_100_entities.jsonl"
DISTILLED_MODEL_DIR="/scratch-ssd/ms25yt/models/factscore_bio_distilled_student_ep${EPOCHS}"
GEN_OUTPUT_DIR=~/SimpleQA/gen_longform_distilled
RUN_TAG="distilled_ep${EPOCHS}"

if [ ! -d "${DISTILLED_MODEL_DIR}" ]; then
    echo "ERROR: checkpoint not found on this node: ${DISTILLED_MODEL_DIR}"
    echo "  (are you on the node where it was trained? check the manifest.)"
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p logs "${GEN_OUTPUT_DIR}"
source ~/.bashrc
conda activate haldist

echo "===== [$(date)] regen ep${EPOCHS} on $(hostname), 100 entities ====="
nvidia-smi

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python generate_longform_responses.py \
        --model_role student \
        --n_samples 183 \
        --model_name "${DISTILLED_MODEL_DIR}" \
        --run_tag "${RUN_TAG}" \
        --question_idx_subset "${ALL_100}" \
        --output_dir "${GEN_OUTPUT_DIR}"
    exit_code=$?
    set -e
    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed (exit ${exit_code})."
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] regen ep${EPOCHS} done ====="