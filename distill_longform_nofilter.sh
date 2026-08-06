#!/bin/bash
#SBATCH --job-name=distill_longform
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage:
#   sbatch --nodelist=oat16 --job-name=distill_ep3  distill_longform.sh 3
#   sbatch --nodelist=oat16 --job-name=distill_ep20 distill_longform.sh 20
#
# One positional arg: EPOCHS. Single teacher (Qwen3-32B). Distills, then
# generates on the same node in the same job so the node-local scratch-ssd
# checkpoint stays readable.

set -e
set -x

export WANDB_MODE=disabled

EPOCHS="$1"
if ! [[ "${EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: argument must be an integer epoch count, got '${EPOCHS}'."
    exit 1
fi

TEACHER_MODEL="Qwen3-32B"

PROJECT_DIR=~/SimpleQA
GEN_DATA_DIR="${PROJECT_DIR}/gen_longform_data"
DISTILL_TARGETS="${GEN_DATA_DIR}/distill_targets_nofilter.jsonl"
TEACHER_GEN="${GEN_DATA_DIR}/gen_factscore_bio_${TEACHER_MODEL}.jsonl"
DISTILLED_MODEL_DIR="/scratch-ssd/ms25yt/models/factscore_bio_distilled_student_nofilter_ep${EPOCHS}"
GEN_OUTPUT_DIR=~/SimpleQA/gen_longform_distilled
RUN_TAG="distilled_nofilter_ep${EPOCHS}"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

if [ ! -f "${DISTILL_TARGETS}" ]; then
    echo "ERROR: ${DISTILL_TARGETS} not found."
    exit 1
fi
if [ ! -f "${TEACHER_GEN}" ]; then
    echo "ERROR: ${TEACHER_GEN} not found."
    exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p logs "${GEN_OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
nvidia-smi
NODE=$(hostname)
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_ep${EPOCHS} | node=${NODE}" >> "${MANIFEST}"
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | DISTILLED_CHECKPOINT_NODE=${NODE} | path=${DISTILLED_MODEL_DIR}" >> "${MANIFEST}"

# Step 1: Distillation
MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Distill attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python distill_longform.py \
        --distill_targets "${DISTILL_TARGETS}" \
        --teacher_gen "${TEACHER_GEN}" \
        --output_dir "${DISTILLED_MODEL_DIR}" \
        --num_epochs "${EPOCHS}" \
        --manifest_copy_to "${GEN_DATA_DIR}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Distill attempt ${attempt} succeeded."
        break
    fi
    echo "Distill attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_ep${EPOCHS}_retry | attempt=${attempt} | exit_code=${exit_code} | node=${NODE}" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} distill attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_ep${EPOCHS}_done | node=${NODE}" >> "${MANIFEST}"

# Step 2: Generate distilled-student responses (same node, same job)
echo "===== [$(date)] Starting distilled-student generation on ${NODE} ====="
MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Generate attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python generate_longform_responses.py \
        --model_role student \
        --n_samples 183 \
        --model_name "${DISTILLED_MODEL_DIR}" \
        --run_tag "${RUN_TAG}" \
        --question_idx_subset "${DISTILL_TARGETS}" \
        --output_dir "${GEN_OUTPUT_DIR}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Generate attempt ${attempt} succeeded."
        break
    fi
    echo "Generate attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=gen_distilled_ep${EPOCHS}_retry | attempt=${attempt} | exit_code=${exit_code} | node=${NODE}" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} generate attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_ep${EPOCHS}_and_generate_done | node=${NODE}" >> "${MANIFEST}"
echo "ep${EPOCHS} distilled checkpoint lives on node ${NODE} at ${DISTILLED_MODEL_DIR}"