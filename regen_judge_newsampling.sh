#!/bin/bash
#SBATCH --job-name=regen_judge_newsampling
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat12
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# regen_judge_newsampling.sh <model_tag> -- regenerate with fixed sampling
# params (top_p=0.8, top_k=20, no repetition_penalty) from an existing
# checkpoint, then judge. Old gen/judged files must be renamed first.
set -e
set -x
MODEL_TAG="${1:?Usage: sbatch regen_judge_newsampling.sh <model_tag e.g. strict_full_epochs5>}"
PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
cd "${PROJECT_DIR}"
source ~/.bashrc
conda activate haldist

MODEL_PATH="/scratch-ssd/ms25yt/models/simpleqa_Qwen3-4B-Instruct-2507_student_${MODEL_TAG}"
GEN_OUTPUT_DIR="${PROJECT_DIR}/gen_data_distilled"
JUDGED_OUTPUT_DIR="${PROJECT_DIR}/judged_data_distilled"
JUDGE_MODEL="Qwen/Qwen2.5-32B-Instruct"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

echo "===== [$(date)] host=$(hostname) model=${MODEL_PATH} ====="
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: checkpoint not found on this node: ${MODEL_PATH}"
    exit 1
fi

# Step 1: generate with the fixed sampling parameters
python generate_responses.py \
    --model_role student \
    --model_name "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --prompt_style strict \
    --question_indices_file "${PROJECT_DIR}/subset_500_question_indices.json" \
    --n_high_temp_samples 10 \
    --output_dir "${GEN_OUTPUT_DIR}"
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=regen_newsampling_gen_done | model_tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

# Step 2: judge
MODEL_SHORT=$(basename "${MODEL_PATH}")
GEN_FILE="${GEN_OUTPUT_DIR}/gen_${DATASET}_${MODEL_SHORT}_strict.jsonl"
if [ ! -f "${GEN_FILE}" ]; then
    echo "ERROR: generation output not found: ${GEN_FILE}"
    exit 1
fi
python judge_responses.py \
    --input "${GEN_FILE}" \
    --output_dir "${JUDGED_OUTPUT_DIR}" \
    --entailment_backend llm \
    --entailment_llm_model_name "${JUDGE_MODEL}" \
    --strict_entailment

echo "===== [$(date)] done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=regen_newsampling_done | model_tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"