#!/bin/bash
#SBATCH --job-name=repair_degenerate
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# repair_degenerate.sh <file> <model_name> -- regenerate only the small
# number of genuinely degenerate (repetition-loop) fields in an existing
# generate_responses.py checkpoint file, using the fixed sample_responses()
# (repetition_penalty=1.3 added).
#
# Usage:
#   sbatch repair_degenerate.sh ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-4B-Instruct-2507_fewshot.jsonl Qwen/Qwen3-4B-Instruct-2507
#   sbatch repair_degenerate.sh ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-4B-Instruct-2507_strict.jsonl  Qwen/Qwen3-4B-Instruct-2507
#   sbatch repair_degenerate.sh ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-14B_fewshot.jsonl              Qwen/Qwen3-14B
#   sbatch repair_degenerate.sh ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-14B_strict.jsonl               Qwen/Qwen3-14B
#
# Only ~2-45 records need fixing per file, so this is quick: one model
# load + a handful of short generations, well under the 1-hour ceiling.
#
# No --nodelist needed -- reads/writes the checkpoint file in $HOME
# (network storage), not scratch-ssd.

set -e
set -x

FILE="${1:?Usage: sbatch repair_degenerate.sh <file> <model_name>}"
MODEL_NAME="${2:?Usage: sbatch repair_degenerate.sh <file> <model_name>}"

PROJECT_DIR=~/SimpleQA
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
echo "===== file=${FILE} model=${MODEL_NAME} ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=repair_degenerate | file=${FILE} | node=$(hostname)" >> "${MANIFEST}"

python repair_degenerate_responses.py \
    --file "${FILE}" \
    --model_name "${MODEL_NAME}"

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=repair_degenerate_done | file=${FILE} | node=$(hostname)" >> "${MANIFEST}"