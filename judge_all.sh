#!/bin/bash
#SBATCH --job-name=judge_responses
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# judge_all.sh <input_file> -- Phase 2: judge correctness + compute
# semantic entropy for one generate_responses.py output file, using
# Qwen2.5-32B-Instruct as the entailment judge.
#
# --gres=gpu:a100:1 -- the 32B judge fits on one A100 80GB in bf16. Do NOT
# request 2 GPUs: device_map='auto' then splits the model across cards and
# cross-card overhead made judging hang (observed: 16h with ~zero progress).
#
# Usage:
#   sbatch judge_all.sh ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-14B_strict.jsonl
#   sbatch judge_all.sh ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-14B_fewshot.jsonl
#   sbatch judge_all.sh ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-4B-Instruct-2507_strict.jsonl
#   sbatch judge_all.sh ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-4B-Instruct-2507_fewshot.jsonl
#
# No --nodelist needed -- reads/writes under $HOME (network storage).

set -e
set -x

INPUT_FILE="${1:?Usage: sbatch judge_all.sh <input_file> [backend]}"
BACKEND="${2:-llm}"

if [[ "${BACKEND}" != "llm" && "${BACKEND}" != "deberta" ]]; then
    echo "ERROR: backend must be 'llm' or 'deberta', got '${BACKEND}'"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
if [ "${BACKEND}" == "deberta" ]; then
    OUTPUT_DIR="${PROJECT_DIR}/judged_data_seed44_deberta"
else
    OUTPUT_DIR="${PROJECT_DIR}/judged_data_seed44"
fi
JUDGE_MODEL="Qwen/Qwen3-32B"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
echo "===== input=${INPUT_FILE} ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_responses | input=${INPUT_FILE} | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python judge_responses.py \
    --input "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --entailment_backend "${BACKEND}" \
    --grader qwen \
    --grader_model Qwen/Qwen3-32B \
    --strict_entailment
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_responses_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 10
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_responses_done | input=${INPUT_FILE} | node=$(hostname)" >> "${MANIFEST}"