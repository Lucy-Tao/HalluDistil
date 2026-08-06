#!/bin/bash
#SBATCH --job-name=judge_llama
#SBATCH --partition=msc
#SBATCH --nodelist=oat11
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# judge_llama.sh <input_file> [backend]
#
# Judge stays Qwen3-32B for both model families, so the grading standard is
# held constant and the two lines stay comparable.
#
# Single GPU on purpose. device_map='auto' across 2 cards made judging hang
# (~16h, near-zero progress).
#
# No --nodelist: this loads no Llama weights, only Qwen3-32B plus a jsonl,
# so any node with a complete Qwen3-32B cache works.
#
# Usage:
#   sbatch judge_llama.sh ~/SimpleQA/gen_data_seed44_llama/gen_simpleqa_Llama-3.1-8B-Instruct_strict.jsonl deberta

set -e
set -x

INPUT_FILE="${1:?Usage: sbatch judge_llama.sh <input_file> [backend]}"
BACKEND="${2:-deberta}"

if [[ "${BACKEND}" != "llm" && "${BACKEND}" != "deberta" ]]; then
    echo "ERROR: backend must be llm or deberta, got '${BACKEND}'"
    exit 1
fi
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: input file not found: ${INPUT_FILE}"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
if [ "${BACKEND}" == "deberta" ]; then
    OUTPUT_DIR="${PROJECT_DIR}/judged_data_seed44_llama_deberta"
else
    OUTPUT_DIR="${PROJECT_DIR}/judged_data_seed44_llama"
fi
JUDGE_MODEL="Qwen/Qwen3-32B"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

export HF_HOME=/scratch-ssd/oatml/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch-ssd/ms25yt/datasets

echo "===== [$(date)] host=$(hostname) input=${INPUT_FILE} ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_llama | input=${INPUT_FILE} | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python -u judge_responses.py \
        --input "${INPUT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --entailment_backend "${BACKEND}" \
        --grader qwen \
        --grader_model "${JUDGE_MODEL}" \
        --strict_entailment
    exit_code=$?
    set -e
    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed, exit ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=judge_llama_retry | attempt=${attempt} | exit=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts."
        exit "${exit_code}"
    fi
    sleep 60
done

echo "===== [$(date)] Done. Output in ${OUTPUT_DIR} ====="
