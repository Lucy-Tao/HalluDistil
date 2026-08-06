#!/bin/bash
#SBATCH --job-name=decompose_distilled_ep5
#SBATCH --partition=msc
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# decompose_distilled_ep5.sh -- decompose + verify the ep5 distilled
# student's 173 responses. No --gres=gpu -- pure API/network I/O through
# Oxford's Lagrange gateway, same as decompose_student.sh.
#
# ADAPTER STEP: generate_longform_responses.py's raw output has fields
# {question_idx, entity, prompt, response} -- but decompose_and_verify.py
# expects a "{model_role}_response" field (e.g. "distilled_student_response"
# for --model_role distilled_student), matching answered_both.jsonl's
# schema (teacher_response / student_response). This script converts the
# raw generation file into that expected shape first, writing a small
# adapted copy -- it does NOT modify the original generation file.

set -e
set -x

PROJECT_DIR=~/SimpleQA
GEN_DATA_DIR="${PROJECT_DIR}/gen_longform_data"
RAW_INPUT="${GEN_DATA_DIR}/gen_factscore_bio_factscore_bio_distilled_student_ep5_distilled_ep5.jsonl"
ADAPTED_INPUT="${GEN_DATA_DIR}/answered_distilled_ep5.jsonl"
OUTPUT="${GEN_DATA_DIR}/claims_distilled_ep5.jsonl"
MODEL="gpt-5-mini"
BASE_URL="https://lagrange.uksouth.cloudapp.azure.com/openai"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=decompose_distilled_ep5 | node=$(hostname)" >> "${MANIFEST}"

if [ -z "${LAGRANGE_API_KEY}" ]; then
    echo "ERROR: LAGRANGE_API_KEY is not set. Export it in ~/.bashrc and resubmit."
    exit 1
fi

# ── Adapter: rename "response" -> "distilled_student_response" ──────
python3 -c "
import json
with open('${RAW_INPUT}') as f_in, open('${ADAPTED_INPUT}', 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        f_out.write(json.dumps({
            'question_idx': rec['question_idx'],
            'entity': rec['entity'],
            'prompt': rec['prompt'],
            'distilled_student_response': rec['response'],
        }, ensure_ascii=False) + '\n')
print('Adapted', '${RAW_INPUT}', '->', '${ADAPTED_INPUT}')
"

# ── Decompose + verify ────────────────────────────────────────────
MAX_RETRIES=20
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python decompose_and_verify.py \
        --input "${ADAPTED_INPUT}" \
        --model_role distilled_student \
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
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=decompose_distilled_ep5_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 30
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=decompose_distilled_ep5_done | node=$(hostname)" >> "${MANIFEST}"