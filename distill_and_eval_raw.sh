#!/bin/bash
#SBATCH --job-name=distill_eval_raw
#SBATCH --partition=msc
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/ms25yt/SimpleQA/logs/%x_%j.out
#SBATCH --error=/users/ms25yt/SimpleQA/logs/%x_%j.err
#
# distill_and_eval_raw.sh <prompt_style> <n_samples> [epochs]
#
# Multi-sample target variant. Targets come from raw_responses[:k]
# (T=1.0) instead of the single low_temp_response (T=0.1), so this
# changes three things at once against the baseline: sample count,
# sampling temperature, and the number of gradient steps per epoch.
# Disentangling them needs three points per prompt style:
#
#   raw 1, ep20   temperature control, one T=1.0 target
#   raw 5, ep20   epoch aligned, 5x the gradient steps
#   raw 5, ep4    gradient step aligned, 5 x 4 = 1 x 20
#
# Judging is a separate job, see judge_se.sh. Set RUN_JUDGE=1 for the
# old all-in-one behaviour.
#
# Usage:
#   sbatch distill_and_eval_raw.sh strict 5 20

set -e
set -x

PROMPT_STYLE="${1:?Usage: sbatch distill_and_eval_raw.sh <strict|fewshot> <n_samples> [epochs]}"
RAW_SAMPLES="${2:?Usage: sbatch distill_and_eval_raw.sh <strict|fewshot> <n_samples> [epochs]}"
EPOCHS="${3:-20}"
RUN_JUDGE="${RUN_JUDGE:-0}"

if [[ "${PROMPT_STYLE}" != "strict" && "${PROMPT_STYLE}" != "fewshot" ]]; then
    echo "ERROR: prompt_style must be 'strict' or 'fewshot', got '${PROMPT_STYLE}'"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
SUBSET_FILE="${PROJECT_DIR}/subset_500_seed44_question_indices.json"
JUDGED_TEACHER_FILE="${PROJECT_DIR}/judged_data_seed44_deberta/judged_simpleqa_Qwen3-32B_${PROMPT_STYLE}.jsonl"

MODEL_TAG="${PROMPT_STYLE}_lowtemp_seed44_raw${RAW_SAMPLES}_ep${EPOCHS}"
STUDENT_SHORT="Qwen3-4B-Instruct-2507"
DISTILLED_MODEL_PATH="/scratch-ssd/ms25yt/models/${DATASET}_${STUDENT_SHORT}_student_${MODEL_TAG}"
GEN_OUTPUT_DIR="${PROJECT_DIR}/gen_data_distilled_seed44_raw"
JUDGED_OUTPUT_DIR="${PROJECT_DIR}/judged_data_distilled_seed44_raw"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"
HYPERPARAM_LOG="${PROJECT_DIR}/logs/hyperparameter_log.jsonl"
JUDGE_MODEL="Qwen/Qwen3-32B"

cd "${PROJECT_DIR}"
mkdir -p logs "${GEN_OUTPUT_DIR}" "${JUDGED_OUTPUT_DIR}"

set +x
source ~/.bashrc
conda activate haldist
set -x

# Switch to the personal cache only when it actually holds the student
# weights. Testing that the directory exists is not enough: a node can
# have an empty hub/ left behind after a move, or hold an unrelated
# model, in which case switching hides a complete group cache.
if ls /scratch-ssd/ms25yt/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/*.safetensors >/dev/null 2>&1; then
    export HF_HOME=/scratch-ssd/ms25yt/hf
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "===== [$(date)] host=$(hostname) HF_HOME=${HF_HOME} style=${PROMPT_STYLE} raw=${RAW_SAMPLES} epochs=${EPOCHS} ====="
echo "===== distilled model -> ${DISTILLED_MODEL_PATH} ====="
nvidia-smi

if [ ! -f "${SUBSET_FILE}" ]; then
    echo "ERROR: subset file not found: ${SUBSET_FILE}"; exit 1
fi
if [ ! -f "${JUDGED_TEACHER_FILE}" ]; then
    echo "ERROR: teacher judged file not found: ${JUDGED_TEACHER_FILE}"; exit 1
fi

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_and_eval_raw | style=${PROMPT_STYLE} | raw_samples=${RAW_SAMPLES} | epochs=${EPOCHS} | node=$(hostname) | model_path=${DISTILLED_MODEL_PATH}" >> "${MANIFEST}"

CURRENT_HYPERPARAMS=$(EPOCHS_OVERRIDE="${EPOCHS}" RAW_N="${RAW_SAMPLES}" python3 -c "
import json, os
from config import cfg
ep = os.environ.get('EPOCHS_OVERRIDE', '')
print(json.dumps({
    'num_epochs': int(ep) if ep else cfg.num_epochs,
    'batch_size': cfg.batch_size,
    'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
    'learning_rate': cfg.learning_rate,
    'raw_samples': int(os.environ['RAW_N']),
}, sort_keys=True))
")
echo "Effective hyperparameters: ${CURRENT_HYPERPARAMS}"

# ── Step 1: Distill ──────────────────────────────────────────
if [ -d "${DISTILLED_MODEL_PATH}" ] && [ -n "$(ls -A "${DISTILLED_MODEL_PATH}" 2>/dev/null)" ]; then
    LAST_LOGGED=$(python3 -c "
import json
model_tag = '${MODEL_TAG}'
last = None
try:
    with open('${HYPERPARAM_LOG}') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get('model_tag') == model_tag:
                last = rec
except FileNotFoundError:
    pass
print('NO_RECORD' if last is None else json.dumps(last.get('hyperparameters', {}), sort_keys=True))
")
    if [ "${LAST_LOGGED}" == "NO_RECORD" ]; then
        echo "ERROR: model exists but no hyperparameter log entry. Delete it or pick a new tag."
        exit 1
    elif [ "${LAST_LOGGED}" != "${CURRENT_HYPERPARAMS}" ]; then
        echo "ERROR: model exists with different hyperparameters."
        echo "  Logged:  ${LAST_LOGGED}"
        echo "  Current: ${CURRENT_HYPERPARAMS}"
        exit 1
    fi
    echo "===== [$(date)] STEP 1/3: SKIPPED, model exists ====="
else
    echo "===== [$(date)] STEP 1/3: Distillation with raw_samples=${RAW_SAMPLES} ====="
    python run.py --mode distill \
        --judged_file "${JUDGED_TEACHER_FILE}" \
        --raw_samples "${RAW_SAMPLES}" \
        --model_tag "${MODEL_TAG}" \
        --epochs "${EPOCHS}"
    python3 -c "
import json
from datetime import datetime, timezone
rec = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'model_tag': '${MODEL_TAG}',
    'prompt_style': '${PROMPT_STYLE}',
    'raw_samples': ${RAW_SAMPLES},
    'job_id': '${SLURM_JOB_ID:-none}',
    'hyperparameters': ${CURRENT_HYPERPARAMS},
}
with open('${HYPERPARAM_LOG}', 'a') as f:
    f.write(json.dumps(rec) + '\n')
"
fi

if [ ! -d "${DISTILLED_MODEL_PATH}" ]; then
    echo "ERROR: distillation did not produce a model at ${DISTILLED_MODEL_PATH}"
    exit 1
fi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_raw_done | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

# ── Step 2: Generate ─────────────────────────────────────────
echo "===== [$(date)] STEP 2/3: Generating ====="
python generate_responses.py \
    --model_role student \
    --model_name "${DISTILLED_MODEL_PATH}" \
    --dataset "${DATASET}" \
    --prompt_style "${PROMPT_STYLE}" \
    --question_indices_file "${SUBSET_FILE}" \
    --n_high_temp_samples 10 \
    --output_dir "${GEN_OUTPUT_DIR}"

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_raw_gen_done | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

# ── Step 3: Judge (separate job by default) ──────────────────
MODEL_SHORT=$(basename "${DISTILLED_MODEL_PATH}")
GEN_FILE="${GEN_OUTPUT_DIR}/gen_${DATASET}_${MODEL_SHORT}_${PROMPT_STYLE}.jsonl"
if [ ! -f "${GEN_FILE}" ]; then
    echo "ERROR: expected generation output not found: ${GEN_FILE}"
    exit 1
fi

if [ "${RUN_JUDGE}" != "1" ]; then
    echo "===== [$(date)] STEP 3/3: SKIPPED, judge separately ====="
    echo "GEN_FILE=${GEN_FILE}"
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_raw_gen_only | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"
    exit 0
fi

echo "===== [$(date)] STEP 3/3: Judging (${GEN_FILE}) ====="
python judge_responses.py \
    --input "${GEN_FILE}" \
    --output_dir "${JUDGED_OUTPUT_DIR}" \
    --entailment_backend deberta \
    --grader qwen \
    --grader_model "${JUDGE_MODEL}" \
    --strict_entailment

echo "===== [$(date)] All 3 steps done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_and_eval_raw_done | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"
