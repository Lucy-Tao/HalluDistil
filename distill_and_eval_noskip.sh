#!/bin/bash
#SBATCH --job-name=distill_eval_noskip
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# distill_and_eval_noskip.sh <prompt_style> [run_tag] [epochs]
#
# distill -> generate -> judge in one job. Full distillation only.
# Ablation of distill_and_eval_v3.sh: teacher NOT_ATTEMPTED targets are
# KEPT (--no_skip_abstain) instead of skipped, so the student also
# trains on the teacher's non-answers. Adds 5 items on strict and 6 on
# fewshot out of 500. Everything else is identical to v3.
#
# Usage:
#   sbatch distill_and_eval_noskip.sh strict
#   sbatch distill_and_eval_noskip.sh fewshot myrun 5

set -e
set -x

PROMPT_STYLE="${1:?Usage: sbatch distill_and_eval_noskip.sh <strict|fewshot> [run_tag] [epochs]}"
RUN_TAG="${2:-}"
EPOCHS="${3:-3}"

# Judging needs about 66G against 12G for distillation and generation,
# and SLURM accounts for GPUs without partitioning device memory, so an
# all-in-one job either hangs during model load or dies partway. Judge
# separately with judge_se.sh. RUN_JUDGE=1 restores the old behaviour.
RUN_JUDGE="${RUN_JUDGE:-0}"

if [[ "${PROMPT_STYLE}" != "strict" && "${PROMPT_STYLE}" != "fewshot" ]]; then
    echo "ERROR: prompt_style must be 'strict' or 'fewshot', got '${PROMPT_STYLE}'"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
SUBSET_FILE="${PROJECT_DIR}/subset_500_seed44_question_indices.json"
# Teacher 32B judged file (seed44) — source of low_temp targets.
JUDGED_TEACHER_FILE="${PROJECT_DIR}/judged_data_seed44_deberta/judged_simpleqa_Qwen3-32B_${PROMPT_STYLE}.jsonl"

if [ -n "${RUN_TAG}" ]; then
    MODEL_TAG="${PROMPT_STYLE}_lowtemp_seed44_noskip_${RUN_TAG}"
else
    MODEL_TAG="${PROMPT_STYLE}_lowtemp_seed44_noskip"
fi
STUDENT_SHORT="Qwen3-4B-Instruct-2507"
DISTILLED_MODEL_PATH="/scratch-ssd/ms25yt/models/${DATASET}_${STUDENT_SHORT}_student_${MODEL_TAG}"
JUDGE_MODEL="Qwen/Qwen3-32B"
GEN_OUTPUT_DIR="${PROJECT_DIR}/gen_data_distilled_seed44_noskip"
JUDGED_OUTPUT_DIR="${PROJECT_DIR}/judged_data_distilled_seed44_noskip"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"
HYPERPARAM_LOG="${PROJECT_DIR}/logs/hyperparameter_log.jsonl"

cd "${PROJECT_DIR}"
mkdir -p logs "${GEN_OUTPUT_DIR}" "${JUDGED_OUTPUT_DIR}"

source ~/.bashrc
conda activate haldist

# Prefer a node-local personal cache when present, else the group cache
# from .bashrc. Lets one script run on nodes where the group cache is
# incomplete.
# Switch to the personal cache only when it actually holds the student
# weights. Testing that the directory exists is not enough: a node can
# have an empty hub/ left behind after a move, or hold an unrelated
# model, in which case switching hides a complete group cache.
if ls /scratch-ssd/ms25yt/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/*.safetensors >/dev/null 2>&1; then
    export HF_HOME=/scratch-ssd/ms25yt/hf
fi
echo "HF_HOME=${HF_HOME} host=$(hostname)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "===== [$(date)] host=$(hostname) prompt_style=${PROMPT_STYLE} model_tag=${MODEL_TAG} epochs=${EPOCHS} ====="
echo "===== distilled model -> ${DISTILLED_MODEL_PATH} ====="
nvidia-smi

if [ ! -f "${SUBSET_FILE}" ]; then
    echo "ERROR: subset file not found: ${SUBSET_FILE}"
    exit 1
fi
if [ ! -f "${JUDGED_TEACHER_FILE}" ]; then
    echo "ERROR: teacher judged file not found: ${JUDGED_TEACHER_FILE}"
    exit 1
fi

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_and_eval | prompt_style=${PROMPT_STYLE} | node=$(hostname) | model_path=${DISTILLED_MODEL_PATH}" >> "${MANIFEST}"

# Effective hyperparameters = config.py + epochs override.
CURRENT_HYPERPARAMS=$(EPOCHS_OVERRIDE="${EPOCHS}" python3 -c "
import json, os
from config import cfg
ep = os.environ.get('EPOCHS_OVERRIDE', '')
print(json.dumps({
    'num_epochs': int(ep) if ep else cfg.num_epochs,
    'batch_size': cfg.batch_size,
    'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
    'learning_rate': cfg.learning_rate,
}, sort_keys=True))
")
echo "Effective hyperparameters: ${CURRENT_HYPERPARAMS}"

# ── Step 1: Distill (full, low_temp target) ──────────────────
if [ -d "${DISTILLED_MODEL_PATH}" ] && [ -n "$(ls -A "${DISTILLED_MODEL_PATH}" 2>/dev/null)" ]; then
    # Reuse only if logged hyperparameters match.
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
        echo "ERROR: model exists but no hyperparameter log entry. Delete it or pick a new run_tag."
        exit 1
    elif [ "${LAST_LOGGED}" != "${CURRENT_HYPERPARAMS}" ]; then
        echo "ERROR: model exists with different hyperparameters."
        echo "  Logged:  ${LAST_LOGGED}"
        echo "  Current: ${CURRENT_HYPERPARAMS}"
        exit 1
    fi
    echo "===== [$(date)] STEP 1/3: SKIPPED — model exists, hyperparameters match ====="
else
    echo "===== [$(date)] STEP 1/3: Full distillation ====="
    python run.py --mode distill \
        --no_skip_abstain \
        --judged_file "${JUDGED_TEACHER_FILE}" \
        --model_tag "${MODEL_TAG}" \
        --epochs "${EPOCHS}"
    python3 -c "
import json
from datetime import datetime, timezone
rec = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'model_tag': '${MODEL_TAG}',
    'prompt_style': '${PROMPT_STYLE}',
    'run_tag': '${RUN_TAG}',
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
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_done | prompt_style=${PROMPT_STYLE} | node=$(hostname)" >> "${MANIFEST}"

# ── Step 2: Generate from distilled model ────────────────────
echo "===== [$(date)] STEP 2/3: Generating from distilled model ====="
python generate_responses.py \
    --model_role student \
    --model_name "${DISTILLED_MODEL_PATH}" \
    --dataset "${DATASET}" \
    --prompt_style "${PROMPT_STYLE}" \
    --question_indices_file "${SUBSET_FILE}" \
    --n_high_temp_samples 10 \
    --output_dir "${GEN_OUTPUT_DIR}"

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_gen_done | prompt_style=${PROMPT_STYLE} | node=$(hostname)" >> "${MANIFEST}"

# ── Step 3: Judge (deberta cluster + qwen SimpleQA grader) ────
MODEL_SHORT=$(basename "${DISTILLED_MODEL_PATH}")
GEN_FILE="${GEN_OUTPUT_DIR}/gen_${DATASET}_${MODEL_SHORT}_${PROMPT_STYLE}.jsonl"
if [ ! -f "${GEN_FILE}" ]; then
    echo "ERROR: expected generation output not found: ${GEN_FILE}"
    exit 1
fi
if [ "${RUN_JUDGE}" != "1" ]; then
    echo "===== [$(date)] STEP 3/3: SKIPPED, judge separately ====="
    echo "GEN_FILE=${GEN_FILE}"
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
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_and_eval_done | prompt_style=${PROMPT_STYLE} | node=$(hostname) | model_path=${DISTILLED_MODEL_PATH}" >> "${MANIFEST}"
echo "Distilled model:  ${DISTILLED_MODEL_PATH}"
echo "Generated data:   ${GEN_FILE}"
echo "Judged data:      ${JUDGED_OUTPUT_DIR}/"