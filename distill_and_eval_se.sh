#!/bin/bash
#SBATCH --job-name=distill_eval_se
#SBATCH --partition=msc
#SBATCH --nodes=1
#SBATCH --nodelist=oat10,oat11
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# distill_and_eval_se.sh <prompt_style> <se_level> <se_mode> [epochs]
#
# Semantic-entropy intervention variant of distill_and_eval_v3.sh.
# Items whose teacher semantic entropy exceeds the threshold are either
# dropped (filter) or have their target swapped for "Unknown" (replace).
# Everything else — subset, teacher file, judge, sampling — matches v3.
#
# Thresholds sit at the MIDPOINT between adjacent realisable entropy
# values. Entropy here is discrete (cluster frequencies over 10 samples)
# and two runs can differ in the last bit, so a threshold placed ON a
# realisable value is decided by floating-point noise. See
# analyse_teacher_entropy.py for the cut table these came from.
#
#   loose  = 2.2333   strict: filter 404 / replace 91 swapped
#   medium = 1.9992   strict: filter 308 / replace 187 swapped
#   tight  = 1.7219   strict: filter 234 / replace 261 swapped
#
# NOTE on evaluating replace runs: the student learns to emit "Unknown",
# which the grader marks NOT_ATTEMPTED, which eval_metrics.py drops by
# default. That shrinks the AUROC denominator, so the default AUROC is
# NOT comparable across modes. Report --keep-not-attempted AUROC, AURAC
# (computed over all items), and the abstention rate alongside it.
#
# Usage:
#   sbatch distill_and_eval_se.sh strict loose filter 20
#   sbatch distill_and_eval_se.sh fewshot tight replace 20

set -e
set -x

PROMPT_STYLE="${1:?Usage: sbatch distill_and_eval_se.sh <strict|fewshot> <loose|medium|tight> <filter|replace> [epochs]}"
SE_LEVEL="${2:?Usage: sbatch distill_and_eval_se.sh <strict|fewshot> <loose|medium|tight> <filter|replace> [epochs]}"
SE_MODE="${3:?Usage: sbatch distill_and_eval_se.sh <strict|fewshot> <loose|medium|tight> <filter|replace> [epochs]}"
EPOCHS="${4:-20}"

# Judging needs about 66G against 12G for the first two stages, so it
# runs as a separate job by default. Set RUN_JUDGE=1 for the old
# all-in-one behaviour.
RUN_JUDGE="${RUN_JUDGE:-0}"

if [[ "${PROMPT_STYLE}" != "strict" && "${PROMPT_STYLE}" != "fewshot" ]]; then
    echo "ERROR: prompt_style must be 'strict' or 'fewshot', got '${PROMPT_STYLE}'"
    exit 1
fi
if [[ "${SE_MODE}" != "filter" && "${SE_MODE}" != "replace" ]]; then
    echo "ERROR: se_mode must be 'filter' or 'replace', got '${SE_MODE}'"
    exit 1
fi

case "${SE_LEVEL}" in
    loose)  SE_THRESHOLD=2.2333 ;;
    medium) SE_THRESHOLD=1.9992 ;;
    tight)  SE_THRESHOLD=1.7219 ;;
    *) echo "ERROR: se_level must be 'loose', 'medium' or 'tight', got '${SE_LEVEL}'"; exit 1 ;;
esac

PROJECT_DIR=~/SimpleQA
DATASET=simpleqa
ABSTAIN_STRING="Unknown"
SUBSET_FILE="${PROJECT_DIR}/subset_500_seed44_question_indices.json"
JUDGED_TEACHER_FILE="${PROJECT_DIR}/judged_data_seed44_deberta/judged_simpleqa_Qwen3-32B_${PROMPT_STYLE}.jsonl"

MODEL_TAG="${PROMPT_STYLE}_lowtemp_seed44_se_${SE_LEVEL}_${SE_MODE}_ep${EPOCHS}"
STUDENT_SHORT="Qwen3-4B-Instruct-2507"
DISTILLED_MODEL_PATH="/scratch-ssd/ms25yt/models/${DATASET}_${STUDENT_SHORT}_student_${MODEL_TAG}"
JUDGE_MODEL="Qwen/Qwen3-32B"

# Separate directories per mode so filter and replace results never mix.
GEN_OUTPUT_DIR="${PROJECT_DIR}/gen_data_distilled_seed44_se_${SE_MODE}"
JUDGED_OUTPUT_DIR="${PROJECT_DIR}/judged_data_distilled_seed44_se_${SE_MODE}"
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

echo "===== [$(date)] host=$(hostname) style=${PROMPT_STYLE} level=${SE_LEVEL} mode=${SE_MODE} threshold=${SE_THRESHOLD} epochs=${EPOCHS} ====="
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

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_and_eval_se | style=${PROMPT_STYLE} | level=${SE_LEVEL} | mode=${SE_MODE} | threshold=${SE_THRESHOLD} | node=$(hostname) | model_path=${DISTILLED_MODEL_PATH}" >> "${MANIFEST}"

# Effective hyperparameters = config.py + epochs + intervention settings.
# The threshold and mode are part of the identity of this run, so a
# checkpoint trained at one threshold must never be reused for another.
CURRENT_HYPERPARAMS=$(EPOCHS_OVERRIDE="${EPOCHS}" SE_T="${SE_THRESHOLD}" SE_M="${SE_MODE}" python3 -c "
import json, os
from config import cfg
ep = os.environ.get('EPOCHS_OVERRIDE', '')
print(json.dumps({
    'num_epochs': int(ep) if ep else cfg.num_epochs,
    'batch_size': cfg.batch_size,
    'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
    'learning_rate': cfg.learning_rate,
    'se_threshold': float(os.environ['SE_T']),
    'se_mode': os.environ['SE_M'],
}, sort_keys=True))
")
echo "Effective hyperparameters: ${CURRENT_HYPERPARAMS}"

# ── Step 1: Distill (SE intervention applied inside distill.py) ──
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
    echo "===== [$(date)] STEP 1/3: SKIPPED — model exists, hyperparameters match ====="
else
    echo "===== [$(date)] STEP 1/3: Distillation with SE ${SE_MODE} @ ${SE_THRESHOLD} ====="
    python run.py --mode distill \
        --judged_file "${JUDGED_TEACHER_FILE}" \
        --se_threshold "${SE_THRESHOLD}" \
        --se_mode "${SE_MODE}" \
        --abstain_string "${ABSTAIN_STRING}" \
        --model_tag "${MODEL_TAG}" \
        --epochs "${EPOCHS}"
    python3 -c "
import json
from datetime import datetime, timezone
rec = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'model_tag': '${MODEL_TAG}',
    'prompt_style': '${PROMPT_STYLE}',
    'se_level': '${SE_LEVEL}',
    'se_mode': '${SE_MODE}',
    'se_threshold': ${SE_THRESHOLD},
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
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_se_done | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

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

echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_se_gen_done | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

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
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_se_gen_only | tag=${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"
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
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=distill_and_eval_se_done | tag=${MODEL_TAG} | node=$(hostname) | model_path=${DISTILLED_MODEL_PATH}" >> "${MANIFEST}"
echo "Distilled model:  ${DISTILLED_MODEL_PATH}"
echo "Generated data:   ${GEN_FILE}"
echo "Judged data:      ${JUDGED_OUTPUT_DIR}/"