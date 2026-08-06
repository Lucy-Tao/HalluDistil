#!/bin/bash
#SBATCH --job-name=factscore_eval
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat16
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Usage:
#   # baseline models live in gen_longform_data/ (the default dir):
#   sbatch --job-name=factscore_Qwen3-32B factscore_eval.sh Qwen3-32B
#   sbatch --job-name=factscore_student   factscore_eval.sh Qwen3-4B-Instruct-2507
#
#   # distilled models live in gen_longform_distilled/ -> pass it as arg 2:
#   sbatch --job-name=factscore_ep3 factscore_eval.sh distilled_ep3 gen_longform_distilled
#
# Arg 1: MODEL TAG -- the part of the filename between "gen_factscore_bio_"
#        and ".jsonl".
# Arg 2 (optional): SUBDIR under ~/SimpleQA/ holding the generation file.
#        Defaults to gen_longform_data. Output is always written to
#        gen_longform_data/ regardless, so all factscore_*.jsonl land together.

set -e
set -x

export WANDB_MODE=disabled

MODEL_TAG="$1"
GEN_SUBDIR="${2:-gen_longform_data}"
if [ -z "${MODEL_TAG}" ]; then
    echo "ERROR: pass the model tag as the first argument."
    echo "  e.g. factscore_eval.sh Qwen3-32B"
    echo "  e.g. factscore_eval.sh distilled_ep3 gen_longform_distilled"
    exit 1
fi

PROJECT_DIR=~/SimpleQA
GEN_DIR="${PROJECT_DIR}/${GEN_SUBDIR}"
OUT_DIR="${PROJECT_DIR}/gen_longform_data"
GEN="${GEN_DIR}/gen_factscore_bio_${MODEL_TAG}.jsonl"
OUT="${OUT_DIR}/factscore_${MODEL_TAG}.jsonl"
MANIFEST="${PROJECT_DIR}/logs/experiment_manifest.log"

cd "${PROJECT_DIR}"
mkdir -p logs

source ~/.bashrc
conda activate haldist

if [ -z "${LAGRANGE_API_KEY}" ]; then
    echo "ERROR: LAGRANGE_API_KEY is not set. FActScore judging will fail."
    echo "  export it before sbatch, or set it inside this script."
    exit 1
fi

if [ ! -f "${GEN}" ]; then
    echo "ERROR: generation file not found: ${GEN}"
    echo "  (looked in subdir '${GEN_SUBDIR}'; pass a different arg 2 if it lives elsewhere)"
    exit 1
fi

echo "===== [$(date)] Running on host: $(hostname) ====="
nvidia-smi
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=factscore_${MODEL_TAG} | node=$(hostname)" >> "${MANIFEST}"

MAX_RETRIES=10
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "----- Attempt ${attempt}/${MAX_RETRIES} -----"
    set +e
    python run_factscore_eval.py \
        --gen "${GEN}" \
        --out "${OUT}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -eq 0 ]; then
        echo "Attempt ${attempt} succeeded."
        break
    fi
    echo "Attempt ${attempt} failed with exit code ${exit_code}."
    echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=factscore_${MODEL_TAG}_retry | attempt=${attempt} | exit_code=${exit_code} | node=$(hostname)" >> "${MANIFEST}"
    if [ "${attempt}" -eq "${MAX_RETRIES}" ]; then
        echo "ERROR: exhausted ${MAX_RETRIES} attempts, giving up."
        exit "${exit_code}"
    fi
    sleep 15
done

echo "===== [$(date)] Done ====="
echo "$(date -Iseconds) | job=${SLURM_JOB_ID:-none} | stage=factscore_${MODEL_TAG}_done | node=$(hostname)" >> "${MANIFEST}"