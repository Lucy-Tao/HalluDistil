#!/bin/bash -l
#SBATCH --job-name=simpleqa_full
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

source /scratch-ssd/oatml/miniconda3/etc/profile.d/conda.sh
conda activate haldist

cd ~/SimpleQA

export HF_HOME=/scratch-ssd/oatml/huggingface
export HF_HUB_CACHE=/scratch-ssd/ms25yt/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch-ssd/ms25yt/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/ms25yt/datasets
export MPLCONFIGDIR=/scratch-ssd/ms25yt/.cache/matplotlib

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
mkdir -p /scratch-ssd/ms25yt/models

N_SAMPLES=2000
EPOCHS=1

echo "======================================="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Dataset: simpleqa"
echo "N_SAMPLES: $N_SAMPLES"
echo "EPOCHS: $EPOCHS"
echo "======================================="

python run.py \
  --mode distill \
  --dataset simpleqa \
  --n_samples "$N_SAMPLES" \
  --epochs "$EPOCHS"

HOME_MODEL_DIR="$HOME/SimpleQA/checkpoints/simpleqa_student"
SCRATCH_MODEL_DIR="/scratch-ssd/ms25yt/models/simpleqa_student"

mkdir -p "$(dirname "$HOME_MODEL_DIR")"
rm -rf "$HOME_MODEL_DIR"
mkdir -p "$HOME_MODEL_DIR"
rsync -a "$SCRATCH_MODEL_DIR"/ "$HOME_MODEL_DIR"/

echo "======================================="
echo "Finished: $(date)"
echo "Model synced to: $HOME_MODEL_DIR"
echo "======================================="
