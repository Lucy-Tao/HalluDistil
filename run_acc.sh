#!/bin/bash
#SBATCH --job-name=acc_seeds
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# run_accuracy.sh <model> <style>
#
# Submit one (model, style) accuracy job. Each job runs 5 seeds internally.
# The four jobs to run:
#   sbatch run_accuracy.sh Qwen/Qwen3-32B                strict
#   sbatch run_accuracy.sh Qwen/Qwen3-4B-Instruct-2507   strict
#   sbatch run_accuracy.sh Qwen/Qwen3-32B                fewshot
#   sbatch run_accuracy.sh Qwen/Qwen3-4B-Instruct-2507   fewshot

set -e
set -x

MODEL="${1:?Usage: sbatch run_accuracy.sh <model> <style>}"
STYLE="${2:?Usage: sbatch run_accuracy.sh <model> <style>}"

if [[ "${STYLE}" != "strict" && "${STYLE}" != "fewshot" ]]; then
    echo "ERROR: style must be 'strict' or 'fewshot', got '${STYLE}'"
    exit 1
fi

cd ~/SimpleQA
source ~/.bashrc
conda activate haldist

echo "===== [$(date)] host=$(hostname) model=${MODEL} style=${STYLE} ====="
nvidia-smi

python accuracy_seeds.py --model "${MODEL}" --style "${STYLE}"

echo "===== [$(date)] done ====="