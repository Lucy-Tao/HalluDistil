#!/bin/bash
#SBATCH --job-name=judge_compare
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# judge_compare.sh — run judge_compare_strict.py: generate 100 fresh
# responses under the strict prompt, judge with deberta/qwen14b/qwen32b,
# write out the full comparison + a disagreements-only CSV for manual
# labeling.
#
# No --nodelist needed — doesn't touch scratch-ssd, everything reads from
# HuggingFace (network) and writes to the home-directory figures/ folder.
#
# Time estimate: 100 questions x 1 generation + 100 x 3 judge calls.
# Loading three separate judge models (deberta, 14B, 32B) sequentially
# also takes a few minutes each. 3 hours is a generous ceiling; this
# should finish well under that.

set -e
set -x

PROJECT_DIR=~/SimpleQA
cd "${PROJECT_DIR}"
mkdir -p logs figures

source ~/.bashrc
conda activate haldist

echo "===== [$(date)] Running on host: $(hostname) ====="
nvidia-smi

python judge_compare_strict.py --n 100 --start_idx 200 \
    --model Qwen/Qwen3-4B-Instruct-2507

echo "===== [$(date)] Done ====="
ls -lh figures/judge_strict_*