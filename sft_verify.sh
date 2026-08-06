#!/bin/bash
#SBATCH --job-name=sft_verify_ep10
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
cd ~/SimpleQA
source ~/.bashrc
conda activate haldist
python distill_sfttrainer.py \
    --judged_file gen_data_subset500/gen_simpleqa_Qwen3-14B_strict.jsonl \
    --model_tag strict_full_sfttrainer_ep10b --epochs 10