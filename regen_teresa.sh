#!/bin/bash
#SBATCH --job-name=regen_teresa
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=oat16
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
set -x
cd ~/SimpleQA
source ~/.bashrc
conda activate haldist

python generate_longform_responses.py \
    --model_role student \
    --n_samples 183 \
    --model_name /scratch-ssd/ms25yt/models/factscore_bio_distilled_student_ep10 \
    --run_tag distilled_ep10 \
    --question_idx_subset gen_longform_data/teresa_subset.jsonl \
    --output_dir gen_longform_data
