"""
sample_question_indices.py — generate a FIXED, reproducible uniform
random sample of N question indices from the SimpleQA dataset, and save
it to a JSON file. This single file is the source of truth for "which
500 questions" across select_threshold.py, judge_responses.py, and
distill_and_eval.sh — generate it ONCE, then every downstream step reads
the same list, so there's no risk of different scripts silently using
different subsets.

Samples from range(usable_max), where usable_max = n_total -
n_fewshot_reserved (default 4326 - 5 = 4321). This range already excludes
the last n_fewshot_reserved rows, which data_utils.py's _load_simpleqa()
reserves as the FIXED few-shot pool for prompt_style="fewshot" — so
there's no overlap between a sampled target question and a few-shot
example by construction, without needing to also randomize which rows
serve as few-shot examples.

Uses a fixed random seed so re-running this script (e.g. if the file is
ever lost) reproduces the EXACT same sample, not a new random draw.

Usage:
    python sample_question_indices.py \
        --dataset simpleqa \
        --n_subset 500 \
        --seed 42 \
        --output ~/SimpleQA/subset_500_question_indices.json
"""
import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="simpleqa")
    parser.add_argument("--n_subset", type=int, required=True,
                         help="how many questions to sample")
    parser.add_argument("--n_fewshot_reserved", type=int, default=5,
                         help="how many rows at the end of the dataset are "
                              "reserved as the (fixed) few-shot pool, "
                              "excluded from sampling — must match "
                              "data_utils.py's _SIMPLEQA_N_FEWSHOT.")
    parser.add_argument("--n_total", type=int, default=4326,
                         help="total number of rows in the raw dataset "
                              "before reserving the few-shot pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    usable_max = args.n_total - args.n_fewshot_reserved  # e.g. 4321
    if args.n_subset > usable_max:
        raise ValueError(f"n_subset={args.n_subset} exceeds the usable pool "
                          f"size ({usable_max} = {args.n_total} - "
                          f"{args.n_fewshot_reserved} reserved for few-shot).")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(range(usable_max), args.n_subset))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "n_subset": args.n_subset,
            "seed": args.seed,
            "usable_max": usable_max,
            "question_indices": sampled,
        }, f, indent=2)

    print(f"Sampled {len(sampled)} question_idx (seed={args.seed}) from "
          f"range [0, {usable_max}) -> {args.output}")
    print(f"First 10: {sampled[:10]}")
    print(f"Last 10:  {sampled[-10:]}")


if __name__ == "__main__":
    main()