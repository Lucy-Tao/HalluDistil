#!/usr/bin/env python
"""
accuracy_seeds.py — Accuracy of ONE model under ONE prompt_style across 5 seeds.

For a single (model, style) pair, this script samples 500 SimpleQA questions
per seed (5 seeds), generates one low-temperature (T=0.1, non-greedy) answer
per question, grades each with the official SimpleQA grader (local Qwen3-32B,
non-thinking), and reports CORRECT / INCORRECT / NOT_ATTEMPTED counts per seed
plus the cross-seed accuracy mean/std. Nothing is written to disk; all output
goes to stdout.

Usage:
    python accuracy_seeds.py --model Qwen/Qwen3-32B --style strict
    python accuracy_seeds.py --model Qwen/Qwen3-4B-Instruct-2507 --style fewshot

Sampling pool is fixed at 4321 (the fewshot usable range, i.e. full corpus
minus the 5 few-shot exemplars) so that strict and fewshot draw the SAME 500
questions per seed and remain comparable.
"""
import argparse
import numpy as np

from config import cfg
from data_utils import load_dataset_items
from semantic_utils import sample_responses, QwenGrader
from model_utils import load_model_and_tokenizer

N_SAMPLE     = 500
SEEDS        = [0, 5, 10, 15, 20]
POOL_SIZE    = 4321          # fewshot usable range; shared by both styles
GRADER_MODEL = "Qwen/Qwen3-32B"
GEN_TEMP     = 0.1
MAX_NEW_TOK  = 50


def sampled_indices(seed):
    """Draw N_SAMPLE distinct indices from the shared pool for this seed."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(POOL_SIZE, size=N_SAMPLE, replace=False).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="generation model id, e.g. Qwen/Qwen3-32B")
    ap.add_argument("--style", required=True, choices=["strict", "fewshot"],
                    help="prompt style")
    args = ap.parse_args()

    # Load the full corpus for this style (num_samples large => return all).
    cfg.prompt_style = args.style
    items = load_dataset_items("simpleqa", num_samples=100000)
    print(f"[config] model={args.model}  style={args.style}  "
          f"corpus={len(items)}  pool={POOL_SIZE}  n_per_seed={N_SAMPLE}")
    assert len(items) >= POOL_SIZE, \
        f"corpus {len(items)} smaller than pool {POOL_SIZE}"

    # Load grader (Qwen3-32B) once.
    print(f"[load] grader: {GRADER_MODEL}")
    gm, gt = load_model_and_tokenizer(GRADER_MODEL, device_map=cfg.device_map)
    grader = QwenGrader(gm, gt)

    # Load generation model (reuse grader weights if identical).
    if args.model == GRADER_MODEL:
        print(f"[load] generation model == grader, reusing weights")
        gen_model, gen_tok = gm, gt
    else:
        print(f"[load] generation model: {args.model}")
        gen_model, gen_tok = load_model_and_tokenizer(args.model, device_map=cfg.device_map)

    results = []   # (seed, C, I, N)
    for seed in SEEDS:
        idx = sampled_indices(seed)
        C = I = N = 0
        for k, i in enumerate(idx):
            item = items[i]
            resp = sample_responses(
                gen_model, gen_tok, item["prompt"],
                n_samples=1, temperature=GEN_TEMP, max_new_tokens=MAX_NEW_TOK,
            )[0]
            grade = grader.grade(item["question"], item["answer"], resp)
            if   grade == "CORRECT":   C += 1
            elif grade == "INCORRECT": I += 1
            else:                      N += 1
            if (k + 1) % 100 == 0:
                print(f"  [{args.style} seed={seed}] {k+1}/{N_SAMPLE} "
                      f"C={C} I={I} N={N}")
        results.append((seed, C, I, N))
        print(f"==> {args.model} | {args.style} | seed={seed}: "
              f"CORRECT={C} INCORRECT={I} NOT_ATTEMPTED={N} "
              f"(acc={C/N_SAMPLE:.4f})")

    # Summary across seeds.
    print(f"\n{'='*66}")
    print(f"SUMMARY  model={args.model}  style={args.style}")
    print(f"{'='*66}")
    print(f"{'seed':<8}{'CORRECT':>9}{'INCORRECT':>11}{'NOT_ATT':>9}{'acc':>9}")
    for seed, C, I, N in results:
        print(f"{seed:<8}{C:>9}{I:>11}{N:>9}{C/N_SAMPLE:>9.4f}")
    accs = np.array([C / N_SAMPLE for _, C, _, _ in results])
    print(f"\nacc across seeds: mean={accs.mean():.4f}  std={accs.std(ddof=1):.4f}  "
          f"min={accs.min():.4f}  max={accs.max():.4f}")


if __name__ == "__main__":
    main()