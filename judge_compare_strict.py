"""
judge_compare_strict.py — generate responses to 100 FRESH SimpleQA
questions (indices 200-299, never used in any earlier prompt test) under
the strict prompt, judge each response with three backends (deberta,
qwen14b, qwen32b), and write out:

  figures/judge_strict_all_<timestamp>.csv           — every question, every judge's verdict
  figures/judge_strict_disagreements_<timestamp>.csv  — only rows where the 3 judges disagree

The disagreements file is what you manually label — much smaller than the
full 100, since concordant cases (all 3 judges agree) carry no information
about which judge is more reliable.

Usage (on the cluster, inside the haldist conda env):
    python3 judge_compare_strict.py --n 100 --model Qwen/Qwen3-4B-Instruct-2507
"""
import argparse
import csv
import gc
import time

import torch

from config import cfg
from model_utils import load_model_and_tokenizer
from semantic_utils import (
    sample_responses, judge_correctness, load_local_llm_judge, load_nli_model,
)
from datasets import load_dataset

STRICT_PROMPT = (
    "Question: {question}\n"
    "Answer the question with only the minimal factual answer string.\n"
    "Do not write a full sentence.\n"
    "Do not include explanations, context, hedging, or punctuation.\n"
    "Do not start with phrases like 'The answer is' or 'It is'.\n"
    "Use the most common valid form of the answer.\n"
    "Answer:"
)

JUDGE_CONFIGS = [
    ("deberta", "deberta", None),
    ("qwen14b", "llm", "Qwen/Qwen2.5-14B-Instruct"),
    ("qwen32b", "llm", "Qwen/Qwen2.5-32B-Instruct"),
]


def load_judge(backend: str, model: str | None):
    if backend == "llm":
        return load_local_llm_judge(model)
    if backend == "deberta":
        return load_nli_model(cfg.nli_model_name)
    raise ValueError(f"Unknown backend: {backend!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                         help="number of fresh questions to test")
    parser.add_argument("--start_idx", type=int, default=200,
                         help="starting index into the dataset — default 200 "
                              "skips indices 0-199, which were already used "
                              "by earlier prompt_ab.py tests and shouldn't "
                              "be re-labeled (you may already know the "
                              "answers, which would bias manual labeling)")
    parser.add_argument("--model", type=str, default=cfg.student_model_name,
                         help="model whose responses get judged")
    args = parser.parse_args()

    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    rows = list(ds)[args.start_idx: args.start_idx + args.n]
    questions = [r["problem"] for r in rows]
    answers   = [r["answer"]  for r in rows]
    print(f"Using {len(questions)} fresh questions, dataset indices "
          f"{args.start_idx}-{args.start_idx + len(questions) - 1}.")

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("\n=== Generating responses under the strict prompt (T=0.1, 1 sample each) ===")
    responses = []
    for q in questions:
        prompt = STRICT_PROMPT.format(question=q)
        r = sample_responses(model, tokenizer, prompt, n_samples=1, temperature=0.1)[0]
        responses.append(r)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    results = {}
    for label, backend, judge_model_name in JUDGE_CONFIGS:
        print(f"\n=== Judging with: {label} (backend={backend}, model={judge_model_name}) ===")
        judge_model, judge_tok = load_judge(backend, judge_model_name)

        verdicts = []
        for q, gold, resp in zip(questions, answers, responses):
            verdict = judge_correctness(judge_model, judge_tok, q, gold, resp, backend=backend)
            verdicts.append(verdict)
        results[label] = verdicts

        del judge_model, judge_tok
        gc.collect()
        torch.cuda.empty_cache()

    labels = [c[0] for c in JUDGE_CONFIGS]

    print("\n" + "=" * 60)
    print("SUMMARY: accuracy per judge (on this same 100-question response set)")
    print("=" * 60)
    for label in labels:
        acc = sum(results[label]) / len(results[label])
        print(f"  {label:12}: {acc:.1%}  ({sum(results[label])}/{len(results[label])})")

    ts = time.strftime("%Y%m%d_%H%M%S")

    all_path = f"figures/judge_strict_all_{ts}.csv"
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "gold", "response"] + labels)
        for i, (q, gold, resp) in enumerate(zip(questions, answers, responses)):
            writer.writerow([q, gold, resp] + [results[label][i] for label in labels])
    print(f"\nWrote all {len(questions)} questions to {all_path}")

    disagree_path = f"figures/judge_strict_disagreements_{ts}.csv"
    disagreements = []
    for i, (q, gold, resp) in enumerate(zip(questions, answers, responses)):
        verdicts = [results[label][i] for label in labels]
        if len(set(verdicts)) > 1:
            entry = {"question": q, "gold": gold, "response": resp,
                     "human_label": ""}
            entry.update({label: results[label][i] for label in labels})
            disagreements.append(entry)

    with open(disagree_path, "w", newline="", encoding="utf-8") as f:
        if disagreements:
            fieldnames = ["question", "gold", "response", "human_label"] + labels
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disagreements)
    print(f"Wrote {len(disagreements)} disagreement cases to {disagree_path}")
    print("Fill in the 'human_label' column (True/False) for each row, then send it back.")


if __name__ == "__main__":
    main()