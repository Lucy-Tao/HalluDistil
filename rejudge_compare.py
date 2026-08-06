"""
rejudge_compare.py — re-judge a FIXED set of previously-generated responses
with multiple judge configs, to compare judge reliability without the
confound of re-sampling different generations each time (temperature=0.1
is not temperature=0, so re-running test_prompt_ab.py produces slightly
different responses each time, making raw accuracy numbers across separate
runs not directly comparable — see conversation).

Reads an existing figures/prompt_ab_*.csv (from test_prompt_ab.py) and only
reuses its question/gold/response_A/B/C columns; existing correct_A/B/C
columns are ignored (they were judged by whatever judge that run used).

Usage:
    python3 rejudge_compare.py \
        --input figures/prompt_ab_llm_20260708_091149.csv \
        --judges qwen14b:llm:Qwen/Qwen2.5-14B-Instruct \
                 qwen32b:llm:Qwen/Qwen2.5-32B-Instruct \
                 deberta:deberta:

Each --judges entry is "label:backend:model" (the trailing ":model" part is
ignored for backend=deberta, so "deberta:deberta:" with nothing after the
last colon is fine).

Outputs:
    figures/judge_compare_<timestamp>.csv       — every response, every judge's verdict
    figures/judge_disagreements_<timestamp>.csv — only the rows where judges disagreed
    (plus a printed summary: per-judge accuracy, pairwise agreement rates)
"""
import argparse
import csv
import gc
import time

import torch

from config import cfg
from semantic_utils import judge_correctness, load_local_llm_judge, load_nli_model

PROMPT_LETTERS = ["A", "B", "C"]


def parse_judge_spec(spec: str):
    """'label:backend:model' -> (label, backend, model_or_None)."""
    parts = spec.split(":", 2)
    if len(parts) == 2:
        label, backend = parts
        model = None
    else:
        label, backend, model = parts
        model = model or None
    return label, backend, model


def load_judge(backend: str, model: str | None):
    if backend == "llm":
        return load_local_llm_judge(model)
    if backend == "deberta":
        return load_nli_model(cfg.nli_model_name)
    raise ValueError(f"Unknown backend: {backend!r}. Use 'llm' or 'deberta'.")


def rejudge_csv(input_path: str, judge_specs: list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} questions from {input_path}")

    results = {}

    for spec in judge_specs:
        label, backend, model = parse_judge_spec(spec)
        print(f"\n=== Judging with: {label} (backend={backend}, model={model}) ===")
        judge_model, judge_tok = load_judge(backend, model)

        results[label] = {letter: [] for letter in PROMPT_LETTERS}
        for row in rows:
            question = row["question"]
            gold     = row["gold"]
            for letter in PROMPT_LETTERS:
                response = row[f"response_{letter}"]
                verdict = judge_correctness(
                    judge_model, judge_tok, question, gold, response,
                    backend=backend,
                )
                results[label][letter].append(verdict)

        del judge_model, judge_tok
        gc.collect()
        torch.cuda.empty_cache()

    return rows, results


def print_summary(judge_specs: list[str], results: dict):
    labels = [parse_judge_spec(s)[0] for s in judge_specs]

    print("\n" + "=" * 60)
    print("SUMMARY: accuracy per judge, per prompt variant")
    print("=" * 60)
    for label in labels:
        for letter in PROMPT_LETTERS:
            verdicts = results[label][letter]
            n = len(verdicts)
            acc = sum(verdicts) / n if n else 0.0
            print(f"  {label:12} prompt_{letter}: {acc:.1%}  ({sum(verdicts)}/{n})")

    if len(labels) < 2:
        return

    print("\n" + "=" * 60)
    print("PAIRWISE AGREEMENT (how often two judges give the same verdict, across A+B+C)")
    print("=" * 60)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            total, agree = 0, 0
            for letter in PROMPT_LETTERS:
                for v1, v2 in zip(results[l1][letter], results[l2][letter]):
                    total += 1
                    agree += (v1 == v2)
            print(f"  {l1} vs {l2}: {agree}/{total} = {agree / total:.1%} agreement")


def write_comparison_csv(rows, judge_specs, results, out_path):
    labels = [parse_judge_spec(s)[0] for s in judge_specs]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["question", "gold"]
        for letter in PROMPT_LETTERS:
            header.append(f"response_{letter}")
            header.extend(f"{label}_{letter}" for label in labels)
        writer.writerow(header)

        for i, row in enumerate(rows):
            out_row = [row["question"], row["gold"]]
            for letter in PROMPT_LETTERS:
                out_row.append(row[f"response_{letter}"])
                out_row.extend(results[label][letter][i] for label in labels)
            writer.writerow(out_row)
    print(f"\nWrote full comparison to {out_path}")


def write_disagreements_csv(rows, judge_specs, results, out_path):
    labels = [parse_judge_spec(s)[0] for s in judge_specs]
    disagreements = []
    for i, row in enumerate(rows):
        for letter in PROMPT_LETTERS:
            verdicts = [results[label][letter][i] for label in labels]
            if len(set(verdicts)) > 1:
                entry = {
                    "question": row["question"],
                    "gold": row["gold"],
                    "prompt": letter,
                    "response": row[f"response_{letter}"],
                }
                entry.update({label: results[label][letter][i] for label in labels})
                disagreements.append(entry)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if disagreements:
            writer = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
            writer.writeheader()
            writer.writerows(disagreements)
    print(f"Wrote {len(disagreements)} disagreement cases to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True,
                         help="path to an existing prompt_ab_*.csv "
                              "(needs question, gold, response_A/B/C columns)")
    parser.add_argument("--judges", type=str, nargs="+", required=True,
                         help="one or more 'label:backend:model' specs")
    args = parser.parse_args()

    rows, results = rejudge_csv(args.input, args.judges)
    print_summary(args.judges, results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    write_comparison_csv(rows, args.judges, results, f"figures/judge_compare_{ts}.csv")
    write_disagreements_csv(rows, args.judges, results, f"figures/judge_disagreements_{ts}.csv")


if __name__ == "__main__":
    main()