"""
Teacher semantic-entropy analysis for the seed44 SimpleQA subset.

Two outputs:
  1. A stacked histogram of entropy by grade (CORRECT vs INCORRECT).
     NOT_ATTEMPTED is excluded: only a handful of items, and the
     question here is whether entropy separates right from wrong.
  2. A cut table listing, for every entropy value actually present,
     what a threshold at that value would keep and discard. This is
     the input to the filtering and replacement interventions.

Entropy is computed over 10 T=1.0 samples via cluster frequencies, so
it can only take the values realisable by partitions of 10 into
clusters. That makes it DISCRETE and unevenly spaced on the axis
(0.6109, 0.6390, 0.6730, 0.6931 sit within 0.09 of each other, while
nothing at all exists between 0 and 0.3251). Percentiles are therefore
misleading here (p90 and p95 are both ln 10); pick thresholds off the
cut table instead.

Usage:
    python analyse_teacher_entropy.py
    python analyse_teacher_entropy.py --styles strict
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JUDGED_DIR = "judged_data_seed44_deberta"
TEACHER = "Qwen3-32B"
MAX_ENT = math.log(10)


def load(style, judged_dir, teacher):
    path = f"{judged_dir}/judged_simpleqa_{teacher}_{style}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cut_table(recs, style):
    """Rows of (threshold, kept, correct kept) for every realisable value."""
    n = len(recs)
    total_c = sum(1 for r in recs if r["grade"] == "CORRECT")
    vals = sorted({round(r["semantic_entropy"], 4) for r in recs})

    print(f"\n--- {style}   n={n}  CORRECT={total_c}  "
          f"({100 * total_c / n:.1f}%)")
    print(f"{'thresh':>8} {'kept':>5} {'kept%':>6} {'C_kept':>7} "
          f"{'C_recall':>9} {'acc_kept':>9}")

    rows = []
    for v in vals:
        keep = [r for r in recs if round(r["semantic_entropy"], 4) <= v]
        ck = sum(1 for r in keep if r["grade"] == "CORRECT")
        rows.append({
            "threshold": v,
            "kept": len(keep),
            "kept_frac": len(keep) / n,
            "correct_kept": ck,
            "correct_recall": ck / total_c if total_c else 0.0,
            "acc_kept": ck / len(keep) if keep else 0.0,
        })
        print(f"{v:8.4f} {len(keep):5d} {100 * len(keep) / n:5.1f}% {ck:7d} "
              f"{100 * ck / total_c if total_c else 0:8.1f}% "
              f"{100 * ck / len(keep) if keep else 0:8.2f}%")
    return rows


def plot(ax, recs, style, bins):
    correct = [r["semantic_entropy"] for r in recs if r["grade"] == "CORRECT"]
    incorrect = [r["semantic_entropy"] for r in recs
                 if r["grade"] == "INCORRECT"]

    ax.hist([incorrect, correct], bins=bins, stacked=True,
            color=["#e8998d", "#8fbf9f"], alpha=0.85,
            edgecolor="white", linewidth=0.3,
            label=[f"INCORRECT (n={len(incorrect)}, "
                   f"mean={np.mean(incorrect):.2f})",
                   f"CORRECT (n={len(correct)}, "
                   f"mean={np.mean(correct):.2f})"])

    ax.axvline(MAX_ENT, ls="--", lw=1, color="grey")
    ax.set_xlabel("semantic entropy")
    ax.set_title(style)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=8, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", nargs="+", default=["strict", "fewshot"])
    ap.add_argument("--judged_dir", default=JUDGED_DIR)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--nbins", type=int, default=21)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bins = np.linspace(0, MAX_ENT, args.nbins)

    fig, axes = plt.subplots(1, len(args.styles),
                             figsize=(5.5 * len(args.styles), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    summary = {}
    for ax, style in zip(axes, args.styles):
        recs = load(style, args.judged_dir, args.teacher)
        plot(ax, recs, style, bins)
        summary[style] = cut_table(recs, style)

    axes[0].set_ylabel("questions")
    fig.tight_layout()

    fig_path = f"{args.outdir}/teacher_entropy_by_grade.png"
    fig.savefig(fig_path, dpi=150)

    json_path = f"{args.outdir}/teacher_entropy_cut_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nfigure    -> {fig_path}")
    print(f"cut table -> {json_path}")


if __name__ == "__main__":
    main()