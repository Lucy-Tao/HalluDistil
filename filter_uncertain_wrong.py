"""
filter_uncertain_wrong.py — Step 2: filter scan_teacher_entropy.py's output
into "genuinely uncertain then wrong" candidate lists, at two thresholds.

Purpose
-------
scan_teacher_entropy.py already computed (choice_entropy, is_correct) for
every question in the dataset, with NO filtering. This script reads that
already-computed JSON and applies two entropy thresholds to separate:

  - "confidently wrong"            (entropy < threshold, is_correct=False)
    NOT your research target — the teacher simply doesn't know the answer,
    there's no uncertainty being suppressed by distillation.

  - "genuinely uncertain then wrong" (entropy >= threshold, is_correct=False)
    YOUR research target — the teacher was torn between options and its
    argmax happened to land on the wrong one. This is exactly the scenario
    single_prompt_distill_curve.py is designed to probe.

No model is re-run here — this is pure filtering over already-computed
data, so it's instant regardless of dataset size.

Two thresholds, two output lists
----------------------------------
  --loose_threshold  (default 0.1)
      A broad candidate pool: excludes only the most extreme
      confidently-wrong cases. Larger list, useful as a general filter.

  --strict_threshold (default 0.5)
      A tighter pool closer to your original "P(A)~=P(B)~=0.4" scenario —
      genuinely close to a 50/50-ish split rather than just "not
      completely certain". Smaller list, better suited for hand-picking
      individual cases for single_prompt_distill_curve.py.

Usage
-----
  python filter_uncertain_wrong.py \\
      --scan_file ./figures/teacher_entropy_scan_gpqa_Qwen_Qwen3-14B.json

  python filter_uncertain_wrong.py \\
      --scan_file ./figures/teacher_entropy_scan_gpqa_Qwen_Qwen3-14B.json \\
      --loose_threshold 0.1 --strict_threshold 0.5

Output
------
  {scan_file basename}_loose_t{loose_threshold}.json
  {scan_file basename}_strict_t{strict_threshold}.json

  Each is a list of records (same fields as the scan output) restricted to
  is_correct=False and choice_entropy >= threshold, sorted by choice_entropy
  descending (most uncertain first — usually the most interesting cases to
  look at first).
"""

from __future__ import annotations

import argparse
import json
import os


def load_scan(scan_file: str) -> list[dict]:
    with open(scan_file, encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} scanned questions from {scan_file}")
    return records


def filter_uncertain_wrong(records: list[dict], threshold: float) -> list[dict]:
    """
    Keep only records where the teacher was wrong (is_correct=False) AND
    choice_entropy >= threshold. Sorted most-uncertain-first so the
    questions most likely to show a clean "torn between two options"
    pattern appear at the top of the list.
    """
    filtered = [
        r for r in records
        if (not r["is_correct"]) and r["choice_entropy"] >= threshold
    ]
    filtered.sort(key=lambda r: r["choice_entropy"], reverse=True)
    return filtered


def print_candidate_preview(filtered: list[dict], label: str, n_preview: int = 10):
    print(f"\n{'='*70}")
    print(f"{label}: {len(filtered)} candidate questions")
    print(f"{'='*70}")
    if not filtered:
        print("  (no questions meet this threshold)")
        return

    print(f"{'idx':>5}  {'entropy':>8}  {'gold':>5}  {'argmax':>7}  "
          f"{'P(gold)':>8}  question")
    print("-" * 70)
    for r in filtered[:n_preview]:
        q_preview = r["question"][:60]
        print(f"{r['question_idx']:>5}  {r['choice_entropy']:>8.4f}  "
              f"{r['gold_answer']:>5}  {r['teacher_argmax']:>7}  "
              f"{r['prob_on_gold']:>8.4f}  {q_preview}")
    if len(filtered) > n_preview:
        print(f"  ... and {len(filtered) - n_preview} more "
              f"(see the saved JSON for the full list)")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Filter scan_teacher_entropy.py output into "
                     "genuinely-uncertain-then-wrong candidate lists."
    )
    parser.add_argument("--scan_file", required=True,
                        help="Path to the JSON produced by scan_teacher_entropy.py")
    parser.add_argument("--loose_threshold", type=float, default=0.1,
                        help="Broad candidate pool threshold (default: 0.1)")
    parser.add_argument("--strict_threshold", type=float, default=0.5,
                        help="Tight candidate pool threshold, closer to a "
                             "genuine near-50/50 split (default: 0.5)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save the filtered lists "
                             "(default: same directory as --scan_file)")
    args = parser.parse_args()

    records = load_scan(args.scan_file)

    loose    = filter_uncertain_wrong(records, args.loose_threshold)
    strict   = filter_uncertain_wrong(records, args.strict_threshold)

    print_candidate_preview(
        loose, f"LOOSE  (entropy >= {args.loose_threshold})"
    )
    print_candidate_preview(
        strict, f"STRICT (entropy >= {args.strict_threshold})"
    )

    base_dir  = args.output_dir or os.path.dirname(args.scan_file)
    base_name = os.path.splitext(os.path.basename(args.scan_file))[0]

    loose_path  = os.path.join(
        base_dir, f"{base_name}_loose_t{args.loose_threshold}.json"
    )
    strict_path = os.path.join(
        base_dir, f"{base_name}_strict_t{args.strict_threshold}.json"
    )

    with open(loose_path, "w", encoding="utf-8") as f:
        json.dump(loose, f, indent=2, ensure_ascii=False)
    with open(strict_path, "w", encoding="utf-8") as f:
        json.dump(strict, f, indent=2, ensure_ascii=False)

    print(f"\n  Loose candidate list  -> {loose_path}")
    print(f"  Strict candidate list -> {strict_path}")
    print(f"\nNext step: pick a question_idx from either list and run")
    print(f"  python single_prompt_distill_curve.py --question_idx <idx> ...")


if __name__ == "__main__":
    main()