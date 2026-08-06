"""
extract_question_indices.py — read a filter_questions.py scan output JSON
and print space-separated question_idx values to stdout, for use as
`$(python extract_question_indices.py ...)` inside a shell script building
run.py's --question_indices argument.

Two modes:
  --all                 print every question_idx in scan_file["teacher_records"]
  --threshold <T>       print scan_file["thresholds"][i]["question_indices"]
                         for the entry whose "threshold" == T (the "both
                         teacher and base student have semantic_entropy >= T"
                         set built by filter_high_entropy_questions())

Usage:
    python extract_question_indices.py --scan_file figures/strict/scan_simpleqa_....json --all
    python extract_question_indices.py --scan_file figures/strict/scan_simpleqa_....json --threshold 1.5
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_file", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                        help="print every question_idx in the scan file")
    group.add_argument("--threshold", type=float, default=None,
                        help="print the 'both high entropy' question_idx "
                             "list for this threshold")
    args = parser.parse_args()

    with open(args.scan_file, "r", encoding="utf-8") as f:
        scan_data = json.load(f)

    if args.all:
        indices = sorted(r["question_idx"] for r in scan_data["teacher_records"])
    else:
        matches = [t for t in scan_data["thresholds"] if t["threshold"] == args.threshold]
        if not matches:
            available = [t["threshold"] for t in scan_data["thresholds"]]
            print(f"ERROR: threshold {args.threshold} not found in {args.scan_file}. "
                  f"Available thresholds: {available}", file=sys.stderr)
            sys.exit(1)
        indices = matches[0]["question_indices"]

    print(" ".join(str(i) for i in indices))


if __name__ == "__main__":
    main()