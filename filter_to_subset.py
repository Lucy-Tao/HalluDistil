"""
filter_to_subset.py — back up a .jsonl file, then keep only the records
whose question_idx appears in a subset file (e.g. sample_question_indices.py's
output). Two uses:

  1. Trim an ALREADY-COMPLETE gen_data file (teacher/base student, full
     4321/4326 questions) down to just the 500-question random subset,
     writing a NEW file (--output) — the original full file is left
     untouched, since it's valuable to keep around for a future
     larger-scale run.

  2. Clean up an EXISTING judged_data file that has leftover records from
     an earlier sequential run (question_idx not in the new random
     subset) mixed in with the ones that ARE in the subset — rewrites the
     file IN PLACE (backs up to <file>.bak first), so it only contains
     subset-relevant records going forward. After this, judge_responses.py
     (pointed at the correspondingly-trimmed gen_data file) needs no
     special-casing to "know about" the subset — its checkpoint/resume
     logic just naturally sees a clean, already-scoped file.

Usage:
    # Trim a full gen_data file into a new subset-only file:
    python filter_to_subset.py \
        --file ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-14B_strict.jsonl \
        --subset_file ~/SimpleQA/subset_500_question_indices.json \
        --output ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-14B_strict.jsonl

    # Clean up an existing judged_data file in place (backs up first):
    python filter_to_subset.py \
        --file ~/SimpleQA/judged_data/judged_simpleqa_Qwen3-14B_strict.jsonl \
        --subset_file ~/SimpleQA/subset_500_question_indices.json
"""
import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, required=True,
                         help="the .jsonl file to filter")
    parser.add_argument("--subset_file", type=str, required=True,
                         help="a JSON file with a 'question_indices' list "
                              "(e.g. sample_question_indices.py's output)")
    parser.add_argument("--output", type=str, default=None,
                         help="if set, write the filtered result here "
                              "instead of overwriting --file in place. "
                              "When omitted, --file is overwritten in "
                              "place (a .bak backup is written first).")
    args = parser.parse_args()

    with open(args.subset_file, "r", encoding="utf-8") as f:
        allowed_idx = set(json.load(f)["question_indices"])
    print(f"Subset: {len(allowed_idx)} question_idx from {args.subset_file}")

    with open(args.file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    print(f"Read {len(lines)} record(s) from {args.file}")

    kept = []
    kept_idx = set()
    for line in lines:
        rec = json.loads(line)
        if rec["question_idx"] in allowed_idx:
            kept.append(line if line.endswith("\n") else line + "\n")
            kept_idx.add(rec["question_idx"])

    missing = sorted(allowed_idx - kept_idx)

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_path = args.output
    else:
        backup_path = args.file + ".bak"
        shutil.copy2(args.file, backup_path)
        print(f"Backed up original -> {backup_path}")
        out_path = args.file

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(kept)

    print(f"Kept {len(kept)}/{len(lines)} record(s) -> {out_path}")
    if missing:
        print(f"NOTE: {len(missing)} question_idx from the subset were NOT "
              f"found in {args.file} (not yet generated/judged for those): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        print("All subset question_idx were present and kept.")


if __name__ == "__main__":
    main()