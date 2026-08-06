"""
clean_checkpoint.py — validate every line of a generate_responses.py
checkpoint (.jsonl) file, strip out any corrupted/unparseable lines (e.g.
from two processes writing to the same file concurrently), and write a
clean version back.

IMPORTANT: only run this AFTER the job writing to the file has fully
finished (check `squeue -u $USER` first) — running this while a job is
still actively appending to the file risks losing whatever it writes
during the read/rewrite window.

If the same question_idx appears more than once, only the FIRST valid
occurrence is kept.

Usage:
    python clean_checkpoint.py --file ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-14B_strict.jsonl
    python clean_checkpoint.py --file <path> --dry_run   # report only, don't modify the file
"""
import argparse
import json
import os
import shutil


REQUIRED_FIELDS = {"question_idx", "question", "answer", "prompt",
                    "low_temp_response", "raw_responses"}


def clean_file(path: str, dry_run: bool = False) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    valid_lines = []
    seen_idx = set()
    bad_line_numbers = []
    duplicate_idx = []

    for line_no, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            bad_line_numbers.append(line_no)
            continue

        if not REQUIRED_FIELDS.issubset(rec.keys()):
            bad_line_numbers.append(line_no)
            continue

        qidx = rec["question_idx"]
        if qidx in seen_idx:
            duplicate_idx.append(qidx)
            continue
        seen_idx.add(qidx)
        valid_lines.append(stripped)

    report = {
        "total_lines_read":   len(raw_lines),
        "valid_records":      len(valid_lines),
        "bad_line_numbers":   bad_line_numbers,
        "duplicate_idx_skipped": duplicate_idx,
        "unique_question_idx": sorted(seen_idx),
    }

    if dry_run:
        return report

    if bad_line_numbers or duplicate_idx:
        backup_path = path + ".bak"
        shutil.copy2(path, backup_path)
        with open(path, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        report["backup_written_to"] = backup_path
    else:
        report["backup_written_to"] = None

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true",
                         help="report what would be cleaned without modifying the file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}")
        return

    report = clean_file(args.file, dry_run=args.dry_run)

    print(f"File: {args.file}")
    print(f"  Total lines read:      {report['total_lines_read']}")
    print(f"  Valid records kept:    {report['valid_records']}")
    print(f"  Corrupted lines found: {len(report['bad_line_numbers'])}"
          f"  {report['bad_line_numbers'] if report['bad_line_numbers'] else ''}")
    print(f"  Duplicate question_idx skipped: {len(report['duplicate_idx_skipped'])}"
          f"  {report['duplicate_idx_skipped'] if report['duplicate_idx_skipped'] else ''}")

    all_idx = set(report["unique_question_idx"])
    if all_idx:
        expected_range = set(range(min(all_idx), max(all_idx) + 1))
        missing = sorted(expected_range - all_idx)
        if missing:
            print(f"\n  NOTE: question_idx present in the file span "
                  f"{min(all_idx)}-{max(all_idx)}, but these are MISSING "
                  f"from that range:")
            print(f"    {missing}")
            print(f"  These will be regenerated automatically the next "
                  f"time this generation job is resumed.")

    if args.dry_run:
        print("\n(dry run — file was NOT modified)")
    elif report.get("backup_written_to"):
        print(f"\nFile cleaned. Original backed up to: {report['backup_written_to']}")
    else:
        print("\nNo corruption found — file left unchanged.")


if __name__ == "__main__":
    main()