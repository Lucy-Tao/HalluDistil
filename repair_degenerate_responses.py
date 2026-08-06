"""
repair_degenerate_responses.py — find genuine repetition-loop degenerate
outputs (e.g. "William H. H. H. H. H. H. H. ...") in an existing
generate_responses.py checkpoint file, and regenerate ONLY the affected
field(s) using the now-fixed sample_responses() (repetition_penalty=1.3
added — see semantic_utils.py), rewriting the checkpoint in place.

This is much cheaper than a full re-run: only loads the model ONCE per
file, and only regenerates the small number of specifically-broken
low_temp_response / raw_responses[i] values, not entire records.

Usage:
    python repair_degenerate_responses.py \
        --file ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-4B-Instruct-2507_fewshot.jsonl \
        --model_name Qwen/Qwen3-4B-Instruct-2507 \
        --dry_run   # report only, don't modify anything or load the model

Drop --dry_run to actually load the model, regenerate, and rewrite the
file (a .bak backup is written first, same convention as clean_checkpoint.py).
"""
import argparse
import json
import os
import re
import shutil

DEGENERATE_RE = re.compile(r'(.{1,6}?)\1{3,}')


def find_real_degenerate(text: str) -> str | None:
    """Return the matched repeating unit if `text` contains a genuine
    repetition-loop pattern (a short unit repeated 4+ times consecutively,
    total match length >= 8 chars), excluding legitimate large numbers
    (e.g. "600000000") which would otherwise false-positive on repeated
    "0" digits."""
    for m in DEGENERATE_RE.finditer(text):
        unit = m.group(1)
        if unit.strip("0123456789., ") == "":
            continue
        if len(m.group(0)) >= 8:
            return m.group(0)
    return None


def scan_file(path: str) -> list[dict]:
    """Return a list of {question_idx, line_no, fields: [...]} for every
    record with at least one degenerate field. `fields` is a list of
    field descriptors: "low_temp" or ("raw", i)."""
    issues = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            rec = json.loads(line)
            bad_fields = []
            if find_real_degenerate(rec["low_temp_response"]):
                bad_fields.append("low_temp")
            for i, r in enumerate(rec["raw_responses"]):
                if find_real_degenerate(r):
                    bad_fields.append(("raw", i))
            if bad_fields:
                issues.append({
                    "question_idx": rec["question_idx"],
                    "line_no": line_no,
                    "fields": bad_fields,
                    "prompt": rec["prompt"],
                })
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True,
                         help="e.g. Qwen/Qwen3-4B-Instruct-2507 or Qwen/Qwen3-14B "
                              "— must match whichever model originally generated this file")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    issues = scan_file(args.file)
    print(f"File: {args.file}")
    print(f"Found {len(issues)} record(s) with at least one degenerate field:")
    n_low  = sum(1 for it in issues if "low_temp" in it["fields"])
    n_high = sum(sum(1 for fld in it["fields"] if isinstance(fld, tuple)) for it in issues)
    print(f"  low_temp_response affected: {n_low}")
    print(f"  raw_responses entries affected: {n_high}")
    for it in issues[:10]:
        print(f"    question_idx={it['question_idx']}  fields={it['fields']}")
    if len(issues) > 10:
        print(f"    ... and {len(issues) - 10} more")

    if args.dry_run:
        print("\n(dry run — no model loaded, file NOT modified)")
        return

    if not issues:
        print("\nNothing to repair.")
        return

    from model_utils import load_model_and_tokenizer
    from semantic_utils import sample_responses

    print(f"\nLoading model: {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    fixes_by_idx = {}
    for it in issues:
        qidx = it["question_idx"]
        prompt = it["prompt"]
        fix = {}
        if "low_temp" in it["fields"]:
            print(f"[{qidx}] regenerating low_temp_response...")
            new_low = sample_responses(model, tokenizer, prompt,
                                        n_samples=1, temperature=0.1)[0]
            fix["low_temp_response"] = new_low
            print(f"    -> {new_low!r}")
        raw_indices_to_fix = [i for fld in it["fields"]
                               if isinstance(fld, tuple) for i in [fld[1]]]
        if raw_indices_to_fix:
            print(f"[{qidx}] regenerating {len(raw_indices_to_fix)} "
                  f"raw_responses entr(y/ies) at index(es) {raw_indices_to_fix}...")
            new_samples = sample_responses(model, tokenizer, prompt,
                                            n_samples=len(raw_indices_to_fix),
                                            temperature=1.0)
            fix["raw_fixes"] = dict(zip(raw_indices_to_fix, new_samples))
            for i, s in fix["raw_fixes"].items():
                print(f"    raw[{i}] -> {s!r}")
        fixes_by_idx[qidx] = fix

    # Rewrite the file, patching only the affected records
    backup_path = args.file + ".bak"
    shutil.copy2(args.file, backup_path)
    print(f"\nOriginal backed up to: {backup_path}")

    with open(args.file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(args.file, "w", encoding="utf-8") as f:
        for line in lines:
            rec = json.loads(line)
            qidx = rec["question_idx"]
            if qidx in fixes_by_idx:
                fix = fixes_by_idx[qidx]
                if "low_temp_response" in fix:
                    rec["low_temp_response"] = fix["low_temp_response"]
                if "raw_fixes" in fix:
                    for i, s in fix["raw_fixes"].items():
                        rec["raw_responses"][i] = s
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nRepaired {len(fixes_by_idx)} record(s) in {args.file}")


if __name__ == "__main__":
    main()