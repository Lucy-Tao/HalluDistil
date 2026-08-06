"""
normalize_abstentions.py -- Replace detected abstention responses with the
canonical string "Unknown", writing a NEW file with identical structure.
The input file is never modified.

Detection: full-text substring scan (no position window), calibrated and
fully manually verified on the 2026-07 new-sampling generation files
(including a dedicated review of every 'unknown' occurrence).

Usage:
  python normalize_abstentions.py --input <in.jsonl> --output <out.jsonl>
"""
import argparse, json, re

STRONG_PATTERNS = [
    # direct abstention
    "i don't know", "i do not know", "unknown",
    "cannot be determined", "cannot determine",
    "not applicable", "no reliable", "not publicly", "unable to",
    "no information", "not specified", "unclear", "uncertain",
    "insufficient information",
    # question-challenging abstention
    "based on a misconception", "misconception or", "factual error",
    "contains inaccurac", "no record of", "no verified",
    "no historical record", "no verifiable", "no documented",
    "not documented", "no such", "does not exist", "did not exist",
    "never occurred", "cannot be verified", "cannot be confirmed",
    "no evidence of", "no widely known",
    "question is incomplete", "question appears to be",
]
_NA_RE = re.compile(r'(?<![a-z0-9])n/a(?![a-z0-9])')   # word-boundary n/a
CANON = "Unknown"


def is_abstention(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    if _NA_RE.search(t):
        return True
    return any(p in t for p in STRONG_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.input == args.output:
        raise ValueError("Refusing to overwrite the input file.")

    n_records = n_lt = n_raw = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_records += 1
            if is_abstention(r["low_temp_response"]):
                r["low_temp_response"] = CANON
                n_lt += 1
            for i, t in enumerate(r["raw_responses"]):
                if is_abstention(t):
                    r["raw_responses"][i] = CANON
                    n_raw += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{args.input}: {n_records} records | replaced low_temp={n_lt}, raw={n_raw}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()