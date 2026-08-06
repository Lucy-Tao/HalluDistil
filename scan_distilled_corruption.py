"""
scan_distilled_corruption.py — systematically scan a distilled model's
gen_data output (and, for comparison, teacher/base-student's) for the
specific corruption patterns observed manually: emoji characters,
UUID/hex-fragment-like strings, and anomalously long / instructional-
sounding text mixed into what should be a short factual answer.

This is diagnostic, not a fix — the goal is to get a reliable COUNT
(instead of continuing to eyeball individual examples) of how often this
happens in the distilled model vs. teacher/base student, to confirm
whether the corruption is really distillation-introduced (near-zero rate
in teacher/student, non-trivial rate in the distilled model) rather than
inherited from training data.

Usage:
    python scan_distilled_corruption.py \
        --file ~/SimpleQA/gen_data_distilled/gen_simpleqa_..._strict.jsonl \
        --label distilled_strict
    python scan_distilled_corruption.py \
        --file ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-14B_strict.jsonl \
        --label teacher_strict
    python scan_distilled_corruption.py \
        --file ~/SimpleQA/gen_data_subset500/gen_simpleqa_Qwen3-4B-Instruct-2507_strict.jsonl \
        --label student_strict
"""
import argparse
import json
import re
import unicodedata

# Emoji: anything in common emoji Unicode blocks.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator (flag) letters
    "]"
)

# UUID / long hex-fragment-like strings: 6+ consecutive hex-looking chars
# with at least one dash, or a bare run of 12+ hex chars.
# NOTE: digits 0-9 are also valid hex characters, so a long PURE-NUMBER
# answer (e.g. a legitimate large ID) can trigger this — treat the
# "uuid_fragment" category as lower-precision than the others; check the
# actual example text before concluding a hit is genuinely malformed.
_UUID_FRAGMENT_RE = re.compile(
    r"(?:[0-9a-fA-F]{4,}-[0-9a-fA-F-]{4,})|(?:\b[0-9a-fA-F]{12,}\b)"
)

# Instructional/meta-sounding phrases that shouldn't appear in a short
# factual-answer response.
_INSTRUCTIONAL_PHRASES = [
    "must match", "conformed", "notfound", "unrecognized", "invalid format",
    "standard conformed definition", "colon separated", "exact lookup value",
    "acting as reference", "skip unrecognized",
]


def check_record(text: str) -> dict:
    has_emoji = bool(_EMOJI_RE.search(text))
    has_uuid_fragment = bool(_UUID_FRAGMENT_RE.search(text))
    has_instructional = any(p in text.lower() for p in _INSTRUCTIONAL_PHRASES)
    is_anomalously_long = len(text) > 100  # a short factual answer shouldn't be this long
    return {
        "emoji": has_emoji,
        "uuid_fragment": has_uuid_fragment,
        "instructional": has_instructional,
        "anomalously_long": is_anomalously_long,
        "any": has_emoji or has_uuid_fragment or has_instructional or is_anomalously_long,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--label", type=str, required=True,
                         help="a short label for this file in the printed report")
    args = parser.parse_args()

    n_records = 0
    n_skipped = 0
    n_records_with_issue = 0
    n_fields_checked = 0
    n_fields_with_issue = 0
    by_category = {"emoji": 0, "uuid_fragment": 0, "instructional": 0, "anomalously_long": 0}
    example_hits = []

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Tolerate a partially-written last line if this file is still
            # being actively appended to by a concurrent generation job —
            # skip it rather than crash the whole scan.
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            n_records += 1
            record_has_issue = False

            all_fields = [("low_temp", rec["low_temp_response"])] + [
                (f"raw[{i}]", r) for i, r in enumerate(rec["raw_responses"])
            ]
            for field_name, text in all_fields:
                n_fields_checked += 1
                result = check_record(text)
                if result["any"]:
                    n_fields_with_issue += 1
                    record_has_issue = True
                    for cat in by_category:
                        if result[cat]:
                            by_category[cat] += 1
                    if len(example_hits) < 10:
                        example_hits.append((rec["question_idx"], field_name, text[:120]))

            if record_has_issue:
                n_records_with_issue += 1

    print(f"{'='*60}")
    print(f"Corruption scan: {args.label}  ({args.file})")
    print(f"{'='*60}")
    print(f"Records scanned:           {n_records}")
    if n_skipped:
        print(f"Lines skipped (unparseable, likely a file still being "
              f"written to): {n_skipped}")
    print(f"Records with >=1 issue:    {n_records_with_issue}  "
          f"({100*n_records_with_issue/n_records:.2f}%)")
    print(f"Fields scanned:            {n_fields_checked}")
    print(f"Fields with issue:         {n_fields_with_issue}  "
          f"({100*n_fields_with_issue/n_fields_checked:.3f}%)")
    print(f"\nBy category (field-level counts):")
    for cat, count in by_category.items():
        print(f"  {cat:20s}: {count}")
    if example_hits:
        print(f"\nExample hits (up to 10):")
        for qidx, field, preview in example_hits:
            print(f"  [{qidx}] {field}: {preview!r}")
    print()


if __name__ == "__main__":
    main()