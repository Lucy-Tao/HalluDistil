"""
compare_abstention_rules.py -- standalone, read-only diagnostic.

Compares two abstention/non-answer detection rules on the long-form
generation files, WITHOUT modifying anything:

  1. SHARS  : the is_non_answer() rule from TreeLLi/hallucination-rejection-
              sampling (pure phrase match, ~45 phrases, NO position check).
  2. YOURS  : compute_abstention_rate.py's rule (narrow phrase list +
              position check at max(150, 20% of length)).

For each file it prints:
  - how many each rule flags,
  - the symmetric difference (entries where the two disagree),
  - the FULL response text for every disagreement, so you can decide
    by hand whether it is a genuine refusal or a false positive.

Usage
-----
  python compare_abstention_rules.py \\
      gen_longform_data/gen_factscore_bio_Qwen3-32B.jsonl \\
      gen_longform_data/gen_factscore_bio_Qwen3-4B-Instruct-2507.jsonl

Pass any number of jsonl files as positional args. No files are written.
"""
import argparse
import json
import re


# ---------------------------------------------------------------------------
# Rule 1: SHARS reference -- pure phrase match, no position check.
# Transcribed from TreeLLi/hallucination-rejection-sampling utils.py
# is_non_answer(). Duplicate "no widely known" / "i'm not aware" /
# "i do not know" in the original are harmless and left as-is.
# ---------------------------------------------------------------------------
SHARS_PATTERNS = [
    "no widely known", "not widely known", "not widely available",
    "not publicly available", "not a well-known", "not well-known",
    "not known", "not publicly", "no prominent", "no notable", "no known",
    "no famously known", "i could not find", "couldn't find",
    "does not exist", "not a real person", "i'm not aware", "i do not know",
    "not available", "no information available", "cannot guarantee",
    "recommend checking", "may want to check", "please check",
    "consider checking", "little information", "common name", "can refer to",
    "several individuals", "multiple individuals", "few individuals",
    "not a widely recognized", "further clarification may help",
    "i do not have information", "i don't have information",
    "i have no information", "i lack information", "i'm not sure",
    "i cannot confirm", "i cannot guarantee", "i cannot verify",
    "may not be accurate", "information may be incomplete",
    "information may not be reliable", "details are limited",
]


def shars_rule(response: str):
    """Return (is_non_answer, matched_phrase_or_None, position_or_None)."""
    if not response or not isinstance(response, str):
        return True, "<empty>", 0
    low = response.strip().lower()
    for pat in SHARS_PATTERNS:
        idx = low.find(pat)
        if idx != -1:
            return True, pat, idx
    return False, None, None


# ---------------------------------------------------------------------------
# Rule 2: your compute_abstention_rate.py rule -- narrow phrases + position.
# ---------------------------------------------------------------------------
YOUR_PATTERN = re.compile(
    r"no widely known|no widely recognized|no publicly known|"
    r"no publicly available|no publicly documented|"
    r"not (?:be )?widely (?:known|recognized)|"
    r"no well-known|no verified (?:biography|information)",
    re.IGNORECASE,
)


def your_rule(response: str):
    """Return (is_abstained, matched_phrase_or_None, position_or_None)."""
    m = YOUR_PATTERN.search(response)
    if m is None:
        return False, None, None
    cutoff = max(150, int(0.20 * len(response)))
    return (m.start() < cutoff), m.group(0), m.start()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="generation jsonl file(s)")
    ap.add_argument("--full", action="store_true",
                    help="print the entire response for disagreements "
                         "(default: first 400 chars)")
    args = ap.parse_args()

    for path in args.files:
        entries = load(path)
        print("=" * 78)
        print(f"FILE: {path}   (n={len(entries)})")
        print("=" * 78)

        shars_flag = {}
        yours_flag = {}
        detail = {}
        for d in entries:
            qi = d["question_idx"]
            resp = d.get("response", "")
            s_hit, s_pat, s_pos = shars_rule(resp)
            y_hit, y_pat, y_pos = your_rule(resp)
            shars_flag[qi] = s_hit
            yours_flag[qi] = y_hit
            detail[qi] = (d["entity"], resp, s_pat, s_pos, y_pat, y_pos)

        s_set = {q for q, v in shars_flag.items() if v}
        y_set = {q for q, v in yours_flag.items() if v}

        print(f"SHARS flags : {len(s_set)}   -> {sorted(s_set)}")
        print(f"YOUR  flags : {len(y_set)}   -> {sorted(y_set)}")
        print(f"agree on both flagged   : {sorted(s_set & y_set)}")
        print(f"SHARS only (you say ANSWER, SHARS says REFUSE): {sorted(s_set - y_set)}")
        print(f"YOURS only (you say REFUSE, SHARS says ANSWER): {sorted(y_set - s_set)}")
        print()

        disagree = sorted(s_set ^ y_set)
        if not disagree:
            print("No disagreements on this file.\n")
            continue

        print("-" * 78)
        print("DISAGREEMENTS -- read each and decide: genuine refusal, or false positive?")
        print("-" * 78)
        for qi in disagree:
            entity, resp, s_pat, s_pos, y_pat, y_pos = detail[qi]
            who = "SHARS-only" if qi in (s_set - y_set) else "YOURS-only"
            print(f"\n[qidx={qi}] {entity}   ({who})")
            if shars_flag[qi]:
                print(f"  SHARS: matched {s_pat!r} at char {s_pos} of {len(resp)}")
            else:
                print(f"  SHARS: no match")
            if yours_flag[qi]:
                cutoff = max(150, int(0.20 * len(resp)))
                print(f"  YOURS: matched {y_pat!r} at char {y_pos}, cutoff={cutoff} -> abstained")
            else:
                if y_pat is not None:
                    cutoff = max(150, int(0.20 * len(resp)))
                    print(f"  YOURS: matched {y_pat!r} at char {y_pos} but cutoff={cutoff} -> NOT abstained")
                else:
                    print(f"  YOURS: no phrase in list matched")
            body = resp if args.full else resp[:400] + ("..." if len(resp) > 400 else "")
            print(f"  RESPONSE:\n    {body}")

        print()


if __name__ == "__main__":
    main()