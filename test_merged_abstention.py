"""
test_merged_abstention.py -- standalone, read-only.

Tests the MERGED abstention rule before it goes into
compute_abstention_rate.py:

  - phrase list = SHARS non-answer phrases + your original variants
    + the missing ones ("no well-known", "not a widely known",
    "not a widely recognized", ...), deduplicated.
  - hedging phrases ("multiple individuals", "several individuals",
    "common name", "can refer to", ...) are KEPT -- these count as
    non-answers per your decision.
  - position check: a phrase only counts if it appears within the
    first 200 characters of the response. This is what stops a
    mid-text mention like "the fish is not widely known" (Botak Chin)
    from being misread as a refusal.

Prints, per file: the entities judged abstained, plus any entity that
matched a phrase only AFTER char 200 (i.e. kept as a real answer thanks
to the position check) so you can eyeball the false-positives it avoids.

Writes nothing.

Usage
-----
  python test_merged_abstention.py \\
      gen_longform_data/gen_factscore_bio_Qwen3-32B.jsonl \\
      gen_longform_data/gen_factscore_bio_Qwen3-4B-Instruct-2507.jsonl

  # add --full to print the whole response for each abstained entity
  python test_merged_abstention.py FILE.jsonl --full
"""
import argparse
import json

CUTOFF_CHARS = 200

MERGED_PATTERNS = sorted(set([
    # refusal / not-found
    "no widely known", "no widely recognized", "not widely known",
    "not widely recognized", "not a widely known", "not a widely recognized",
    "no well-known", "no well known", "not a well-known", "not well-known",
    "not known", "no known", "no prominent", "no notable", "no famously known",
    "not widely available", "not publicly available", "not publicly",
    "no publicly known", "no publicly available", "no publicly documented",
    "not publicly documented", "does not exist", "not a real person",
    "no verified biography", "no verified information", "no verified record",
    "no information available", "no prominent record", "no prominent information",
    # first-person disclaimers
    "i could not find", "couldn't find", "i'm not aware", "i am not aware",
    "i do not know", "i don't know", "i'm not sure", "i cannot confirm",
    "i cannot guarantee", "i cannot verify", "i do not have information",
    "i don't have information", "i have no information", "i lack information",
    # hedging (KEPT as non-answers per decision)
    "common name", "can refer to", "several individuals",
    "multiple individuals", "few individuals",
    # soft disclaimers
    "not available", "cannot guarantee", "recommend checking",
    "may want to check", "please check", "consider checking",
    "little information", "further clarification may help",
    "may not be accurate", "information may be incomplete",
    "information may not be reliable", "details are limited",
]))


def abstained(response: str):
    """Return (is_abstained, phrase_or_None, position_or_None).

    Finds the EARLIEST-matching phrase, then applies the position check.
    """
    if not response or not isinstance(response, str):
        return True, "<empty>", 0
    low = response.strip().lower()
    best = None
    for pat in MERGED_PATTERNS:
        idx = low.find(pat)
        if idx != -1 and (best is None or idx < best[1]):
            best = (pat, idx)
    if best is None:
        return False, None, None
    return (best[1] < CUTOFF_CHARS), best[0], best[1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--full", action="store_true",
                    help="print the full response for each abstained entity")
    args = ap.parse_args()

    for path in args.files:
        entries = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        print("=" * 74)
        print(f"FILE: {path}  (n={len(entries)})   cutoff = first {CUTOFF_CHARS} chars")
        print("=" * 74)

        abst = []
        late = []   # matched a phrase but only after the cutoff -> kept as answer
        for d in entries:
            hit, pat, pos = abstained(d["response"])
            if hit:
                abst.append((d["question_idx"], d["entity"], pat, pos, d["response"]))
            elif pat is not None:
                late.append((d["question_idx"], d["entity"], pat, pos, len(d["response"])))

        print(f"ABSTAINED: {len(abst)}")
        for q, en, pat, pos, resp in abst:
            print(f"   qidx={q:4d}  {en[:36]:36s} | {pat!r} @char{pos}")
            if args.full:
                print(f"       {resp}")
        print()
        print(f"phrase matched but AFTER char {CUTOFF_CHARS} -> KEPT as answer: {len(late)}")
        for q, en, pat, pos, L in late:
            print(f"   qidx={q:4d}  {en[:36]:36s} | {pat!r} @char{pos} of {L}")
        print()


if __name__ == "__main__":
    main()