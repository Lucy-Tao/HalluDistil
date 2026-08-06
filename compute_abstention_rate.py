"""
compute_abstention_rate.py — Phase 1.5: flag abstained / non-answering
responses in the FActScore(Bio) long-form generation output, and report
the abstention (non-response) rate per model.

Classification rule: PURE PHRASE MATCH. A response is classified as
abstained if it contains any of these phrase variants (case-insensitive):
  - "no widely known" / "no widely recognized"
  - "not widely known" / "not widely recognized"  (negated form)
  - "no publicly known" / "no publicly available" / "no publicly documented"
  - "no well-known" / "no verified biography" / "no verified information"

Limitations (stated plainly, not hidden):
  - This is a heuristic, not a certified classifier, validated by hand
    against a couple dozen cases, not a full labeled abstention set.
  - Phrase list is tuned to Qwen3-14B/4B's specific refusal phrasing on
    this dataset — re-derive it by reading actual outputs if used on a
    different model or dataset, don't assume it transfers.

Usage
-----
  python compute_abstention_rate.py \\
      --input ~/gen_longform_data/gen_factscore_bio_Qwen3-14B.jsonl \\
      --output ~/gen_longform_data/abstention_Qwen3-14B.jsonl

  python compute_abstention_rate.py \\
      --input ~/gen_longform_data/gen_factscore_bio_Qwen3-4B-Instruct-2507.jsonl \\
      --output ~/gen_longform_data/abstention_Qwen3-4B-Instruct-2507.jsonl

Output: one jsonl line per entity:
  {
    "question_idx": int,
    "entity": str,
    "is_abstained": bool,
  }
Plus a summary printed to stdout (n abstained / n total, %).
"""
import argparse
import json
import re

REFUSAL_PATTERN = re.compile(
    r"no widely known|no widely recognized|no publicly known|no publicly available|"
    r"no publicly documented|not (?:be )?widely (?:known|recognized)|"
    r"no well-known|no verified (?:biography|information)",
    re.IGNORECASE,
)


def classify(question_idx: int, entity: str, response: str) -> dict:
    m = REFUSAL_PATTERN.search(response)
    if m is None:
        return {"question_idx": question_idx, "entity": entity, "is_abstained": False}
        
    early_cutoff = max(150, int(0.20 * len(response)))
    is_abstained = m.start() < early_cutoff

    return {
        "question_idx": question_idx,
        "entity": entity,
        "is_abstained": is_abstained,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help="generation jsonl (from generate_longform_responses.py)")
    parser.add_argument("--output", required=True,
                         help="where to write the per-entity abstention labels")
    args = parser.parse_args()

    results = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            results.append(classify(rec["question_idx"], rec["entity"], rec["response"]))

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_abstained = sum(r["is_abstained"] for r in results)
    n_total = len(results)
    print(f"{args.input}")
    print(f"  {n_abstained}/{n_total} abstained ({100 * n_abstained / n_total:.1f}%)")
    print(f"  wrote labels to {args.output}")


if __name__ == "__main__":
    main()