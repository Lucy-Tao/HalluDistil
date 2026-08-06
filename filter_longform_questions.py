"""
filter_longform_questions.py — Phase 1.6: keep entities where the TEACHER
answered (did not abstain), and write each surviving entity's question_idx,
entity name, and teacher response for the distillation step.

Only the teacher matters here: the teacher bio is the SFT target, so a
teacher abstention leaves no usable target. The student is not consulted at
this stage (it is initialised from base weights, not from any stored
generation), so this script no longer takes a student file.

Abstention is judged by the shared rule in abstention_rule.py, the same rule
used when blanking abstained responses for FActScore scoring.

Usage
-----
  python filter_longform_questions.py \
    --teacher_gen gen_longform_data/gen_factscore_bio_Qwen3-32B.jsonl \
    --output gen_longform_data/distill_targets.jsonl

Output: one jsonl line per kept entity:
  {
    "question_idx": int,
    "entity": str,
    "teacher_response": str,
  }
Plus a summary printed to stdout.
"""
import argparse
import json
from abstention_rule import is_abstained


def load_jsonl(path: str) -> dict[int, dict]:
    """Load a jsonl file keyed by question_idx."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["question_idx"]] = rec
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_gen", required=True,
                         help="teacher's raw generation jsonl "
                              "(from generate_longform_responses.py)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    teacher_gen = load_jsonl(args.teacher_gen)

    results = []
    n_abstained = 0
    for idx in sorted(teacher_gen):
        response = teacher_gen[idx]["response"]
        if is_abstained(response):
            n_abstained += 1
            continue
        results.append({
            "question_idx": idx,
            "entity": teacher_gen[idx]["entity"],
            "teacher_response": response,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_total = len(teacher_gen)
    print(f"\n{len(results)}/{n_total} entities kept (teacher did not abstain).")
    print(f"  {n_abstained} dropped because the teacher abstained.")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()