"""
select_threshold.py — Phase 3: given a teacher judged file and a base
student judged file (same prompt_style, both from judge_responses.py),
find questions where BOTH models have semantic_entropy >= threshold, for
a range of candidate thresholds, and save the question_idx list for
whichever threshold you pick.

Deliberately does NOT import filter_questions.py for this — that module
pulls in transformers/torch at import time (needed for its other
functions), even though filter_high_entropy_questions() itself is pure
Python (just set operations). Duplicating these few lines here keeps this
script genuinely lightweight — safe to run on the login node with no GPU
and near-instant startup, unlike anything that imports model-loading code.

Usage:
    python select_threshold.py \
        --teacher_file ~/SimpleQA/judged_data/judged_simpleqa_Qwen3-14B_strict.jsonl \
        --student_file ~/SimpleQA/judged_data/judged_simpleqa_Qwen3-4B-Instruct-2507_strict.jsonl \
        --thresholds 0.1 0.3 0.5 0.7 1.0 1.5 \
        --chosen_threshold 1.5 \
        --output_dir ~/SimpleQA/threshold_data \
        --tag strict
"""
import argparse
import json
import os


def filter_high_entropy_questions(
    teacher_records: list[dict],
    student_records: list[dict],
    thresholds: list[float],
) -> list[dict]:
    """
    For each threshold, find question indices where BOTH teacher and base
    student have semantic_entropy >= threshold. Also reports what
    percentage of all questions meet this bar at each threshold.

    Identical logic to filter_questions.py's function of the same name —
    duplicated here rather than imported, see module docstring above.
    """
    n_total = len(teacher_records)
    results = []
    for t in thresholds:
        teacher_high = {r["question_idx"] for r in teacher_records
                        if r["semantic_entropy"] >= t}
        student_high = {r["question_idx"] for r in student_records
                        if r["semantic_entropy"] >= t}
        both_high = sorted(teacher_high & student_high)

        results.append({
            "threshold":        t,
            "teacher_high":     len(teacher_high),
            "student_high":     len(student_high),
            "both_high":        len(both_high),
            "both_high_pct":    100.0 * len(both_high) / n_total if n_total else 0.0,
            "question_indices": both_high,
        })
    return results


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_file", type=str, required=True)
    parser.add_argument("--student_file", type=str, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+",
                         default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5])
    parser.add_argument("--chosen_threshold", type=float, required=True,
                         help="which threshold's question_idx list to save "
                              "for the high_entropy distillation variant")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True,
                         help="e.g. 'strict' or 'fewshot' — used to name the output file")
    args = parser.parse_args()

    teacher_records = load_records(args.teacher_file)
    student_records = load_records(args.student_file)
    print(f"Loaded {len(teacher_records)} teacher record(s), "
          f"{len(student_records)} student record(s).")

    teacher_idx = {r["question_idx"] for r in teacher_records}
    student_idx = {r["question_idx"] for r in student_records}
    if teacher_idx != student_idx:
        only_teacher = teacher_idx - student_idx
        only_student = student_idx - teacher_idx
        print(f"WARNING: teacher and student don't cover the exact same "
              f"question_idx set. Only in teacher: {len(only_teacher)}, "
              f"only in student: {len(only_student)}. "
              f"Proceeding with the intersection only for the threshold table.")

    results = filter_high_entropy_questions(teacher_records, student_records, args.thresholds)

    print(f"\n{'Threshold':>10} {'Teacher':>10} {'Student':>10} {'Both':>10} {'% of total':>12}")
    print("-" * 56)
    for r in results:
        print(f"{r['threshold']:>10} {r['teacher_high']:>10} {r['student_high']:>10} "
              f"{r['both_high']:>10} {r['both_high_pct']:>11.1f}%")

    chosen = next((r for r in results if r["threshold"] == args.chosen_threshold), None)
    if chosen is None:
        available = [r["threshold"] for r in results]
        raise ValueError(f"chosen_threshold={args.chosen_threshold} not in "
                          f"--thresholds {available}. Add it to --thresholds "
                          f"if you want to select it.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"high_entropy_{args.tag}_thr{args.chosen_threshold}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": args.chosen_threshold,
            "n_questions": len(chosen["question_indices"]),
            "question_indices": chosen["question_indices"],
        }, f, indent=2)

    print(f"\nSelected threshold={args.chosen_threshold}: "
          f"{len(chosen['question_indices'])} question(s) "
          f"({chosen['both_high_pct']:.1f}% of total)")
    print(f"Saved question_idx list -> {out_path}")


if __name__ == "__main__":
    main()
    