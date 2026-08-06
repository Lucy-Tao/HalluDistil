"""
sample_and_filter_100_entities.py -- one-time script: randomly sample N
entities (default 100) out of the already-completed 183-entity run, back
up the original files, and filter every downstream file down to the
sampled subset.

WHY THIS IS PURE FILTERING, NOT RE-RUNNING ANYTHING:
  - Abstention (compute_abstention_rate.py's output) is a per-entity
    property -- whether an entity was sampled into this smaller run
    doesn't change whether it was abstained on. The existing
    abstention_*.jsonl files already cover all 183 entities and remain
    fully valid for any subset; NOT touched by this script.
  - "Both teacher and student answered" (filter_longform_questions.py's
    output, answered_both.jsonl) is likewise a per-entity property. Any
    entity satisfying it in the full 183-entity run still satisfies it
    when restricted to a 100-entity sample -- the sampled-and-still-
    qualifying set is GUARANTEED to be a subset of the original
    answered_both.jsonl, so filtering that file by question_idx is
    sufficient -- no need to re-run filter_longform_questions.py.
  - Claim decomposition + correctness verification (decompose_and_
    verify.py's claims_*.jsonl) were already computed for every entity
    in the original answered_both.jsonl. Since the new sample's
    qualifying entities are a subset of that, their claims are already
    sitting in claims_teacher.jsonl / claims_student.jsonl -- filtering
    by question_idx is sufficient, no need to re-call GPT-5-mini (saves
    both time and API cost).

This means compute_abstention_rate.py, filter_longform_questions.py, and
decompose_and_verify.py are NOT run again and NOT modified by this change
-- only their existing OUTPUT files get filtered down.

Originals are backed up with a ".full183.bak" suffix before being
overwritten, so nothing is lost if you need to go back to the full run.

Usage
-----
  python sample_and_filter_100_entities.py \\
      --data_dir gen_longform_data \\
      --teacher_gen gen_factscore_bio_Qwen3-14B.jsonl \\
      --student_gen gen_factscore_bio_Qwen3-4B-Instruct-2507.jsonl \\
      --answered_both answered_both.jsonl \\
      --claims_teacher claims_teacher.jsonl \\
      --claims_student claims_student.jsonl \\
      --n 100 \\
      --random_seed 42

Output (in --data_dir):
  - sampled_100_entities.jsonl -- the subset spec itself (question_idx +
    entity per line), reusable elsewhere (e.g. as
    generate_longform_responses.py's --question_idx_subset input for a
    future distilled-model run restricted to this same 100).
  - Each of the 5 input files is overwritten in place with the filtered
    version; the pre-filter original is saved alongside as
    "<original_name>.full183.bak".
"""
import argparse
import json
import os
import random
import shutil


def backup_and_filter_by_question_idx(path, keep_idx, label):
    """Back up `path` to `path + '.full183.bak'`, then overwrite `path`
    with only the lines whose question_idx is in keep_idx. Returns the
    number of lines kept."""
    if not os.path.exists(path):
        print(f"  [{label}] SKIPPED -- file not found: {path}")
        return 0

    backup_path = path + ".full183.bak"
    if os.path.exists(backup_path):
        print(f"  [{label}] backup already exists at {backup_path} -- "
              f"NOT overwriting it (assuming a previous run of this script "
              f"already backed up the true original). Filtering {path} as-is.")
    else:
        shutil.copy2(path, backup_path)
        print(f"  [{label}] backed up {path} -> {backup_path}")

    with open(backup_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    filtered = [r for r in records if r["question_idx"] in keep_idx]

    with open(path, "w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  [{label}] {len(filtered)}/{len(records)} lines kept -> {path}")
    return len(filtered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True,
                         help="directory containing all the files below "
                              "(paths given relative to this)")
    parser.add_argument("--teacher_gen", required=True)
    parser.add_argument("--student_gen", required=True)
    parser.add_argument("--answered_both", required=True)
    parser.add_argument("--claims_teacher", required=True)
    parser.add_argument("--claims_student", required=True)
    parser.add_argument("--n", type=int, default=100,
                         help="how many entities to sample (default 100)")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    def p(name):
        return os.path.join(args.data_dir, name)

    # Sample question_idx from the FULL 183-entity teacher_gen file
    # (assumed to be the not-yet-backed-up original on first run; if this
    # script has already run once, teacher_gen is already the filtered
    # 100-entity version, so re-running would sample from a smaller pool
    # -- if that happens, restore from the .full183.bak file first).
    teacher_gen_path = p(args.teacher_gen)
    with open(teacher_gen_path, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    all_idx = sorted(r["question_idx"] for r in all_records)

    if len(all_idx) < args.n:
        print(f"WARNING: {teacher_gen_path} only has {len(all_idx)} entities, "
              f"fewer than --n={args.n}. Either this script already ran "
              f"once (restore {teacher_gen_path}.full183.bak first if you "
              f"want to resample from the full 183), or the file itself "
              f"has fewer than expected entities.")

    rng = random.Random(args.random_seed)
    sample_n = min(args.n, len(all_idx))
    sampled_idx = set(rng.sample(all_idx, sample_n))

    print(f"Sampled {len(sampled_idx)} entities (seed={args.random_seed}) "
          f"from {len(all_idx)} total in {teacher_gen_path}")

    # Save the subset spec itself, keyed to entity name too (reusable as
    # --question_idx_subset input elsewhere).
    subset_path = p(f"sampled_{sample_n}_entities.jsonl")
    entity_by_idx = {r["question_idx"]: r["entity"] for r in all_records}
    with open(subset_path, "w", encoding="utf-8") as f:
        for idx in sorted(sampled_idx):
            f.write(json.dumps(
                {"question_idx": idx, "entity": entity_by_idx.get(idx)},
                ensure_ascii=False,
            ) + "\n")
    print(f"Wrote subset spec -> {subset_path}")

    print("\nFiltering files (originals backed up with .full183.bak):")
    backup_and_filter_by_question_idx(teacher_gen_path, sampled_idx, "teacher_gen")
    backup_and_filter_by_question_idx(p(args.student_gen), sampled_idx, "student_gen")
    # answered_both / claims files: filtering by the SAME sampled_idx is
    # correct and sufficient -- see module docstring for why the
    # "still-qualifying" entities are guaranteed to already be a subset
    # of what's in these files, no re-computation needed.
    backup_and_filter_by_question_idx(p(args.answered_both), sampled_idx, "answered_both")
    backup_and_filter_by_question_idx(p(args.claims_teacher), sampled_idx, "claims_teacher")
    backup_and_filter_by_question_idx(p(args.claims_student), sampled_idx, "claims_student")

    print("\nDone. compute_abstention_rate.py, filter_longform_questions.py, "
          "and decompose_and_verify.py were NOT re-run -- their existing "
          "outputs were filtered in place, per the module docstring.")


if __name__ == "__main__":
    main()