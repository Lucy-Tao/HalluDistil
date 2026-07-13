"""
generate_responses.py — Phase 1 of the (now decoupled) experiment
pipeline: generate and save RAW responses only. No entailment judging, no
semantic entropy, no correctness checking — that all happens in a
SEPARATE later phase that reads this script's output.

For each question:
    - 1 response at T=0.1  (low_temp_response — for correctness judging later)
    - N responses at T=1.0 (raw_responses — for semantic entropy clustering
      later; kept fully SEPARATE from low_temp_response)

Because judging is a separate step, this script only ever needs to load
ONE model (teacher OR student — never a model + a judge together), so a
single GPU is enough even for the 14B teacher.

Checkpointed incrementally (one JSON line per question) — safe to resume
after a timeout/crash.

Usage:
    python generate_responses.py \
        --model_role teacher \
        --dataset simpleqa \
        --prompt_style strict \
        --n_samples 4321 \
        --n_high_temp_samples 10 \
        --output_dir /scratch-ssd/ms25yt/gen_full

Output file: {output_dir}/gen_{dataset}_{model_short}_{prompt_style}.jsonl
Each line: {
    "question_idx": int,
    "question": str,
    "answer": str,              # gold answer
    "prompt": str,               # exact prompt used
    "low_temp_response": str,    # T=0.1, 1 sample
    "raw_responses": [str, ...]  # T=1.0, n_high_temp_samples samples
}
"""
import argparse
import json
import os

from config import cfg
from data_utils import load_dataset_items
from model_utils import load_model_and_tokenizer, short_model_name
from semantic_utils import sample_responses


def load_done_indices(ckpt_path: str) -> set[int]:
    """Read a checkpoint file (if it exists) and return the set of
    question_idx already completed, so a resumed run can skip them."""
    done = set()
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["question_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_role", type=str, required=True,
                         choices=["teacher", "student"])
    parser.add_argument("--dataset", type=str, default="simpleqa")
    parser.add_argument("--prompt_style", type=str, required=True,
                         choices=["strict", "fewshot"])
    parser.add_argument("--n_samples", type=int, required=True,
                         help="number of questions to generate for")
    parser.add_argument("--n_high_temp_samples", type=int, default=10,
                         help="number of T=1.0 samples per question, kept "
                              "separate from the single T=0.1 sample")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None,
                         help="override cfg.teacher_model_name / "
                              "cfg.student_model_name for this run")
    args = parser.parse_args()

    cfg.prompt_style = args.prompt_style
    cfg.dataset = args.dataset

    model_name = args.model_name or (
        cfg.teacher_model_name if args.model_role == "teacher" else cfg.student_model_name
    )
    model_short = short_model_name(model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.output_dir,
        f"gen_{args.dataset}_{model_short}_{args.prompt_style}.jsonl",
    )

    print(f"{'='*60}")
    print(f"PHASE 1: GENERATION ONLY (no judging)")
    print(f"  model_role:          {args.model_role}")
    print(f"  model_name:          {model_name}")
    print(f"  dataset:             {args.dataset}")
    print(f"  prompt_style:        {args.prompt_style}")
    print(f"  n_samples:           {args.n_samples}")
    print(f"  n_high_temp_samples: {args.n_high_temp_samples}")
    print(f"  checkpoint file:     {ckpt_path}")
    print(f"{'='*60}\n")

    items = load_dataset_items(args.dataset, num_samples=args.n_samples)
    print(f"Loaded {len(items)} question(s).")

    done_indices = load_done_indices(ckpt_path)
    remaining = [(i, item) for i, item in enumerate(items) if i not in done_indices]
    print(f"Checkpoint: {len(done_indices)} question(s) already done, "
          f"{len(remaining)} remaining.")

    if not remaining:
        print("Nothing to do — all questions already generated.")
        return

    print(f"Loading model: {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    with open(ckpt_path, "a", encoding="utf-8") as ckpt_f:
        for question_idx, item in remaining:
            print(f"[{question_idx}] {item['question'][:80]}")

            low_temp_response = sample_responses(
                model, tokenizer, item["prompt"],
                n_samples=1, temperature=0.1,
            )[0]

            raw_responses = sample_responses(
                model, tokenizer, item["prompt"],
                n_samples=args.n_high_temp_samples, temperature=1.0,
            )

            record = {
                "question_idx": question_idx,
                "question": item["question"],
                "answer": item["answer"],
                "prompt": item["prompt"],
                "low_temp_response": low_temp_response,
                "raw_responses": raw_responses,
            }
            ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()

    print(f"\nDone. Wrote to {ckpt_path}")


if __name__ == "__main__":
    main()