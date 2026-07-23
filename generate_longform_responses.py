"""
generate_longform_responses.py — Phase 1 (long-form): generate ONE
biography per entity from a given model, for the FActScore(Bio) dataset.

This is a long-form sibling of generate_responses.py, kept as a SEPARATE
script rather than adding a branch to generate_responses.py, because the
two pipelines diverge in a way that doesn't collapse into one clean
parameterization:
  - generate_responses.py's whole design is "1 low-temp sample + N
    high-temp samples for semantic-entropy clustering". For the
    paragraph-length entropy protocol we're following (Farquhar et al.
    2024), that clustering works completely differently (claim-level
    decomposition, not whole-response entailment) and isn't implemented
    yet — this script only produces ONE response per entity, no high-temp
    sampling at all. Forcing generate_responses.py to carry unused
    high-temp-sampling machinery (e.g. via --n_high_temp_samples 0) would
    be more confusing than a dedicated, single-purpose script.
  - Response length: short-form generation uses a small max_new_tokens
    (sample_responses' own default is 50) and semantic_utils.py's
    DEFAULT_STOP_SEQUENCES, which stops generation at a single "\\n" —
    both are wrong for biography generation (multi-paragraph, needs a much
    larger token budget) and MUST be overridden here.

Despite being a separate script, generation itself still goes through
semantic_utils.sample_responses() — the shared, maintained generation
path — NOT a hand-rolled model.generate() call. See distill.py's history
of generate_teacher_responses() and build_single_prompt_dataset()
independently maintaining their own generate() calls (and drifting out of
sync — a fix like enable_thinking=False landing in one but not the other)
for why that pattern is worth avoiding here.

Checkpointed incrementally (one JSON line per entity) — safe to resume
after a SLURM timeout/crash, same pattern as generate_responses.py /
filter_questions.py.

Usage
-----
NOTE ON --output_dir: point this at somewhere under your home folder
(e.g. ~/gen_longform), NOT /scratch-ssd/ms25yt/... — scratch-ssd is
node-local and per-job (see pipeline_paths.py / config.py notes elsewhere
in this repo): it can vanish or become unreadable once the SLURM job
ends or if a later job lands on a different node. These jsonl files are
the actual generation outputs you need to keep and read back later (for
FActScore scoring, inspection, etc.), not intermediate scratch data, so
they belong somewhere persistent and node-independent — the home
directory, same as where the conda env itself lives.

  # Teacher
  python generate_longform_responses.py \\
      --model_role teacher \\
      --n_samples 183 \\
      --output_dir ~/gen_longform

  # Base student (before distillation)
  python generate_longform_responses.py \\
      --model_role student \\
      --n_samples 183 \\
      --output_dir ~/gen_longform

  # Distilled student — point --model_name at the checkpoint (the
  # checkpoint itself can still live on scratch-ssd if it's only read
  # from within a job pinned to that node; that's a different concern
  # from where THIS script's output goes), and set --run_tag so this
  # doesn't overwrite the base-student run above (both use
  # --model_role student, so the output filename would otherwise clash).
  # --question_idx_subset restricts to the entities where both teacher
  # and base student already answered, per answered_both.jsonl.
  python generate_longform_responses.py \\
      --model_role student \\
      --n_samples 183 \\
      --model_name /scratch-ssd/ms25yt/models/factscore_bio_distilled_student \\
      --run_tag distilled \\
      --question_idx_subset gen_longform_data/answered_both.jsonl \\
      --output_dir ~/gen_longform

Output file: {output_dir}/gen_factscore_bio_{model_short}[_{run_tag}].jsonl
Each line: {
    "question_idx": int,
    "entity": str,
    "prompt": str,          # exact prompt used
    "response": str,        # single T=0.1 generation
}
"""
import argparse
import json
import os

import torch

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
    parser.add_argument("--n_samples", type=int, default=183,
                         help="number of entities to generate for "
                              "(default 183 = the full FActScore labeled set)")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser("~/gen_longform"),
        help="Where to write the output jsonl. Defaults to ~/gen_longform "
             "(your home folder) — deliberately NOT scratch-ssd, since "
             "scratch-ssd is node-local/per-job and these outputs need to "
             "persist and be readable regardless of which node a later "
             "job lands on.",
    )
    parser.add_argument("--model_name", type=str, default=None,
                         help="override cfg.teacher_model_name / "
                              "cfg.student_model_name for this run — use "
                              "this to point --model_role student at a "
                              "distilled checkpoint path")
    parser.add_argument("--run_tag", type=str, default=None,
                         help="optional suffix on the output filename "
                              "(e.g. 'distilled') so a distilled-student "
                              "run doesn't overwrite the base-student run's "
                              "output — both use --model_role student")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                         help="override cfg.factscore_max_new_tokens for "
                              "this run only")
    parser.add_argument("--question_idx_subset", type=str, default=None,
                         help="path to a jsonl file with a 'question_idx' "
                              "field per line (e.g. answered_both.jsonl) -- "
                              "if given, ONLY generate for these question_idx "
                              "instead of the first --n_samples of the full "
                              "183-entity dataset. Use this for the distilled "
                              "student, which should only be evaluated on the "
                              "entities where both teacher and base student "
                              "already answered.")
    parser.add_argument("--random_n_entities", type=int, default=None,
                         help="randomly sample this many entities (by "
                              "question_idx, fixed seed via --random_seed) "
                              "out of the full 183-entity dataset, instead "
                              "of generating for all of them. Mutually "
                              "exclusive with --question_idx_subset -- if "
                              "you need this AND a subset restriction "
                              "together, generate the subset file yourself "
                              "and pass it via --question_idx_subset "
                              "instead.")
    parser.add_argument("--random_seed", type=int, default=42,
                         help="seed for --random_n_entities sampling")
    args = parser.parse_args()

    cfg.dataset = "factscore_bio"

    model_name = args.model_name or (
        cfg.teacher_model_name if args.model_role == "teacher" else cfg.student_model_name
    )
    model_short = short_model_name(model_name)
    max_new_tokens = args.max_new_tokens or cfg.factscore_max_new_tokens

    os.makedirs(args.output_dir, exist_ok=True)
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    ckpt_path = os.path.join(
        args.output_dir,
        f"gen_factscore_bio_{model_short}{tag_suffix}.jsonl",
    )

    print(f"{'='*60}")
    print("LONG-FORM GENERATION (FActScore Bio) — single sample per entity")
    print(f"  model_role:      {args.model_role}")
    print(f"  model_name:      {model_name}")
    print(f"  n_samples:       {args.n_samples}")
    print(f"  max_new_tokens:  {max_new_tokens}")
    print(f"  checkpoint file: {ckpt_path}")
    print(f"{'='*60}\n")

    items = load_dataset_items("factscore_bio", num_samples=args.n_samples)
    print(f"Loaded {len(items)} entity prompt(s).")

    # items is a plain list indexed 0..N-1 by position, which doubles as
    # question_idx everywhere else in this pipeline (see enumerate() below)
    # -- so restricting to a subset means keeping only the (question_idx,
    # item) pairs whose index is in the subset, not just truncating the list.
    if args.question_idx_subset and args.random_n_entities is not None:
        raise SystemExit(
            "--question_idx_subset and --random_n_entities are mutually "
            "exclusive -- see --help for why."
        )

    if args.question_idx_subset:
        with open(args.question_idx_subset, "r", encoding="utf-8") as f:
            wanted_idx = {json.loads(line)["question_idx"] for line in f if line.strip()}
        items_indexed = [(i, item) for i, item in enumerate(items) if i in wanted_idx]
        missing = wanted_idx - {i for i, _ in items_indexed}
        if missing:
            print(f"WARNING: {len(missing)} question_idx from "
                  f"{args.question_idx_subset} not found in the full "
                  f"183-entity dataset (out of range?): {sorted(missing)[:10]}"
                  f"{'...' if len(missing) > 10 else ''}")
        print(f"--question_idx_subset applied: restricting to "
              f"{len(items_indexed)}/{len(items)} entities from "
              f"{args.question_idx_subset}")
    elif args.random_n_entities is not None:
        import random
        all_idx = list(range(len(items)))
        rng = random.Random(args.random_seed)
        if args.random_n_entities >= len(all_idx):
            print(f"--random_n_entities={args.random_n_entities} >= "
                  f"{len(all_idx)} available -- using all of them.")
            sampled_idx = set(all_idx)
        else:
            sampled_idx = set(rng.sample(all_idx, args.random_n_entities))
        items_indexed = [(i, item) for i, item in enumerate(items) if i in sampled_idx]
        print(f"--random_n_entities applied (seed={args.random_seed}): "
              f"{len(items_indexed)}/{len(items)} entities sampled. "
              f"question_idx: {sorted(sampled_idx)}")
    else:
        items_indexed = list(enumerate(items))

    done_indices = load_done_indices(ckpt_path)
    remaining = [(i, item) for i, item in items_indexed if i not in done_indices]
    print(f"Checkpoint: {len(done_indices)} entity(ies) already done, "
          f"{len(remaining)} remaining.")

    if not remaining:
        print("Nothing to do — all entities already generated.")
        return

    print(f"Loading model: {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    with open(ckpt_path, "a", encoding="utf-8") as ckpt_f:
        for question_idx, item in remaining:
            print(f"[{question_idx}] {item['entity']}")

            # temperature=0.1 for now, matching this project's existing
            # "low-temp" convention for stable single-sample generation
            # elsewhere in the codebase (SimpleQA's low_temp_response etc.)
            # — NOT the temp=0.7 FActScore's own authors used when they
            # generated their InstructGPT/ChatGPT reference bios. Left as
            # a parameter to revisit once we've seen actual outputs; not
            # fixed on purpose.
            #
            # stop_sequences=None: sample_responses()'s DEFAULT_STOP_SEQUENCES
            # includes a bare "\n", which is correct for SimpleQA's one-line
            # answers but would truncate a multi-paragraph bio after its
            # first line break. Long-form generation must not inherit that
            # default.
            response = sample_responses(
                model, tokenizer, item["prompt"],
                n_samples=1, temperature=0.1,
                max_new_tokens=max_new_tokens,
                stop_sequences=None,
            )[0]

            record = {
                "question_idx": question_idx,
                "entity": item["entity"],
                "prompt": item["prompt"],
                "response": response,
            }
            ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()

            print(f"  -> {len(response)} chars generated")

    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Wrote to {ckpt_path}")


if __name__ == "__main__":
    main()