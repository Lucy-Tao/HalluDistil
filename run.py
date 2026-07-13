"""
run.py — Single entry point for all pipeline stages.

Usage
-----
  # Multi-prompt distillation (default: all prompts in the dataset)
  python run.py --mode distill
  python run.py --mode distill --dataset simpleqa --n_samples 200

  # Single-prompt distillation (specify question_idx + n_repeats)
  python run.py --mode distill --question_idx 17 --n_repeats 32
  python run.py --mode distill --question_idx 17 --n_repeats 32 --forced_answer B

  # Filtered-questions distillation (train on a specific set of question
  # indices, e.g. the "both teacher and base student high-entropy" set
  # found by filter_questions.py --mode scan; teacher responses are read
  # from that scan file, not regenerated)
  python run.py --mode distill --scan_file figures/scan_simpleqa_14Bto4B.json \
                 --question_indices 3 17 42 58

  # Visualize semantic-cluster distributions for one question
  python run.py --mode visualize --question_idx 0

  # Visualize reusing teacher/base responses from a scan file (faster —
  # only distilled student model is loaded and sampled fresh)
  python run.py --mode visualize --question_idx 1 \
      --scan_file figures/scan_test/scan_simpleqa_14Bto4B.json

  # Distill then visualize in one call
  python run.py --mode all --question_idx 0

Config overrides (no need to edit config.py for quick changes)
--------------------------------------------------------------
  python run.py --mode distill --teacher Qwen/Qwen3-32B --student Qwen/Qwen3-7B
"""

import argparse
from config import cfg
from model_utils import short_model_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distillation pipeline — hallucination transfer study"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["distill", "visualize", "all"],
        help=(
            "distill   : teacher generation + student SFT training\n"
            "visualize : 4-panel semantic-cluster plot for one question\n"
            "all       : distill then visualize"
        ),
    )

    # ── Question selection ─────────────────────────────────────
    parser.add_argument(
        "--question_idx", type=int, default=None,
        help="Question index (0-indexed). "
             "For distill mode: if set (and --question_indices is NOT set), "
             "single-prompt mode (distill only this question, repeated "
             "--n_repeats times). If neither is set, multi-prompt mode "
             "(use --n_samples prompts from the dataset). "
             "For visualize mode: which question to visualize (defaults to 0)."
    )

    # ── Single-prompt distillation options ─────────────────────
    parser.add_argument(
        "--n_repeats", type=int, default=32,
        help="Single-prompt distill mode only: how many copies of the single "
             "(prompt, response) pair to create for training. Default: 32"
    )
    parser.add_argument(
        "--forced_answer", type=str, default=None,
        help="Single-prompt distill mode only: manually specify the teacher's "
             "response instead of generating one. E.g. --forced_answer B"
    )

    # ── Filtered-questions distillation options ─────────────────
    parser.add_argument(
        "--question_indices", type=int, nargs="+", default=None,
        help="Filtered-questions distill mode: a specific list of question "
             "indices to train on (e.g. from filter_questions.py's "
             "'both high entropy' filter). Mutually exclusive with "
             "--question_idx. Requires --scan_file. The teacher model is "
             "NOT loaded in this mode — its responses to these questions "
             "are read from --scan_file instead of being generated."
    )
    parser.add_argument(
        "--scan_file", type=str, default=None,
        help="Filtered-questions distill mode only: path to a "
             "filter_questions.py scan output JSON to read teacher "
             "low-temperature responses from. Required when "
             "--question_indices is set."
    )

    # ── Batch options ──────────────────────────────────────────
    # (batch_entropy mode removed — use filter_questions.py --mode scan instead)

    # ── Config overrides ───────────────────────────────────────
    parser.add_argument("--teacher",   type=str,   default=None,
                        help="Override cfg.teacher_model_name")
    parser.add_argument("--student",   type=str,   default=None,
                        help="Override cfg.student_model_name")
    parser.add_argument("--dataset",   type=str,   default=None,
                        choices=["truthfulqa", "gpqa", "mmlu_pro", "simpleqa"],
                        help="Override cfg.dataset")
    parser.add_argument("--n_samples", type=int,   default=None,
                        help="Override cfg.num_train_samples (multi-prompt mode)")
    parser.add_argument("--epochs",    type=int,   default=None,
                        help="Override cfg.num_epochs")
    parser.add_argument("--lr",        type=float, default=None,
                        help="Override cfg.learning_rate")
    parser.add_argument("--model_tag", type=str, default=None,
                        help="Extra tag appended to the auto-derived "
                             "distilled_model_path, e.g. 'strict_full' or "
                             "'fewshot_high_entropy' — required whenever "
                             "you run more than one distillation variant "
                             "for the same dataset+student, or later runs "
                             "silently overwrite earlier ones' checkpoint.")

    return parser.parse_args()


def apply_overrides(args):
    """Push CLI flags into the global cfg singleton before any stage runs."""
    if args.teacher:   cfg.teacher_model_name = args.teacher
    if args.student:   cfg.student_model_name = args.student
    if args.dataset:   cfg.dataset            = args.dataset
    if args.n_samples: cfg.num_train_samples  = args.n_samples
    if args.epochs:    cfg.num_epochs         = args.epochs
    if args.lr:        cfg.learning_rate      = args.lr

    # Auto-derive the checkpoint save path from dataset + student name (+
    # optional --model_tag) so that running distill.py with different
    # --dataset / --student / --model_tag values never silently overwrites
    # a previous run's checkpoint.
    student_short = short_model_name(cfg.student_model_name)
    tag_suffix = f"_{args.model_tag}" if args.model_tag else ""
    cfg.distilled_model_path = (
        f"/scratch-ssd/ms25yt/models/{cfg.dataset}_{student_short}_student{tag_suffix}"
    )


def main():
    args = parse_args()
    apply_overrides(args)

    if args.question_idx is not None and args.question_indices is not None:
        raise ValueError(
            "--question_idx and --question_indices are mutually exclusive — "
            "use --question_idx for single-prompt mode or --question_indices "
            "for filtered-questions mode, not both."
        )
    if args.question_indices is not None and args.scan_file is None:
        raise ValueError("--question_indices requires --scan_file.")

    print(f"\n{'='*60}")
    print(f"Mode    : {args.mode}")
    print(f"Teacher : {cfg.teacher_model_name}")
    print(f"Student : {cfg.student_model_name}")
    print(f"Dataset : {cfg.dataset}")
    if args.question_indices is not None:
        print(f"Questions: {args.question_indices}  (filtered-questions mode, "
              f"scan_file={args.scan_file})")
    elif args.question_idx is not None:
        # Only show n_repeats label for distill/all mode — it's meaningless
        # (and was misleading) when the mode is purely visualize.
        if args.mode in ("distill", "all"):
            print(f"Question: {args.question_idx}  (single-prompt mode, "
                  f"n_repeats={args.n_repeats})")
        else:
            print(f"Question: {args.question_idx}")
    if args.scan_file is not None and args.mode in ("visualize", "all"):
        print(f"Scan file: {args.scan_file}  (teacher/base responses reused)")
    print(f"{'='*60}\n")

    if args.mode in ("distill", "all"):
        from distill import run_distillation
        run_distillation(
            question_idx=args.question_idx,
            n_repeats=args.n_repeats,
            forced_answer=args.forced_answer,
            question_indices=args.question_indices,
            scan_file=args.scan_file,
        )

    if args.mode in ("visualize", "all"):
        from visualize import run_visualization
        if args.question_indices is not None:
            vis_idx = args.question_indices[0]
        elif args.question_idx is not None:
            vis_idx = args.question_idx
        else:
            vis_idx = 0
        run_visualization(question_idx=vis_idx, scan_file=args.scan_file)


if __name__ == "__main__":
    main()