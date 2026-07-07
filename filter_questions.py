"""
filter_questions.py — Scan teacher/base-student/distilled-student for
semantic entropy + correctness, filter for "both-models-uncertain"
question sets, and compute AUROC comparisons.

Two-phase workflow
------------------
Phase A — scan (`--mode scan`):
  Scan teacher AND base student over a batch of questions. For each
  question and each model:
    1. Sample 1 response at T=0.1 ("low-temp answer").
    2. Sample --n_high_temp_samples MORE responses at T=1.0, combined
       with the low-temp answer via get_semantic_entropy's fixed_response
       mechanism (default 9, so 10 total) -> semantic entropy via
       bidirectional entailment clustering. The low-temp answer is one of
       the 10 entropy samples, not a separate, wasted 11th generation.
    3. Judge correctness: does the low-temp answer match gold?
  This is fully symmetric for teacher and base student. The teacher's
  low-temp answer additionally serves as the distillation training
  target later (see distill.py's build_dataset_from_filtered_questions())
  — no separate generation needed for that.

  Then, for one or more semantic-entropy thresholds, find the question
  indices where BOTH teacher and base student have semantic_entropy >=
  threshold ("genuinely hard for both models, before any distillation").
  Reports what fraction of the scanned questions fall into this set at
  each threshold.

  Saves everything (full per-question records — including each model's
  low-temp answer, raw high-temp samples, cluster_probs, semantic_entropy,
  correctness — plus the threshold filter results) to a JSON file.
  distill.py reads the teacher's low-temp answers straight from this file
  for filtered-questions training; no re-generation needed.

Phase B — auroc (`--mode auroc`):
  Compute and plot an AUROC bar chart: how well does semantic entropy
  predict incorrectness, for up to 3 models (teacher, base student,
  distilled student)?

  The question set to evaluate on is configurable — NOT hardcoded to
  either the training set or the full dataset:
    --question_indices i1 i2 i3 ...   explicit list
    --scan_threshold T                use the "both >= T" set from --scan_file
    --n_samples N                     fallback: first N items in the
                                       dataset (the "evaluate on the
                                       full/random SimpleQA set" option)

  Teacher and base student records are REUSED from --scan_file when
  available (matched by question_idx) rather than re-scanned. The
  distilled student (if --distilled is given) is always scanned fresh,
  since it didn't exist at scan time.

Why this two-phase split
-------------------------
Filtering ("which questions are hard for both models") only needs teacher
+ base student and must happen BEFORE distillation (the distilled model
doesn't exist yet). AUROC comparison across all three models only makes
sense AFTER distillation. Splitting into two --mode values matches that
dependency directly, and lets you re-run Phase B against different
evaluation sets (training set, full dataset, a different threshold...)
without re-scanning teacher/base student each time.

A note on what AUROC computed on the training set does (and doesn't) show:
  Evaluating on the SAME questions the distilled student was just
  fine-tuned on tells you whether SFT had its intended effect (did the
  student actually become more confident on its own training data) — a
  manipulation check, not a generalization claim. It will not by itself
  show that distillation suppresses uncertainty on novel/held-out hard
  questions; that needs a separate evaluation set the distilled student
  was NOT trained on (pass a different --question_indices / --scan_threshold
  covering held-out questions to test that later).

Usage
-----
  # Phase A: scan + filter
  python filter_questions.py --mode scan --n_samples 500 \
      --thresholds 0.3 0.5 0.7 1.0

  # (then: use the printed "both_high" question indices for a threshold
  #  to distill — see run.py --mode distill --question_indices ...)

  # Phase B: AUROC on the same filtered set, now including the distilled student
  python filter_questions.py --mode auroc \
      --scan_file figures/scan_simpleqa_14Bto4B.json --scan_threshold 0.5 \
      --distilled /scratch-ssd/ms25yt/models/simpleqa_4B_student

  # Phase B: AUROC on the FULL (unfiltered) SimpleQA set instead
  python filter_questions.py --mode auroc --n_samples 500 \
      --distilled /scratch-ssd/ms25yt/models/simpleqa_4B_student

Output
------
  {output_dir}/scan_{dataset}_{pair_name}.json            — Phase A full scan + filter results
  {output_dir}/scan_{dataset}_{pair_name}_thresholds.png   — Phase A threshold chart
  {output_dir}/auroc_{dataset}_{pair_name}_{suffix}.json   — Phase B AUROC results
  {output_dir}/auroc_{dataset}_{pair_name}_{suffix}_bar.png — Phase B AUROC bar chart
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import cfg
from data_utils import load_dataset_items
from model_utils import load_model_and_tokenizer, pair_name, short_model_name
from semantic_utils import (
    cluster_by_entailment, compute_semantic_distribution, get_semantic_entropy,
    judge_correctness, load_local_llm_judge, load_nli_model, sample_responses,
)


# ══════════════════════════════════════════════════════════════
# Judge loading (shared between scan + auroc)
# ══════════════════════════════════════════════════════════════

def load_judge():
    """Load whichever entailment judge cfg.entailment_backend points to."""
    if cfg.entailment_backend == "llm":
        return load_local_llm_judge(cfg.entailment_llm_model_name)
    if cfg.entailment_backend == "deberta":
        return load_nli_model(cfg.nli_model_name)
    raise ValueError(f"Unknown cfg.entailment_backend={cfg.entailment_backend!r}")


# ══════════════════════════════════════════════════════════════
# Per-model scanning: SE (low-temp reused) + correctness
# ══════════════════════════════════════════════════════════════

def scan_model(
    model_name: str,
    items: list[dict],
    nli_model,
    nli_tokenizer,
    n_high_temp_samples: int,
    label: str,
    checkpoint_file: str | None = None,
) -> list[dict]:
    """
    For one model, process all prompts:
      1. Sample 1 response at T=0.1 ("low-temp answer").
      2. Sample n_high_temp_samples more responses at T=1.0, combined
         with the low-temp answer (n_high_temp_samples + 1 total) via
         get_semantic_entropy's fixed_response mechanism -> semantic
         entropy over ALL of them together. The low-temp answer is one
         of the entropy samples, not a separate, wasted generation.
      3. Judge correctness between the low-temp answer and gold, using
         cfg.entailment_backend.

    Returns one record per prompt. "low_temp_response" is saved so a
    later distillation run can read the teacher's response straight off
    disk (distill.py's build_dataset_from_filtered_questions()) instead
    of re-generating it.

    Args:
        checkpoint_file: if set, each record is appended to this JSON-lines
            file immediately after processing. On restart, already-processed
            question_idx values are skipped automatically — so a job that
            was interrupted (SLURM timeout, node failure) resumes from where
            it left off rather than starting over. The file is created on
            first run; if it already exists its contents are loaded as the
            starting state.
    """
    # ── Resume from checkpoint if available ───────────────────
    done_by_idx: dict[int, dict] = {}
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as _f:
            for line in _f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_by_idx[rec["question_idx"]] = rec
        if done_by_idx:
            print(f"  Checkpoint: resuming from {checkpoint_file} "
                  f"({len(done_by_idx)} question(s) already done, skipping them).")

    remaining = [it for it in items if it["question_idx"] not in done_by_idx]
    if not remaining:
        print(f"  All {len(items)} question(s) already in checkpoint — nothing to do.")
        records = [done_by_idx[it["question_idx"]] for it in items]
        return records

    print(f"\n  Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map=cfg.device_map)

    ckpt_fh = None
    if checkpoint_file is not None:
        ckpt_fh = open(checkpoint_file, "a", encoding="utf-8")

    try:
        for item in tqdm(remaining, desc=f"Scanning ({label})"):
            low_temp_response = sample_responses(
                model, tokenizer, item["prompt"],
                n_samples=1, temperature=0.1,
                max_new_tokens=cfg.semantic_max_new_tokens,
            )[0]

            se_result = get_semantic_entropy(
                model, tokenizer, item["prompt"],
                nli_model=nli_model, nli_tokenizer=nli_tokenizer,
                n_samples=n_high_temp_samples + 1,
                temperature=1.0,
                max_new_tokens=cfg.semantic_max_new_tokens,
                threshold=cfg.entailment_threshold,
                fixed_response=low_temp_response,
                judge_backend=cfg.entailment_backend,
                question=item["question"],
            )

            gold = item.get("answer", "")
            is_correct = judge_correctness(
                nli_model, nli_tokenizer, item["question"], gold, low_temp_response,
                backend=cfg.entailment_backend, threshold=cfg.entailment_threshold,
            )

            rec = {
                "question_idx":       item["question_idx"],
                "question":           item["question"],
                "prompt":             item["prompt"],
                "gold_answer":        gold,
                "low_temp_response":  low_temp_response,
                "is_correct":         is_correct,
                "semantic_entropy":   se_result["semantic_entropy"],
                "cluster_probs":      se_result["cluster_probs"],
                "predicted_response": se_result["predicted_response"],
                "raw_responses":      se_result["raw_responses"],
            }
            done_by_idx[item["question_idx"]] = rec

            if ckpt_fh is not None:
                ckpt_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ckpt_fh.flush()   # flush after every record — cheap and safe
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()

    del model
    torch.cuda.empty_cache()

    records = [done_by_idx[it["question_idx"]] for it in items]
    n_correct = sum(r["is_correct"] for r in records)
    mean_se = float(np.mean([r["semantic_entropy"] for r in records])) if records else 0.0
    print(f"  {label} done. Correct: {n_correct}/{len(records)}  Mean SE: {mean_se:.4f}")

    return records


# ══════════════════════════════════════════════════════════════
# AUROC computation
# ══════════════════════════════════════════════════════════════

def compute_auroc(records: list[dict]) -> float | None:
    """
    Compute AUROC: can semantic entropy predict whether the model answered
    incorrectly? Higher SE should correlate with incorrect answers.

    Labels: 1 = incorrect (hallucination), 0 = correct
    Scores: semantic_entropy (higher = more uncertain)

    Returns None if all answers are correct or all are incorrect (AUROC
    is undefined when only one class is present) — common on small or
    heavily-filtered subsets, not a bug.
    """
    labels = [0 if r["is_correct"] else 1 for r in records]
    scores = [r["semantic_entropy"] for r in records]

    if len(set(labels)) < 2:
        print(f"  WARNING: only one class present among {len(records)} records "
              f"(all correct or all incorrect). AUROC is undefined.")
        return None

    return roc_auc_score(labels, scores)


# ══════════════════════════════════════════════════════════════
# Multi-threshold filtering: questions hard for BOTH teacher and student
# ══════════════════════════════════════════════════════════════

def filter_high_entropy_questions(
    teacher_records: list[dict],
    student_records: list[dict],
    thresholds: list[float],
) -> list[dict]:
    """
    For each threshold, find question indices where BOTH teacher and base
    student have semantic_entropy >= threshold — genuinely hard questions
    where neither model is confident, before any distillation. Also
    reports what percentage of all scanned questions meet this bar at
    each threshold (useful both for picking a threshold and for reporting
    "X% of SimpleQA is hard for both models" in the writeup).
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


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def plot_auroc_bar(
    aurocs: dict[str, float | None],
    output_path: str,
    title_suffix: str = "",
):
    """
    Generic AUROC bar chart — one bar per entry in `aurocs`, e.g.
    {"Teacher\\n(Qwen3-14B)": 0.81, "Base Student\\n(Qwen3-4B)": 0.76,
     "Distilled Student\\n(Qwen3-4B)": 0.52}. Works for 2, 3, or more bars
    (entries with value None are skipped, matching compute_auroc()'s
    "undefined" case).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))

    palette = ["#4C72B0", "#C44E52", "#8172B2", "#55A868", "#CCB974"]
    names, values, colors = [], [], []
    for i, (name, val) in enumerate(aurocs.items()):
        if val is None:
            continue
        names.append(name)
        values.append(val)
        colors.append(palette[i % len(palette)])

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white",
                  linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC", fontsize=12)
    title = "Semantic Entropy as Hallucination Detector"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  AUROC bar chart saved -> {output_path}")


def plot_threshold_analysis(
    threshold_results: list[dict],
    n_total: int,
    output_path: str,
):
    """Bar chart: number (and %) of questions where both models are uncertain, per threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))

    thresholds     = [r["threshold"] for r in threshold_results]
    teacher_counts = [r["teacher_high"] for r in threshold_results]
    student_counts = [r["student_high"] for r in threshold_results]
    both_counts    = [r["both_high"] for r in threshold_results]

    x = np.arange(len(thresholds))
    width = 0.25

    ax.bar(x - width, teacher_counts, width, label="Teacher", color="#4C72B0", alpha=0.85)
    ax.bar(x, student_counts, width, label="Base Student", color="#C44E52", alpha=0.85)
    ax.bar(x + width, both_counts, width, label="Both", color="#55A868", alpha=0.85)

    ax.set_xlabel("Semantic Entropy Threshold", fontsize=11)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(f"Questions with SE \u2265 Threshold (total scanned: {n_total})",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"\u2265{t}" for t in thresholds])
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (count, r) in enumerate(zip(both_counts, threshold_results)):
        if count > 0:
            ax.text(x[i] + width, count + 0.5, f"{count}\n({r['both_high_pct']:.1f}%)",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Threshold analysis chart saved -> {output_path}")


# ══════════════════════════════════════════════════════════════
# Phase A: scan
# ══════════════════════════════════════════════════════════════

def run_scan(args):
    teacher_name = args.teacher or cfg.teacher_model_name
    student_name = args.student or cfg.student_model_name
    output_dir   = args.output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_role = getattr(args, "model_role", "both")  # "teacher" | "student" | "both"

    print("\n" + "=" * 60)
    print("PHASE A: SCAN  (semantic entropy + correctness)")
    print(f"  Dataset:              {args.dataset}")
    print(f"  Teacher:              {teacher_name}")
    print(f"  Student:              {student_name}")
    print(f"  Model role:           {model_role}")
    print(f"  Prompts:              {args.n_samples}")
    print(f"  High-temp samples:    {args.n_high_temp_samples} "
          f"(+1 low-temp = {args.n_high_temp_samples + 1} total per model per question)")
    print(f"  Judge backend:        {cfg.entailment_backend}")
    print(f"  Thresholds:           {args.thresholds}")
    print("=" * 60)

    items = load_dataset_items(args.dataset, num_samples=args.n_samples)
    for i, item in enumerate(items):
        item["question_idx"] = i

    pn     = pair_name(teacher_name, student_name)
    prefix = f"scan_{args.dataset}_{pn}"

    print("\nLoading entailment judge...")
    nli_model, nli_tokenizer = load_judge()

    teacher_records = None
    student_records = None

    if model_role in ("teacher", "both"):
        ckpt = os.path.join(output_dir, f"{prefix}_teacher_ckpt.jsonl") \
               if args.n_samples > 100 else None
        if ckpt:
            print(f"  Checkpoint file: {ckpt}")
        print("\n" + "-" * 40)
        print("Scanning teacher")
        print("-" * 40)
        teacher_records = scan_model(
            teacher_name, items, nli_model, nli_tokenizer,
            args.n_high_temp_samples, "Teacher",
            checkpoint_file=ckpt,
        )

    if model_role in ("student", "both"):
        ckpt = os.path.join(output_dir, f"{prefix}_student_ckpt.jsonl") \
               if args.n_samples > 100 else None
        if ckpt:
            print(f"  Checkpoint file: {ckpt}")
        print("\n" + "-" * 40)
        print("Scanning base student")
        print("-" * 40)
        student_records = scan_model(
            student_name, items, nli_model, nli_tokenizer,
            args.n_high_temp_samples, "Base Student",
            checkpoint_file=ckpt,
        )

    # ── If only one role was scanned, try to load the other from an
    # existing partial scan file or checkpoint so we can still write a
    # complete output file when both halves are available.
    json_path = os.path.join(output_dir, f"{prefix}.json")
    if model_role == "teacher" and student_records is None:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                _prev = json.load(f)
            if _prev.get("student_records"):
                student_records = _prev["student_records"]
                print(f"  Loaded existing student_records from {json_path}")
    if model_role == "student" and teacher_records is None:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                _prev = json.load(f)
            if _prev.get("teacher_records"):
                teacher_records = _prev["teacher_records"]
                print(f"  Loaded existing teacher_records from {json_path}")

    if teacher_records is None or student_records is None:
        missing = "teacher" if teacher_records is None else "student"
        print(f"\n  Only '{model_role}' scan completed. "
              f"'{missing}' records not yet available — "
              f"partial results NOT written to {json_path}. "
              f"Run with --model_role {missing} (or --model_role both) "
              f"to complete the scan and merge.")
        return

    thresh_results = filter_high_entropy_questions(
        teacher_records, student_records, args.thresholds
    )

    print("\n" + "-" * 60)
    print("THRESHOLD ANALYSIS: questions where BOTH models have SE >= threshold")
    print("-" * 60)
    print(f"{'Threshold':>10}  {'Teacher':>8}  {'Student':>8}  {'Both':>6}  {'% of total':>10}")
    print("-" * 60)
    for r in thresh_results:
        print(f"{r['threshold']:>10.2f}  {r['teacher_high']:>8}  {r['student_high']:>8}  "
              f"{r['both_high']:>6}  {r['both_high_pct']:>9.1f}%")

    output = {
        "metadata": {
            "dataset":              args.dataset,
            "teacher_model":        teacher_name,
            "student_model":        student_name,
            "n_prompts":            len(items),
            "n_high_temp_samples":  args.n_high_temp_samples,
            "judge_backend":        cfg.entailment_backend,
            "entailment_threshold": cfg.entailment_threshold,
        },
        "thresholds":      thresh_results,
        "teacher_records": teacher_records,
        "student_records": student_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full scan results saved -> {json_path}")
    print(f"  Next (AUROC on this filtered set): python filter_questions.py "
          f"--mode auroc --scan_file {json_path} --scan_threshold <T>")
    print(f"  Next (distill on this filtered set): python run.py --mode "
          f"distill --scan_file {json_path} --question_indices <ids from the table above>")

    plot_threshold_analysis(
        thresh_results, len(items),
        os.path.join(output_dir, f"{prefix}_thresholds.png"),
    )


# ══════════════════════════════════════════════════════════════
# Phase B: AUROC
# ══════════════════════════════════════════════════════════════

def _resolve_question_indices_and_items(args) -> tuple[list[int], dict[int, dict]]:
    """
    Figure out which question indices to evaluate AUROC on, and return
    (question_indices, items_by_idx) with enough per-question info
    (prompt/question/answer) for any model that needs fresh scanning.

    Priority: --question_indices > --scan_threshold (reads from --scan_file)
    > --n_samples (first N items in the dataset — the "full/random set" option).
    """
    if args.question_indices is not None:
        question_indices = list(args.question_indices)
    elif args.scan_threshold is not None:
        if args.scan_file is None:
            raise ValueError("--scan_threshold requires --scan_file.")
        with open(args.scan_file, "r", encoding="utf-8") as f:
            scan_data = json.load(f)
        match = [r for r in scan_data["thresholds"] if r["threshold"] == args.scan_threshold]
        if not match:
            available = [r["threshold"] for r in scan_data["thresholds"]]
            raise ValueError(
                f"threshold {args.scan_threshold} not found in {args.scan_file!r}. "
                f"Available thresholds: {available}"
            )
        question_indices = match[0]["question_indices"]
    else:
        # Full/random set: first n_samples items in the dataset.
        items = load_dataset_items(args.dataset, num_samples=args.n_samples)
        question_indices = list(range(len(items)))

    if not question_indices:
        raise ValueError("Resolved question_indices is empty — nothing to evaluate.")

    items = load_dataset_items(args.dataset, num_samples=max(question_indices) + 1)
    for i, item in enumerate(items):
        item["question_idx"] = i
    items_by_idx = {item["question_idx"]: item for item in items}

    return question_indices, items_by_idx


def _get_or_scan_records(
    model_name: str,
    label: str,
    question_indices: list[int],
    items_by_idx: dict[int, dict],
    scan_file: str | None,
    nli_model,
    nli_tokenizer,
    n_high_temp_samples: int,
    force_rescan: bool,
    record_key: str,
) -> list[dict]:
    """
    Reuse already-scanned records from scan_file (matched by question_idx)
    when available; otherwise scan model_name fresh on exactly
    question_indices. record_key is "teacher_records" or "student_records"
    — which half of scan_file to reuse from.
    """
    if scan_file is not None and not force_rescan:
        with open(scan_file, "r", encoding="utf-8") as f:
            scan_data = json.load(f)
        by_idx = {r["question_idx"]: r for r in scan_data.get(record_key, [])}
        if all(idx in by_idx for idx in question_indices):
            print(f"  Reusing {label} records for {len(question_indices)} "
                  f"question(s) from {scan_file} (no generation needed).")
            return [by_idx[idx] for idx in question_indices]
        missing = [idx for idx in question_indices if idx not in by_idx]
        preview = missing[:10] + (["..."] if len(missing) > 10 else [])
        print(f"  {label}: {len(missing)} question(s) not found in {scan_file} "
              f"({preview}) — scanning fresh for ALL requested questions "
              f"(not just the missing ones, to keep one consistent record set).")

    items = [items_by_idx[idx] for idx in question_indices]
    return scan_model(model_name, items, nli_model, nli_tokenizer,
                       n_high_temp_samples, label)


def run_auroc(args):
    teacher_name = args.teacher or cfg.teacher_model_name
    student_name = args.student or cfg.student_model_name
    output_dir   = args.output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    question_indices, items_by_idx = _resolve_question_indices_and_items(args)

    print("\n" + "=" * 60)
    print("PHASE B: AUROC  (semantic entropy as hallucination detector)")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Teacher:          {teacher_name}")
    print(f"  Student:          {student_name}")
    print(f"  Distilled:        {args.distilled or '(not evaluated)'}")
    print(f"  Question set:     {len(question_indices)} question(s)")
    print(f"  Scan file reuse:  {args.scan_file or '(none — scanning everything fresh)'}")
    print("=" * 60)

    print("\nLoading entailment judge...")
    nli_model, nli_tokenizer = load_judge()

    teacher_records = _get_or_scan_records(
        teacher_name, "Teacher", question_indices, items_by_idx,
        args.scan_file, nli_model, nli_tokenizer,
        args.n_high_temp_samples, args.force_rescan, "teacher_records",
    )
    student_records = _get_or_scan_records(
        student_name, "Base Student", question_indices, items_by_idx,
        args.scan_file, nli_model, nli_tokenizer,
        args.n_high_temp_samples, args.force_rescan, "student_records",
    )

    aurocs = {
        f"Teacher\n({short_model_name(teacher_name)})": compute_auroc(teacher_records),
        f"Base Student\n({short_model_name(student_name)})": compute_auroc(student_records),
    }

    distilled_records = None
    if args.distilled:
        print("\n" + "-" * 40)
        print("Scanning distilled student (always fresh — didn't exist at scan time)")
        print("-" * 40)
        items = [items_by_idx[idx] for idx in question_indices]
        distilled_records = scan_model(
            args.distilled, items, nli_model, nli_tokenizer,
            args.n_high_temp_samples, "Distilled Student",
        )
        aurocs[f"Distilled Student\n({short_model_name(args.distilled)})"] = compute_auroc(distilled_records)

    print("\n" + "-" * 40)
    print("AUROC RESULTS")
    print("-" * 40)
    for name, val in aurocs.items():
        flat_name = name.replace("\n", " ")
        print(f"  {flat_name}: {val:.4f}" if val is not None
              else f"  {flat_name}: undefined (one class only)")

    pn = pair_name(teacher_name, student_name)
    suffix = "with_distilled" if args.distilled else "no_distilled"
    prefix = f"auroc_{args.dataset}_{pn}_{suffix}"

    output = {
        "metadata": {
            "dataset":             args.dataset,
            "teacher_model":       teacher_name,
            "student_model":       student_name,
            "distilled_model":     args.distilled,
            "question_indices":    question_indices,
            "n_questions":         len(question_indices),
            "n_high_temp_samples": args.n_high_temp_samples,
            "judge_backend":       cfg.entailment_backend,
            "scan_file":           args.scan_file,
        },
        "auroc":             dict(aurocs),
        "teacher_records":   teacher_records,
        "student_records":   student_records,
        "distilled_records": distilled_records,
    }
    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full AUROC results saved -> {json_path}")

    plot_auroc_bar(
        aurocs, os.path.join(output_dir, f"{prefix}_bar.png"),
        title_suffix=f"({len(question_indices)} question(s))",
    )


# ══════════════════════════════════════════════════════════════
# Phase C (optional): rejudge — same sampled responses, different judge
# ══════════════════════════════════════════════════════════════
#
# scan_model() couples "generate responses" and "cluster them" into one
# call. That's fine for a normal run, but it makes a clean JUDGE
# comparison impossible: re-running scan with a different
# entailment_backend would also re-SAMPLE the teacher/student responses
# (temperature=1.0 is stochastic), so any difference in semantic_entropy
# between two scans would be a mix of "the judge disagreed" and "the
# random samples were different this time" — you can't tell which.
#
# rejudge_records() fixes this: it takes the raw_responses ALREADY saved
# in a scan-mode output file and re-runs ONLY the clustering step with a
# new judge, on the exact same text. No teacher/student model is loaded
# at all for this — it's pure entailment-judge compute, much cheaper than
# a full scan too.

def rejudge_records(
    records: list[dict],
    nli_model,
    nli_tokenizer,
    judge_backend: str,
    threshold: float,
) -> list[dict]:
    """
    Re-cluster each record's already-sampled raw_responses with a new
    judge, recomputing semantic_entropy / cluster_probs / predicted_response.
    Everything else in the record (low_temp_response, is_correct,
    gold_answer, question, prompt, question_idx) is copied through
    unchanged — correctness was judged against gold independently of
    which judge clusters the high-temp samples, so it doesn't need
    (and for a fair comparison, shouldn't need) to be redone.
    """
    new_records = []
    for r in records:
        clusters = cluster_by_entailment(
            r["raw_responses"], nli_model, nli_tokenizer, threshold,
            backend=judge_backend, question=r["question"],
        )
        dist = compute_semantic_distribution(r["raw_responses"], clusters)
        new_records.append({
            **r,
            "semantic_entropy":   dist["semantic_entropy"],
            "cluster_probs":      dist["cluster_probs"],
            "predicted_response": dist["predicted_response"],
        })
    return new_records


def run_rejudge(args):
    if args.scan_file is None:
        raise ValueError(
            "--mode rejudge requires --scan_file (the source of "
            "already-sampled raw_responses to re-cluster)."
        )
    output_dir = args.output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.scan_file, "r", encoding="utf-8") as f:
        scan_data = json.load(f)

    teacher_name = scan_data["metadata"]["teacher_model"]
    student_name = scan_data["metadata"]["student_model"]

    print("\n" + "=" * 60)
    print("PHASE C: REJUDGE  (same sampled responses, new entailment judge)")
    print(f"  Source scan_file:  {args.scan_file}")
    print(f"  Teacher:           {teacher_name}")
    print(f"  Student:           {student_name}")
    print(f"  New judge backend: {cfg.entailment_backend}"
          + (f" ({cfg.entailment_llm_model_name})" if cfg.entailment_backend == "llm" else ""))
    print(f"  Questions:         {len(scan_data['teacher_records'])}")
    print(f"  Thresholds:        {args.thresholds}")
    print("=" * 60)

    print("\nLoading entailment judge (no teacher/student model loaded for this mode)...")
    nli_model, nli_tokenizer = load_judge()

    print("\nRe-clustering teacher records...")
    teacher_records = rejudge_records(
        scan_data["teacher_records"], nli_model, nli_tokenizer,
        cfg.entailment_backend, cfg.entailment_threshold,
    )
    print("Re-clustering base student records...")
    student_records = rejudge_records(
        scan_data["student_records"], nli_model, nli_tokenizer,
        cfg.entailment_backend, cfg.entailment_threshold,
    )

    thresh_results = filter_high_entropy_questions(
        teacher_records, student_records, args.thresholds
    )

    print("\n" + "-" * 60)
    print("THRESHOLD ANALYSIS (recomputed under the new judge)")
    print("-" * 60)
    print(f"{'Threshold':>10}  {'Teacher':>8}  {'Student':>8}  {'Both':>6}  {'% of total':>10}")
    print("-" * 60)
    for r in thresh_results:
        print(f"{r['threshold']:>10.2f}  {r['teacher_high']:>8}  {r['student_high']:>8}  "
              f"{r['both_high']:>6}  {r['both_high_pct']:>9.1f}%")

    pn = pair_name(teacher_name, student_name)
    judge_tag = (cfg.entailment_llm_model_name if cfg.entailment_backend == "llm"
                 else "deberta")
    prefix = f"rejudge_{args.dataset}_{pn}_{short_model_name(judge_tag)}"

    output = {
        "metadata": {
            **scan_data["metadata"],
            "source_scan_file":  args.scan_file,
            "rejudge_backend":   cfg.entailment_backend,
            "rejudge_llm_model": cfg.entailment_llm_model_name
                                 if cfg.entailment_backend == "llm" else None,
        },
        "thresholds":      thresh_results,
        "teacher_records": teacher_records,
        "student_records": student_records,
    }
    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Rejudged results saved -> {json_path}")

    plot_threshold_analysis(
        thresh_results, len(teacher_records),
        os.path.join(output_dir, f"{prefix}_thresholds.png"),
    )


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scan for high-entropy questions and compute semantic-"
                     "entropy-as-hallucination-detector AUROC, following "
                     "Kuhn et al. (2023)."
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["scan", "auroc", "rejudge"])
    parser.add_argument("--model_role", type=str, default="both",
                        choices=["teacher", "student", "both"],
                        help="scan mode only: which model(s) to scan in this job. "
                             "'teacher' or 'student' scans only one model (lets you "
                             "run two jobs in parallel on two GPUs). 'both' (default) "
                             "scans teacher then student sequentially. When both halves "
                             "are done the full scan JSON is written automatically.")
    parser.add_argument("--dataset", type=str, default="simpleqa", choices=["simpleqa"])
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--student", type=str, default=None)
    parser.add_argument("--distilled", type=str, default=None,
                        help="auroc mode only: path/name of a distilled student "
                             "model to include as a third bar.")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="scan mode: number of prompts to scan. auroc mode: "
                             "fallback question-set size when neither "
                             "--question_indices nor --scan_threshold is given "
                             "(i.e. evaluate on the first N questions of the "
                             "full dataset).")
    parser.add_argument("--n_high_temp_samples", type=int, default=9,
                        help="High-temperature (T=1.0) samples per model per "
                             "question, ADDED TO the 1 low-temp (T=0.1) sample "
                             "for semantic entropy (default 9, giving 10 total "
                             "per model per question, matching Kuhn et al. 2023).")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
                        help="scan/rejudge mode: semantic-entropy thresholds to "
                             "report the 'both high' filter at.")
    parser.add_argument("--question_indices", type=int, nargs="+", default=None,
                        help="auroc mode only: explicit question indices to "
                             "evaluate AUROC on.")
    parser.add_argument("--scan_file", type=str, default=None,
                        help="Path to a scan-mode output JSON. auroc mode: "
                             "reuse teacher/student records from here when "
                             "possible (matched by question_idx) instead of "
                             "re-scanning. rejudge mode: REQUIRED — the source "
                             "of already-sampled raw_responses to re-cluster.")
    parser.add_argument("--scan_threshold", type=float, default=None,
                        help="auroc mode only: pull the 'both high entropy' "
                             "question_indices for this threshold from "
                             "--scan_file's threshold results, instead of "
                             "passing --question_indices explicitly.")
    parser.add_argument("--force_rescan", action="store_true",
                        help="auroc mode only: ignore --scan_file's saved "
                             "records and re-scan teacher/student from scratch.")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--entailment_backend", type=str, default=None,
                        choices=["deberta", "llm"],
                        help="Override cfg.entailment_backend for this run "
                             "only (doesn't touch config.py). Lets a "
                             "comparison script run scan/rejudge multiple "
                             "times with different judges without hand-"
                             "editing config.py between runs.")
    parser.add_argument("--entailment_llm_model_name", type=str, default=None,
                        help="Override cfg.entailment_llm_model_name for this "
                             "run only. Only used when the active "
                             "entailment_backend is 'llm' (either from this "
                             "flag or from config.py).")

    args = parser.parse_args()

    # CLI overrides for the judge config — applied here, once, before any
    # mode runs, so run_scan/run_auroc/run_rejudge and load_judge() (which
    # all just read cfg.entailment_backend / cfg.entailment_llm_model_name)
    # don't need to know these came from the CLI instead of config.py.
    if args.entailment_backend is not None:
        cfg.entailment_backend = args.entailment_backend
    if args.entailment_llm_model_name is not None:
        cfg.entailment_llm_model_name = args.entailment_llm_model_name

    if args.mode == "scan":
        run_scan(args)
    elif args.mode == "auroc":
        run_auroc(args)
    else:
        run_rejudge(args)


if __name__ == "__main__":
    main()