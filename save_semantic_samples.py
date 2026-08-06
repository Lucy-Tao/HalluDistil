"""
save_semantic_samples.py — End-to-end small-scale distillation test with
semantic entropy analysis for all three models (teacher, base student,
distilled student).

For each prompt in the dataset:
  1. Teacher:           sample N responses → NLI cluster → semantic entropy
  2. Base student:      sample N responses → NLI cluster → semantic entropy
  3. Distilled student: sample N responses → NLI cluster → semantic entropy
     (= the same student model AFTER SFT training on teacher's responses)

This script handles the full pipeline in one run:
  - Teacher generates SFT training data (1 response per prompt, for training)
  - Teacher also generates N responses per prompt (for SE analysis)
  - Base student generates N responses per prompt (before training)
  - Student is SFT-trained on teacher data
  - Distilled student generates N responses per prompt (after training)
  - NLI clustering + SE computation is done for all three models
  - Everything is saved to a single JSON file

Supports both full-dataset mode (--n_samples 50) and single-prompt mode
(--question_idx 7), so you can run the same script for a quick single-
question sanity check or a batch experiment.

Usage
-----
  # Batch mode: first 50 questions
  python save_semantic_samples.py --n_samples 50

  # Single-prompt mode: question 7 only
  python save_semantic_samples.py --question_idx 7

  # Override models
  python save_semantic_samples.py --n_samples 50 \\
      --teacher Qwen/Qwen3-14B --student Qwen/Qwen3-4B

  # Override number of semantic samples (default from cfg.num_semantic_samples)
  python save_semantic_samples.py --n_samples 50 --n_semantic_samples 10

Output
------
  {output_dir}/semantic_samples_{dataset}_{pair_name}[_q{idx}].json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from config import cfg
from data_utils import load_dataset_items
from distill import generate_teacher_responses, prepare_sft_dataset
from model_utils import load_model_and_tokenizer, pair_name, short_model_name
from semantic_utils import get_semantic_entropy, load_nli_model


# ══════════════════════════════════════════════════════════════
# Core: sample N responses + compute SE for one model on all prompts
# ══════════════════════════════════════════════════════════════

def collect_semantic_results(
    model, tokenizer, items: list[dict],
    nli_model, nli_tokenizer,
    n_semantic_samples: int,
    label: str,
) -> list[dict]:
    """
    For each prompt, sample N responses, cluster by NLI entailment, compute
    semantic entropy, and return one record per prompt.

    Args:
        label: human-readable name for progress bar (e.g. "Teacher", "Base student")

    Returns a list of dicts, one per prompt, each containing:
        raw_responses, cluster_probs, predicted_response, semantic_entropy
    """
    results = []
    for item in tqdm(items, desc=f"Sampling ({label})"):
        se_result = get_semantic_entropy(
            model, tokenizer, item["prompt"],
            nli_model=nli_model, nli_tokenizer=nli_tokenizer,
            n_samples=n_semantic_samples,
            temperature=cfg.semantic_sample_temperature,
            max_new_tokens=cfg.semantic_max_new_tokens,
            threshold=cfg.entailment_threshold,
        )
        results.append({
            "raw_responses":      se_result["raw_responses"],
            "cluster_probs":      se_result["cluster_probs"],
            "predicted_response": se_result["predicted_response"],
            "semantic_entropy":   se_result["semantic_entropy"],
        })
    return results


# ══════════════════════════════════════════════════════════════
# SFT training (minimal, reuses distill.py's prepare_sft_dataset)
# ══════════════════════════════════════════════════════════════

def train_student(student_model, student_tokenizer, teacher_data: list[dict]):
    """
    SFT-train the student model on teacher-generated (prompt, response) pairs.
    Modifies student_model IN PLACE (no checkpoint saved — the caller uses
    the trained model directly for post-training SE sampling).
    """
    student_tokenizer.padding_side = "right"

    dataset   = prepare_sft_dataset(teacher_data, student_tokenizer)
    train_cut = int(0.9 * len(dataset))
    train_ds  = dataset.select(range(train_cut))
    eval_ds   = dataset.select(range(train_cut, len(dataset)))
    print(f"  SFT dataset: train={len(train_ds)}, eval={len(eval_ds)}")

    data_collator = DataCollatorForSeq2Seq(
        student_tokenizer, model=student_model, padding=True, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir="/tmp/sft_temp",    # temporary, not saved permanently
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    trainer.train()

    # Switch back to left-padding for generation
    student_tokenizer.padding_side = "left"
    student_model.eval()
    print("  SFT training complete.")


# ══════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════

def run(
    dataset: str,
    teacher_model_name: str,
    student_model_name: str,
    items: list[dict],
    n_semantic_samples: int,
    output_dir: str,
    question_idx: int | None,
):
    print("\n" + "=" * 60)
    print(f"SEMANTIC SAMPLES COLLECTION")
    print(f"  Dataset:    {dataset} ({len(items)} prompts)")
    print(f"  Teacher:    {teacher_model_name}")
    print(f"  Student:    {student_model_name}")
    print(f"  N samples:  {n_semantic_samples} per prompt per model")
    print("=" * 60)

    # ── NLI model (small, stays resident throughout) ────────────
    print("\n[NLI] Loading entailment model...")
    nli_model, nli_tokenizer = load_nli_model(cfg.nli_model_name)

    # ── Phase 1: Teacher ──────────────────────────────────────
    print("\n[1/4] Loading teacher model...")
    teacher_model, teacher_tok = load_model_and_tokenizer(
        teacher_model_name, device_map=cfg.device_map
    )

    # 1a. Generate SFT training data (1 response per prompt)
    print("  Generating SFT training data...")
    prompts = [item["prompt"] for item in items]
    teacher_data = generate_teacher_responses(
        teacher_model, teacher_tok, prompts, dataset=dataset
    )

    # 1b. Sample N responses per prompt for SE analysis
    print(f"  Sampling {n_semantic_samples} responses per prompt for SE...")
    teacher_se = collect_semantic_results(
        teacher_model, teacher_tok, items,
        nli_model, nli_tokenizer, n_semantic_samples, "Teacher"
    )
    del teacher_model
    torch.cuda.empty_cache()
    print("  Teacher freed.")

    # ── Phase 2: Base student (BEFORE training) ───────────────
    print("\n[2/4] Loading base student model...")
    student_model, student_tok = load_model_and_tokenizer(
        student_model_name, device_map=cfg.device_map
    )

    print(f"  Sampling {n_semantic_samples} responses per prompt for SE (base)...")
    base_se = collect_semantic_results(
        student_model, student_tok, items,
        nli_model, nli_tokenizer, n_semantic_samples, "Base student"
    )

    # ── Phase 3: SFT training ────────────────────────────────
    print("\n[3/4] Training student on teacher's responses...")
    train_student(student_model, student_tok, teacher_data)

    # ── Phase 4: Distilled student (AFTER training) ──────────
    print(f"\n[4/4] Sampling {n_semantic_samples} responses per prompt "
          f"for SE (distilled)...")
    distilled_se = collect_semantic_results(
        student_model, student_tok, items,
        nli_model, nli_tokenizer, n_semantic_samples, "Distilled student"
    )
    del student_model
    torch.cuda.empty_cache()

    # ── Assemble per-prompt records ──────────────────────────
    records = []
    for i, item in enumerate(items):
        t_ent = teacher_se[i]["semantic_entropy"]
        b_ent = base_se[i]["semantic_entropy"]
        d_ent = distilled_se[i]["semantic_entropy"]

        records.append({
            "question_idx":    question_idx if question_idx is not None else i,
            "question":        item["question"],
            "gold_answer":     item.get("answer", ""),
            "teacher": {
                "responses":          teacher_se[i]["raw_responses"],
                "cluster_probs":      teacher_se[i]["cluster_probs"],
                "predicted_response": teacher_se[i]["predicted_response"],
                "semantic_entropy":   t_ent,
            },
            "base_student": {
                "responses":          base_se[i]["raw_responses"],
                "cluster_probs":      base_se[i]["cluster_probs"],
                "predicted_response": base_se[i]["predicted_response"],
                "semantic_entropy":   b_ent,
            },
            "distilled_student": {
                "responses":          distilled_se[i]["raw_responses"],
                "cluster_probs":      distilled_se[i]["cluster_probs"],
                "predicted_response": distilled_se[i]["predicted_response"],
                "semantic_entropy":   d_ent,
            },
            "entropy_gap_distilled_vs_teacher": d_ent - t_ent,
        })

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SEMANTIC ENTROPY SUMMARY")
    print("=" * 60)
    print(f"{'Q':>4}  {'Teacher':>10}  {'Base':>10}  {'Distilled':>10}  {'Gap(D-T)':>10}")
    print("-" * 60)
    for r in records:
        t = r["teacher"]["semantic_entropy"]
        b = r["base_student"]["semantic_entropy"]
        d = r["distilled_student"]["semantic_entropy"]
        g = r["entropy_gap_distilled_vs_teacher"]
        print(f"{r['question_idx']:>4}  {t:>10.4f}  {b:>10.4f}  {d:>10.4f}  {g:>+10.4f}")

    avg_t = sum(r["teacher"]["semantic_entropy"] for r in records) / len(records)
    avg_b = sum(r["base_student"]["semantic_entropy"] for r in records) / len(records)
    avg_d = sum(r["distilled_student"]["semantic_entropy"] for r in records) / len(records)
    print("-" * 60)
    print(f"{'AVG':>4}  {avg_t:>10.4f}  {avg_b:>10.4f}  {avg_d:>10.4f}  "
          f"{avg_d - avg_t:>+10.4f}")

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    pn = pair_name(teacher_model_name, student_model_name)
    if question_idx is not None:
        filename = f"semantic_samples_{dataset}_{pn}_q{question_idx}.json"
    else:
        filename = f"semantic_samples_{dataset}_{pn}_n{len(items)}.json"

    json_path = os.path.join(output_dir, filename)
    output = {
        "metadata": {
            "dataset":              dataset,
            "teacher_model":        teacher_model_name,
            "student_model":        student_model_name,
            "n_prompts":            len(items),
            "n_semantic_samples":   n_semantic_samples,
            "num_epochs":           cfg.num_epochs,
        },
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved -> {json_path}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Distill + sample N responses from teacher/base/distilled "
                     "student + compute semantic entropy for each."
    )
    parser.add_argument("--dataset", type=str, default="simpleqa",
                        choices=["simpleqa"],
                        help="Dataset (currently SimpleQA only, since SE "
                             "analysis requires open-ended generation).")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Override cfg.teacher_model_name")
    parser.add_argument("--student", type=str, default=None,
                        help="Override cfg.student_model_name")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of prompts to use (default 50). "
                             "Ignored if --question_idx is set.")
    parser.add_argument("--question_idx", type=int, default=None,
                        help="Single-prompt mode: only process this one "
                             "question (overrides --n_samples).")
    parser.add_argument("--n_semantic_samples", type=int, default=None,
                        help="Responses per prompt per model for SE "
                             "(default: cfg.num_semantic_samples)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.num_epochs for SFT training")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    teacher_model_name  = args.teacher or cfg.teacher_model_name
    student_model_name  = args.student or cfg.student_model_name
    n_semantic_samples  = args.n_semantic_samples or cfg.num_semantic_samples
    output_dir          = args.output_dir or cfg.output_dir
    if args.epochs:
        cfg.num_epochs = args.epochs

    # Load dataset items — single-prompt or batch
    if args.question_idx is not None:
        all_items = load_dataset_items(args.dataset,
                                       num_samples=args.question_idx + 1)
        items = [all_items[args.question_idx]]
        question_idx = args.question_idx
    else:
        items = load_dataset_items(args.dataset, num_samples=args.n_samples)
        question_idx = None

    run(
        dataset=args.dataset,
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        items=items,
        n_semantic_samples=n_semantic_samples,
        output_dir=output_dir,
        question_idx=question_idx,
    )


if __name__ == "__main__":
    main()