"""
distill_longform.py -- long-form (FActScore Bio) distillation on the
entities kept by filter_longform_questions.py (distill_targets.jsonl).

Training data is built straight from distill_targets.jsonl's
"teacher_response" field; the teacher model is never loaded here. The
prompt string for each entity is looked up from the original teacher
generation file by question_idx, since distill_targets.jsonl doesn't
carry it.

NODE NOTE: the checkpoint is saved to node-local scratch-ssd. The paired
generation step MUST run on the SAME node to read it back.

Usage
-----
  python distill_longform.py \\
      --distill_targets gen_longform_data/distill_targets.jsonl \\
      --teacher_gen gen_longform_data/gen_factscore_bio_Qwen3-32B.jsonl \\
      --output_dir /scratch-ssd/ms25yt/models/factscore_bio_distilled_student
"""
import argparse
import json
import os
import socket

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from config import cfg
from model_utils import load_model_and_tokenizer
from distill import prepare_sft_dataset


def build_training_data(distill_targets_path: str, teacher_gen_path: str) -> list[dict]:
    """Build (question_idx, prompt, response) records from the teacher_response
    field. The prompt is pulled from the teacher generation file by
    question_idx, since distill_targets.jsonl doesn't carry it."""
    with open(distill_targets_path, "r", encoding="utf-8") as f:
        targets = [json.loads(line) for line in f if line.strip()]
    with open(teacher_gen_path, "r", encoding="utf-8") as f:
        prompt_by_idx = {
            json.loads(line)["question_idx"]: json.loads(line)["prompt"]
            for line in f if line.strip()
        }

    training_data = []
    missing_prompt = []
    for record in targets:
        qidx = record["question_idx"]
        if qidx not in prompt_by_idx:
            missing_prompt.append(qidx)
            continue
        training_data.append({
            "question_idx": qidx,
            "prompt": prompt_by_idx[qidx],
            "response": record["teacher_response"],
        })

    if missing_prompt:
        raise ValueError(
            f"{len(missing_prompt)} question_idx from {distill_targets_path!r} "
            f"have no matching prompt in {teacher_gen_path!r}: "
            f"{missing_prompt[:10]}{'...' if len(missing_prompt) > 10 else ''}. "
            f"Check --teacher_gen points at the run distill_targets.jsonl was built from."
        )

    print(f"Built {len(training_data)} (prompt, teacher_response) pairs "
          f"from {distill_targets_path}")
    return training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distill_targets", required=True,
                        help="distill_targets.jsonl from filter_longform_questions.py")
    parser.add_argument("--teacher_gen", required=True,
                        help="teacher generation jsonl, used only to look up "
                             "each question_idx's prompt string")
    parser.add_argument("--output_dir", required=True,
                        help="where to save the distilled checkpoint "
                             "(put it under /scratch-ssd/...)")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="override cfg.num_epochs for this run only")
    parser.add_argument("--init_from_checkpoint", type=str, default=None,
                        help="continue training from an already-distilled "
                             "checkpoint instead of the base student. Note: "
                             "optimizer state and LR schedule restart, so this "
                             "is further fine-tuning, not a true resume.")
    parser.add_argument("--manifest_copy_to", type=str, default=None,
                        help="also copy the manifest JSON here (the primary "
                             "copy lives on node-local scratch-ssd and is "
                             "eventually lost). Point at gen_longform_data/.")
    args = parser.parse_args()

    if args.init_from_checkpoint and \
            os.path.abspath(args.output_dir) == os.path.abspath(args.init_from_checkpoint):
        print(f"WARNING: --output_dir == --init_from_checkpoint ({args.output_dir}); "
              f"the source checkpoint will be overwritten at the end of training.")

    hostname = socket.gethostname()
    print("=" * 60)
    print(f"LONG-FORM DISTILLATION -- node: {hostname}")
    print("=" * 60)
    print("NODE NOTE: checkpoint will only exist on this node's scratch-ssd; "
          "pin the generation step to this same node.")

    training_data = build_training_data(args.distill_targets, args.teacher_gen)

    print("\n[1/2] Loading student model...")
    if args.init_from_checkpoint:
        init_model_name = args.init_from_checkpoint
        print(f"  Continuing from checkpoint: {init_model_name} "
              f"(further fine-tuning, not a true resume)")
    else:
        init_model_name = cfg.student_model_name

    student_model, student_tokenizer = load_model_and_tokenizer(
        init_model_name, device_map=cfg.device_map
    )
    student_tokenizer.padding_side = "right"

    train_ds = prepare_sft_dataset(training_data, student_tokenizer)
    print(f"  Train: {len(train_ds)} samples (no eval split)")

    data_collator = DataCollatorForSeq2Seq(
        student_tokenizer, model=student_model, padding=True, pad_to_multiple_of=8
    )

    num_epochs = args.num_epochs if args.num_epochs is not None else cfg.num_epochs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    if len(train_ds) < effective_batch:
        steps_per_epoch = -(-len(train_ds) // cfg.batch_size)
        print(f"  NOTE: train set ({len(train_ds)}) < effective batch "
              f"({effective_batch}); each epoch flushes one partial step "
              f"({steps_per_epoch} micro-batches).")

    print("\n[2/2] Training...")
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)
    print(f"\nDistilled student saved -> {args.output_dir}")

    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
    manifest = {
        "node": hostname,
        "output_dir": args.output_dir,
        "distill_targets_source": args.distill_targets,
        "teacher_gen_source": args.teacher_gen,
        "n_training_samples": len(train_ds),
        "num_epochs": num_epochs,
        "batch_size_per_device": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "max_grad_norm": cfg.max_grad_norm,
        "question_indices_trained_on": [r["question_idx"] for r in training_data],
        "base_student_model": cfg.student_model_name,
        "init_from_checkpoint": args.init_from_checkpoint,
    }
    manifest_path = args.output_dir.rstrip("/") + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved -> {manifest_path}")

    if args.manifest_copy_to:
        os.makedirs(args.manifest_copy_to, exist_ok=True)
        checkpoint_tag = os.path.basename(args.output_dir.rstrip("/"))
        copy_path = os.path.join(args.manifest_copy_to, f"{checkpoint_tag}_manifest.json")
        with open(copy_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Manifest also copied -> {copy_path}")

    print(f"\nNODE: {hostname} -- pin the generation step to this node.")


if __name__ == "__main__":
    main()