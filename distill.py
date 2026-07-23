"""
distill.py — SeqKD off-policy distillation.

Pipeline
--------
1. Teacher generates one response per prompt  ->  saved to data/teacher_responses.json
2. Student is fine-tuned on those (prompt, response) pairs via cross-entropy SFT.
3. Distilled student checkpoint is saved to cfg.distilled_model_path.

This is the simplest distillation baseline:
  - Pure sequence-level KD (no token-level KL divergence loss)
  - Off-policy only (GKD lambda = 0)
  - Full fine-tune (add LoRA via peft if VRAM is tight)

Usage
-----
  python distill.py
  python run.py --mode distill
"""

from __future__ import annotations

import json
import os

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import cfg
from data_utils import load_prompts
from model_utils import load_model_and_tokenizer, short_model_name


# ══════════════════════════════════════════════════════════════
# Step 0 — Save-path helpers
# ══════════════════════════════════════════════════════════════
#
# _multi_prompt_responses_path() is used by generate_teacher_responses()
# below (currently unreferenced by run_distillation() — see its docstring
# for why the "full multi-prompt regeneration" mode was retired — kept in
# case you want it back for a different dataset/use case later, not
# deleted without explicit confirmation).
#
# _single_prompt_response_path() is used by build_single_prompt_dataset()
# to persist the single-prompt-mode sample. The lookup counterpart that
# used to read these files back (load_teacher_distill_response(), for
# visualize.py's Panel 2) has been removed — visualize.py is being
# rewritten to read clusters/entropy straight from judge_responses.py's
# output instead of re-sampling and looking up a saved distillation
# sample this way.

def _multi_prompt_responses_path(dataset: str, teacher_model_name: str) -> str:
    """Path to the aggregate multi-prompt teacher-responses file."""
    teacher_short = short_model_name(teacher_model_name)
    return os.path.join(cfg.data_dir, f"teacher_responses_{dataset}_{teacher_short}.json")


def _single_prompt_response_path(
    dataset: str, teacher_model_name: str, question_idx: int
) -> str:
    """Path to a single-prompt-mode teacher-response file (one question_idx)."""
    teacher_short = short_model_name(teacher_model_name)
    return os.path.join(
        cfg.data_dir,
        f"teacher_response_single_{dataset}_{teacher_short}_q{question_idx:03d}.json",
    )


# ══════════════════════════════════════════════════════════════
# Step 1 — Teacher generation
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_teacher_responses(
    teacher_model,
    teacher_tokenizer,
    prompts: list[str],
    dataset: str,
) -> list[dict]:
    """
    Run teacher inference on every prompt and collect (prompt, response) pairs.

    One response is sampled per prompt at temperature=cfg.temperature.
    Results are saved to data/teacher_responses_{dataset}_{teacher}.json so
    they can be reused for semantic entropy analysis in Stage 1 without
    re-running inference, and so re-running on a different dataset or with
    a different teacher doesn't silently overwrite a previous run's data.

    Each saved record includes "question_idx" (the record's position in
    `prompts`, which is the same 0-indexed ordering load_dataset_items()
    uses) so visualize.py can look up the EXACT teacher sample that was
    used as distillation training data for a given question, instead of
    matching by list position alone.

    For dataset == "simpleqa", generation uses cfg.semantic_max_new_tokens
    instead of cfg.max_new_tokens. This keeps the distillation sample's
    generation settings identical to the ones used when later sampling
    additional responses for the semantic-entropy panels in visualize.py —
    otherwise the reused distillation sample could be generated under a
    different max_new_tokens budget than the rest of the N samples it gets
    clustered with, which could distort the semantic entropy estimate.
    For other datasets (MCQ), cfg.max_new_tokens is unchanged since there
    is no equivalent "combine with extra panel samples" step for MCQ.

    Args:
        dataset : dataset name (e.g. "gpqa"), used in the output filename.

    Returns:
        list of {"question_idx": int, "prompt": str, "response": str}
    """
    teacher_model.eval()
    data = []

    gen_max_new_tokens = (
        cfg.semantic_max_new_tokens if dataset == "simpleqa" else cfg.max_new_tokens
    )

    from semantic_utils import DEFAULT_STOP_SEQUENCES

    for idx, prompt in enumerate(tqdm(prompts, desc="Teacher generating")):
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False disables Qwen3's <think>...</think> reasoning
        # block, so the teacher's response is the actual short answer, not
        # a reasoning trace truncated by max_new_tokens. This keeps the SFT
        # training targets clean and consistent with the answer-only prompts
        # built in data_utils.py.
        try:
            text = teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = teacher_tokenizer(text, return_tensors="pt").to(teacher_model.device)

        output_ids = teacher_model.generate(
            **inputs,
            max_new_tokens=gen_max_new_tokens,
            temperature=cfg.temperature,
            do_sample=(cfg.temperature > 0),
            pad_token_id=teacher_tokenizer.pad_token_id,
            renormalize_logits=True,
            top_p=1.0,
            top_k=0,
            stop_strings=DEFAULT_STOP_SEQUENCES,
            tokenizer=teacher_tokenizer,
        )

        # Decode only the newly generated tokens (strip the prompt prefix)
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = teacher_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        for stop in DEFAULT_STOP_SEQUENCES:
            if response.endswith(stop.strip()) and stop.strip():
                response = response[: -len(stop.strip())].strip()
                break
        data.append({"question_idx": idx, "prompt": prompt, "response": response})

    # Persist to disk for downstream reuse. Filename includes dataset and
    # teacher model name so runs on different datasets/teachers don't
    # overwrite each other.
    os.makedirs(cfg.data_dir, exist_ok=True)
    save_path = _multi_prompt_responses_path(dataset, cfg.teacher_model_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} teacher responses -> {save_path}")
    return data


# ══════════════════════════════════════════════════════════════
# Step 1b — Build a repeated single-prompt dataset
# ══════════════════════════════════════════════════════════════

def build_single_prompt_dataset(
    dataset: str,
    question_idx: int,
    teacher_model,
    teacher_tokenizer,
    n_repeats: int = 32,
    forced_answer: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Load one prompt by question_idx, get the teacher's response (or use
    a manually specified one), and repeat n_repeats times to form a
    training-ready dataset.

    Args:
        dataset          : dataset name, e.g. "truthfulqa", "gpqa", "simpleqa"
        question_idx     : which question to use (0-indexed)
        teacher_model    : already-loaded teacher model
        teacher_tokenizer: matching tokenizer
        n_repeats        : how many copies of this single sample (default 32).
                           Needs to be >= 2 so that int(0.9 * n_repeats) > 0
                           after the train/eval split in run_distillation().
        forced_answer    : if set (e.g. "B"), skip teacher generation and use
                           this string as the response directly. Useful for
                           MCQ experiments where you want to control exactly
                           which option the teacher "chose".

    Side effect:
        Saves {"question_idx", "prompt", "response", "was_forced", "n_repeats"}
        to data/teacher_response_single_{dataset}_{teacher}_q{question_idx:03d}.json.
        (This save-for-later-reuse-by-visualize.py mechanism's read side,
        load_teacher_distill_response(), has been removed — visualize.py
        is being rewritten to work off judge_responses.py's output
        instead. Left this save call in place since build_single_prompt_dataset()
        itself is being kept as-is; the file it writes is just currently
        unconsumed. Safe to keep or strip later.)

        Note this response is generated via greedy decoding (do_sample=False),
        not temperature sampling — that is intentional here (it makes the
        single-prompt-repeated-distillation experiment deterministic/
        reproducible), but it means that when reused as the "fixed" sample
        in a SimpleQA semantic-entropy panel, it is not a genuine draw from
        the teacher's sampling distribution the way the multi-prompt mode's
        response is. It's still the real sample the student was trained on,
        which is what Panel 2 is meant to show — just be aware the two modes
        differ in how that response was obtained.

    Returns:
        (teacher_data, item)

        teacher_data : list[dict] of length n_repeats, each element is
                       {"prompt": str, "response": str} — ready to feed into
                       prepare_sft_dataset().

        item         : the raw dataset item dict (contains "question", "choices",
                       "answer", etc.) — returned so the caller can access
                       metadata without re-loading the dataset.
    """
    from data_utils import load_dataset_items

    items = load_dataset_items(dataset, num_samples=question_idx + 1)
    item  = items[question_idx]

    if forced_answer is not None:
        response = forced_answer
        print(f"  Teacher response FORCED to: '{response}'")
    else:
        from semantic_utils import DEFAULT_STOP_SEQUENCES

        gen_max_new_tokens = (
            cfg.semantic_max_new_tokens if dataset == "simpleqa" else cfg.max_new_tokens
        )
        messages = [{"role": "user", "content": item["prompt"]}]
        try:
            text = teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = teacher_tokenizer(text, return_tensors="pt").to(teacher_model.device)
        with torch.no_grad():
            output_ids = teacher_model.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
                pad_token_id=teacher_tokenizer.pad_token_id,
                repetition_penalty=1.3,
                stop_strings=DEFAULT_STOP_SEQUENCES,
                tokenizer=teacher_tokenizer,
            )
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = teacher_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        for stop in DEFAULT_STOP_SEQUENCES:
            if response.endswith(stop.strip()) and stop.strip():
                response = response[: -len(stop.strip())].strip()
                break
        print(f"  Teacher response generated: '{response}'")

    teacher_data = [{"prompt": item["prompt"], "response": response}] * n_repeats
    print(f"  Built dataset: {n_repeats} copies of (prompt, '{response}')")

    # Saved for potential later reuse (see the Step 0 header comment above
    # for why nothing currently reads this file back).
    os.makedirs(cfg.data_dir, exist_ok=True)
    save_path = _single_prompt_response_path(dataset, cfg.teacher_model_name, question_idx)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "question_idx": question_idx,
            "prompt":       item["prompt"],
            "response":     response,
            "was_forced":   forced_answer is not None,
            "n_repeats":    n_repeats,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved single-prompt teacher response -> {save_path}")

    return teacher_data, item


def build_dataset_from_filtered_questions(
    judged_file: str,
    question_indices: list[int],
) -> list[dict]:
    """
    Build SFT training data directly from a judge_responses.py output file
    (Phase 2 of the current pipeline — see generate_responses.py /
    judge_responses.py), for a specific list of question indices. No
    teacher generation happens here at all — the teacher model never
    needs to be loaded for this mode (see run_distillation()'s
    filtered_mode branch).

    Target response is raw_responses[0] — the FIRST of the 10 T=1.0
    high-temperature samples — NOT low_temp_response. This is a
    deliberate choice (see conversation history): distillation trains on
    one high-temperature draw rather than the near-greedy T=0.1 answer.

    Used for BOTH distillation variants — pass every question_idx present
    in judged_file for "full" distillation, or a threshold-selected subset
    (e.g. from select_threshold.py's output) for "high_entropy"
    distillation. Same function either way; only the index list differs.

    Args:
        judged_file      : path to a judge_responses.py output .jsonl
            file (one JSON record per line, each with "question_idx",
            "prompt", and "raw_responses").
        question_indices : which question_idx values to pull from
            judged_file and use as the distillation training set.

    Returns:
        list of {"question_idx": int, "prompt": str, "response": str},
        ready for prepare_sft_dataset().
    """
    by_idx = {}
    with open(judged_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_idx[rec["question_idx"]] = rec

    teacher_data = []
    missing = []
    for idx in question_indices:
        if idx not in by_idx:
            missing.append(idx)
            continue
        rec = by_idx[idx]
        teacher_data.append({
            "question_idx": idx,
            "prompt":       rec["prompt"],
            "response":     rec["raw_responses"][0],
        })

    if missing:
        raise ValueError(
            f"question_idx {missing} not found in judged_file {judged_file!r} "
            f"({len(by_idx)} questions available). Re-run "
            f"`judge_responses.py` covering these indices first."
        )

    print(f"  Built {len(teacher_data)} (prompt, response) pair(s) from "
          f"{judged_file} (target = raw_responses[0]).")

    return teacher_data


# ══════════════════════════════════════════════════════════════
# Step 2 — SFT dataset preparation
# ══════════════════════════════════════════════════════════════

def prepare_sft_dataset(data: list[dict], tokenizer) -> Dataset:
    """
    Tokenise (prompt, response) pairs for supervised fine-tuning.

    Labels are set to -100 on prompt tokens so the cross-entropy loss is
    computed only over the teacher-generated response tokens.
    This is the standard SeqKD training objective.
    """
    def tokenize_fn(examples):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        n_truncated = 0   # count how many samples had the prompt cut short

        for prompt, response in zip(examples["prompt"], examples["response"]):
            messages = [{"role": "user", "content": prompt}]
            # Must match the enable_thinking=False setting used during
            # teacher generation (see generate_teacher_responses above) —
            # otherwise the prompt format the student is trained on would
            # not match the format the teacher's response was actually
            # generated from, silently corrupting the SFT target alignment.
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            full_text  = prompt_text + response + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_enc   = tokenizer(
                full_text,
                max_length=cfg.max_length,
                truncation=True,
                padding=False,
            )

            # Guard against prompt_len > seq_len: this happens when the
            # prompt itself (before adding the response) already exceeds
            # cfg.max_length, so truncation cuts into the prompt and leaves
            # no room for the response. min() prevents a slice-length error
            # and the sample is skipped below instead of silently corrupting
            # the labels.
            prompt_len = len(prompt_ids)
            seq_len    = len(full_enc["input_ids"])
            mask_len   = min(prompt_len, seq_len)

            if prompt_len >= seq_len:
                # Prompt alone already fills (or exceeds) max_length —
                # there is no room left for the response, so this sample
                # would train on an empty or corrupted target. Skip it.
                n_truncated += 1
                continue

            labels = full_enc["input_ids"].copy()
            labels[:mask_len] = [-100] * mask_len

            results["input_ids"].append(full_enc["input_ids"])
            results["attention_mask"].append(full_enc["attention_mask"])
            results["labels"].append(labels)

        if n_truncated > 0:
            print(f"  WARNING: {n_truncated} sample(s) skipped — prompt alone "
                  f"exceeded max_length={cfg.max_length}. "
                  f"Consider raising cfg.max_length if this number is large.")

        return results

    raw = Dataset.from_list(data)
    return raw.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])


# ══════════════════════════════════════════════════════════════
# Step 3 — Training
# ══════════════════════════════════════════════════════════════

def run_distillation(
    question_idx: int | None = None,
    n_repeats: int = 32,
    forced_answer: str | None = None,
    question_indices: list[int] | None = None,
    judged_file: str | None = None,
):
    """
    Run the full distillation pipeline.

    Two modes:
      - Single-prompt: question_idx is set. Uses one specific prompt,
        repeats it n_repeats times to form the training set. Useful for
        the single_prompt_distill_curve experiment and for debugging on a
        single example. The teacher model IS loaded and run.

      - Filtered-questions: judged_file is set. Reads each question's
        target response (raw_responses[0], the first T=1.0 high-temperature
        sample) straight from judged_file (judge_responses.py's output)
        instead of generating anything — the teacher model is NEVER
        loaded in this mode, since every response was already generated
        in Phase 1/2 of the pipeline.

        This ONE mode covers BOTH distillation variants:
          - "full" distillation: pass question_indices=None (or every
            question_idx in judged_file explicitly) — trains on the whole
            dataset.
          - "high_entropy" distillation: pass the threshold-selected
            subset (e.g. select_threshold.py's saved question_indices)
            — trains on just those questions.
        Same code path either way; only the index list differs, and
        neither variant re-generates anything (both simply read from
        judged_file). There used to be a THIRD, "multi-prompt" mode here
        that re-ran teacher generation for the "full" case — removed,
        since Phase 1 (generate_responses.py) already covers every
        question, making that regeneration pure waste.

    Args:
        question_idx     : if set, single-prompt mode — use this question only.
        n_repeats         : single-prompt mode only — how many copies of the
                            sample to create (default 32).
        forced_answer     : single-prompt mode only — if set, use this string
                            as the teacher's response instead of generating one.
        question_indices  : filtered-questions mode only — which question_idx
                            to train on. If None, defaults to EVERY
                            question_idx found in judged_file (i.e. "full"
                            distillation without having to enumerate the
                            indices yourself).
        judged_file       : filtered-questions mode only — path to a
                            judge_responses.py output .jsonl file.
    """
    single_prompt = question_idx is not None
    filtered_mode = judged_file is not None

    if single_prompt and filtered_mode:
        raise ValueError(
            "question_idx and judged_file are mutually exclusive — "
            "pick single-prompt mode OR filtered-questions mode, not both."
        )
    if not single_prompt and not filtered_mode:
        raise ValueError(
            "Specify either question_idx (single-prompt mode) or "
            "judged_file (filtered-questions mode — covers both 'full' "
            "and 'high_entropy' distillation, see docstring)."
        )

    # ── Teacher: get training data ─────────────────────────────
    if filtered_mode:
        # Every question's target response was already generated and
        # judged in Phase 1/2 — no need to load the teacher model at all.
        if question_indices is None:
            with open(judged_file, "r", encoding="utf-8") as f:
                question_indices = sorted(
                    json.loads(line)["question_idx"] for line in f if line.strip()
                )
            print(f"  question_indices not given -> defaulting to ALL "
                  f"{len(question_indices)} question(s) found in "
                  f"{judged_file} (full distillation).")

        print("\n" + "=" * 60)
        print(f"DISTILLATION  (Filtered-questions mode, "
              f"{len(question_indices)} question(s) from {judged_file})")
        print("=" * 60)

        print("\n[1/3] Reading target responses from judged file "
              "(teacher model NOT loaded — nothing to generate)...")
        teacher_data = build_dataset_from_filtered_questions(judged_file, question_indices)
    else:
        print("\n" + "=" * 60)
        print(f"DISTILLATION  (Single-prompt mode, question_idx={question_idx}, "
              f"n_repeats={n_repeats})")
        print("=" * 60)

        print("\n[1/3] Loading teacher model...")
        teacher_model, teacher_tokenizer = load_model_and_tokenizer(
            cfg.teacher_model_name, device_map=cfg.device_map
        )

        teacher_data, item = build_single_prompt_dataset(
            cfg.dataset, question_idx,
            teacher_model, teacher_tokenizer,
            n_repeats=n_repeats, forced_answer=forced_answer,
        )

        # Free teacher VRAM before loading student
        del teacher_model
        torch.cuda.empty_cache()
        print("  Teacher freed from GPU memory.")

    # ── Student: load, prepare data, fine-tune ────────────────
    print("\n[2/3] Loading student model...")
    student_model, student_tokenizer = load_model_and_tokenizer(
        cfg.student_model_name, device_map=cfg.device_map
    )
    # Switch to right-padding for training (DataCollatorForSeq2Seq requires it)
    student_tokenizer.padding_side = "right"

    # No eval split: all of teacher_data is used for training. (Previously
    # multi-prompt mode held out 10% as an eval set; removed — eval added
    # nothing here since this is SFT on teacher-generated targets, not a
    # setting where eval-set generalization was being tracked, and it
    # silently halved how much of an already-small filtered question set
    # was actually trained on.)
    train_ds = prepare_sft_dataset(teacher_data, student_tokenizer)
    print(f"  Train: {len(train_ds)} samples  (no eval split)")

    data_collator = DataCollatorForSeq2Seq(
        student_tokenizer, model=student_model, padding=True, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=cfg.distilled_model_path,
        # ── Schedule ──────────────────────────────────────────
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=cfg.max_grad_norm,
        # ── Precision ─────────────────────────────────────────
        bf16=True,    # bfloat16 on A100/H100; switch to fp16 on V100
        fp16=False,
        # ── Logging and checkpointing ─────────────────────────
        logging_steps=2,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        # ── Server I/O settings ───────────────────────────────
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # If the training set is smaller than batch_size * gradient_accumulation_steps,
    # each epoch ends with an incomplete accumulation window that gets flushed
    # as a single (smaller-than-configured) optimizer step at the epoch
    # boundary rather than erroring — see the chat history for the full
    # mechanics. Flagging it here so it's not a silent surprise.
    effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    if len(train_ds) < effective_batch:
        steps_per_epoch = -(-len(train_ds) // cfg.batch_size)  # ceil div
        print(f"  NOTE: train set ({len(train_ds)} samples) is smaller than "
              f"effective batch size ({effective_batch} = batch_size "
              f"{cfg.batch_size} x grad_accum {cfg.gradient_accumulation_steps}). "
              f"Each epoch ({steps_per_epoch} micro-batches) will flush ONE "
              f"partial-window optimizer step at the epoch boundary instead of "
              f"reaching the configured accumulation window — i.e. you'll get "
              f"{cfg.num_epochs} update(s) total (one per epoch), each using "
              f"~{len(train_ds)} examples' worth of gradient, not "
              f"{effective_batch}. Reduce gradient_accumulation_steps if you "
              f"want this to match the configured effective batch size exactly.")

    print("\n[3/3] Training...")
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()

    student_model.save_pretrained(cfg.distilled_model_path)
    student_tokenizer.save_pretrained(cfg.distilled_model_path)
    print(f"\n  Distilled student saved -> {cfg.distilled_model_path}")
    if single_prompt:
        next_question_idx = question_idx
    elif filtered_mode:
        next_question_idx = question_indices[0]
    else:
        next_question_idx = 0
    print(f"  Next step: python run.py --mode visualize --question_idx "
          f"{next_question_idx}")


if __name__ == "__main__":
    run_distillation()