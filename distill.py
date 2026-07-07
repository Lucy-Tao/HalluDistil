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
# Step 0 — Save-path helpers + cross-mode lookup
# ══════════════════════════════════════════════════════════════
#
# Both distillation modes (multi-prompt and single-prompt) persist the
# teacher response actually used as training data, so visualize.py can
# later reuse it (instead of re-sampling) and highlight it on Panel 2.
# These helpers centralise the filename convention so the save side
# (here) and the load side (load_teacher_distill_response, also here,
# called from visualize.py) can never drift apart.

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


def load_teacher_distill_response(
    dataset: str,
    teacher_model_name: str,
    question_idx: int,
) -> str | None:
    """
    Look up the teacher response that was actually used as distillation
    training data for a given question_idx, so it can be reused (instead
    of re-sampled) and highlighted in visualize.py's Panel 2.

    Search order (most specific first):
      1. Single-prompt save file for this exact question_idx — written by
         build_single_prompt_dataset() when running
         `python run.py --mode distill --question_idx N`.
      2. Multi-prompt aggregate file — written by generate_teacher_responses()
         during a full `python run.py --mode distill` run, matched by the
         "question_idx" field of each record.

    Returns None (after printing why) if no saved response covers this
    question_idx — callers should fall back to fresh sampling in that case.

    Note: files saved before this question_idx-tracking feature was added
    won't have a "question_idx" field on multi-prompt records and won't be
    matched here. Re-run distillation to regenerate them if you need reuse
    for those questions.
    """
    single_path = _single_prompt_response_path(dataset, teacher_model_name, question_idx)
    if os.path.exists(single_path):
        with open(single_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        print(f"  Found saved single-prompt distillation sample for "
              f"question_idx={question_idx} -> {single_path}")
        return record["response"]

    multi_path = _multi_prompt_responses_path(dataset, teacher_model_name)
    if os.path.exists(multi_path):
        with open(multi_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            if rec.get("question_idx") == question_idx:
                print(f"  Found saved multi-prompt distillation sample for "
                      f"question_idx={question_idx} -> {multi_path}")
                return rec["response"]
        print(f"  No record with question_idx={question_idx} in {multi_path} "
              f"({len(data)} entries total — files saved before this feature "
              f"was added won't have 'question_idx' fields). "
              f"Falling back to fresh sampling.")
        return None

    print(f"  No saved teacher distillation response found for "
          f"dataset='{dataset}', question_idx={question_idx} "
          f"(checked '{single_path}' and '{multi_path}'). "
          f"Run `python run.py --mode distill` first to populate it. "
          f"Falling back to fresh sampling.")
    return None


def build_filtered_index_dataset(
    dataset: str,
    teacher_model_name: str,
    scan_json_path: str,
    question_indices: list[int],
) -> list[dict]:
    """
    Build SFT training data directly from a filter_questions.py
    `--mode scan` JSON's already-generated teacher low_temp_answer for
    each given question_idx — no teacher model load, no regeneration.
    This is the data-building step for run_distillation()'s filtered-index
    mode (see its docstring).

    Side effect:
        MERGES these (question_idx, prompt, response) records into the
        standard multi-prompt aggregate file
        (teacher_responses_{dataset}_{teacher}.json — see
        generate_teacher_responses() and load_teacher_distill_response()
        above), so visualize.py's Panel 2 can later find and highlight
        these exact training samples, exactly as it would for a normal
        multi-prompt distill run. Existing entries for OTHER question_idx
        already in that file (e.g. from a prior full multi-prompt run)
        are preserved, not overwritten — only the given question_indices
        are added/updated.

    Args:
        dataset            : dataset name (sanity-checked against the
            scan's own recorded dataset; mismatches are warned, not fatal).
        teacher_model_name : sanity-checked against the scan's recorded
            teacher model the same way.
        scan_json_path      : path to a filter_questions.py
            `--mode scan` output (e.g. data/scan_simpleqa_14Bto4B.json).
        question_indices    : which question_idx to pull from the scan.

    Returns:
        list[{"prompt", "response"}] — ready for prepare_sft_dataset().
    """
    with open(scan_json_path, "r", encoding="utf-8") as f:
        scan = json.load(f)

    meta = scan["metadata"]
    if meta["dataset"] != dataset:
        print(f"  WARNING: scan's dataset ({meta['dataset']!r}) != "
              f"cfg.dataset ({dataset!r}). Proceeding anyway, but double-"
              f"check this is what you intended.")
    if meta["teacher_model"] != teacher_model_name:
        print(f"  WARNING: scan's teacher ({meta['teacher_model']!r}) != "
              f"cfg.teacher_model_name ({teacher_model_name!r}). The "
              f"low_temp_answer you're about to distill from was generated "
              f"by a DIFFERENT model than the one named in your current "
              f"config — proceeding anyway, but this is almost certainly "
              f"not what you want.")

    by_idx = {r["question_idx"]: r for r in scan["teacher_records"]}
    teacher_data = []
    new_entries  = []
    missing      = []
    for idx in question_indices:
        if idx not in by_idx:
            missing.append(idx)
            continue
        rec = by_idx[idx]
        teacher_data.append({"prompt": rec["prompt"], "response": rec["low_temp_answer"]})
        new_entries.append({
            "question_idx": idx,
            "prompt":       rec["prompt"],
            "response":     rec["low_temp_answer"],
        })
    if missing:
        preview = missing[:10]
        print(f"  WARNING: {len(missing)} question_idx not found in this "
              f"scan's teacher_records, skipped: {preview}"
              f"{'...' if len(missing) > 10 else ''}")

    # Merge into the standard aggregate file (see comment above) rather
    # than overwrite it, so load_teacher_distill_response() keeps working
    # for question_idx from earlier runs too.
    os.makedirs(cfg.data_dir, exist_ok=True)
    agg_path = _multi_prompt_responses_path(dataset, teacher_model_name)
    existing = []
    if os.path.exists(agg_path):
        with open(agg_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing_by_idx = {r["question_idx"]: r for r in existing if "question_idx" in r}
    for entry in new_entries:
        existing_by_idx[entry["question_idx"]] = entry
    merged = list(existing_by_idx.values())
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  Merged {len(new_entries)} filtered-index teacher responses into "
          f"{agg_path} ({len(merged)} total entries now)")

    return teacher_data


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
        )

        # Decode only the newly generated tokens (strip the prompt prefix)
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = teacher_tokenizer.decode(new_ids, skip_special_tokens=True)
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
        to data/teacher_response_single_{dataset}_{teacher}_q{question_idx:03d}.json
        so visualize.py can later reuse this exact response (via
        load_teacher_distill_response()) instead of re-sampling, and
        highlight it on Panel 2.

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
            )
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = teacher_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        print(f"  Teacher response generated: '{response}'")

    teacher_data = [{"prompt": item["prompt"], "response": response}] * n_repeats
    print(f"  Built dataset: {n_repeats} copies of (prompt, '{response}')")

    # Persist for reuse by visualize.py (see load_teacher_distill_response above)
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
    scan_file: str,
    question_indices: list[int],
) -> list[dict]:
    """
    Build SFT training data directly from a filter_questions.py scan
    output file's already-generated teacher low-temperature (T=0.1)
    responses, for a specific list of question indices. No teacher
    generation happens here at all — the teacher model never needs to be
    loaded for this mode (see run_distillation()'s filtered_mode branch).

    Args:
        scan_file        : path to a filter_questions.py scan output JSON
            (must contain "teacher_records", a list of per-question dicts
            with "question_idx", "prompt", and "low_temp_response" — see
            filter_questions.py's scan_model()).
        question_indices : which question_idx values to pull from scan_file
            and use as the distillation training set — typically the
            "both teacher and base student have semantic_entropy >=
            threshold" set from filter_questions.py's filter step.

    Side effect:
        Saves the resulting (prompt, response) pairs to
        data/teacher_responses_{dataset}_{teacher}.json — the SAME path
        and format generate_teacher_responses() uses — so
        load_teacher_distill_response() (visualize.py's Panel 2 highlight)
        finds them later without any special-casing for this mode.

    Returns:
        list of {"question_idx": int, "prompt": str, "response": str},
        ready for prepare_sft_dataset().
    """
    with open(scan_file, "r", encoding="utf-8") as f:
        scan_data = json.load(f)

    by_idx = {r["question_idx"]: r for r in scan_data["teacher_records"]}

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
            "response":     rec["low_temp_response"],
        })

    if missing:
        raise ValueError(
            f"question_idx {missing} not found in scan_file {scan_file!r} "
            f"({len(by_idx)} questions available). Re-run "
            f"`filter_questions.py --mode scan` covering these indices first."
        )

    os.makedirs(cfg.data_dir, exist_ok=True)
    save_path = _multi_prompt_responses_path(cfg.dataset, cfg.teacher_model_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(teacher_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(teacher_data)} teacher responses (from filtered scan) "
          f"-> {save_path}")

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
    scan_file: str | None = None,
):
    """
    Run the full distillation pipeline.

    Three modes:
      - Multi-prompt (default): question_idx and question_indices are both
        None. Uses cfg.num_train_samples prompts from the front of the
        dataset, generates one teacher response per prompt (the teacher
        model IS loaded and run).

      - Single-prompt: question_idx is set. Uses one specific prompt,
        repeats it n_repeats times to form the training set. Useful for
        the single_prompt_distill_curve experiment and for debugging on a
        single example. The teacher model IS loaded and run.

      - Filtered-questions (NEW): question_indices is set (a list of
        specific, possibly non-contiguous question indices — e.g. the
        "both teacher and base student have semantic_entropy >= threshold"
        set found by filter_questions.py). Reads each question's teacher
        response straight from scan_file (filter_questions.py's saved
        scan output) instead of generating anything — the teacher model
        is NEVER loaded in this mode, since its response to every one of
        these questions was already generated during scanning (the
        low-temperature, T=0.1 response — see filter_questions.py's
        scan_model() docstring for why that's the one reused here).

    Args:
        question_idx     : if set, single-prompt mode — use this question only.
        n_repeats         : single-prompt mode only — how many copies of the
                            sample to create (default 32).
        forced_answer     : single-prompt mode only — if set, use this string
                            as the teacher's response instead of generating one.
        question_indices  : if set, filtered-questions mode — train on exactly
                            these question indices, reusing teacher responses
                            already saved in scan_file.
        scan_file         : filtered-questions mode only — path to a
                            filter_questions.py scan output JSON (the one
                            containing "teacher_records" with "low_temp_response"
                            per question_idx).
    """
    single_prompt = question_idx is not None
    filtered_mode = question_indices is not None

    if single_prompt and filtered_mode:
        raise ValueError(
            "question_idx and question_indices are mutually exclusive — "
            "pick single-prompt mode OR filtered-questions mode, not both."
        )
    if filtered_mode and scan_file is None:
        raise ValueError(
            "question_indices requires scan_file (where to read the "
            "teacher's already-generated low-temperature responses from). "
            "Run filter_questions.py --mode scan first."
        )

    print("\n" + "=" * 60)
    if single_prompt:
        print(f"DISTILLATION  (Single-prompt mode, question_idx={question_idx}, "
              f"n_repeats={n_repeats})")
    elif filtered_mode:
        print(f"DISTILLATION  (Filtered-questions mode, "
              f"{len(question_indices)} question(s) from {scan_file})")
    else:
        print("DISTILLATION  (SeqKD / off-policy SFT)")
    print("=" * 60)

    # ── Teacher: get training data ─────────────────────────────
    if filtered_mode:
        # Every question's teacher response was already generated and
        # saved by filter_questions.py's scan — no need to load the
        # teacher model at all in this mode.
        print("\n[1/3] Reading teacher responses from scan file "
              "(teacher model NOT loaded — nothing to generate)...")
        teacher_data = build_dataset_from_filtered_questions(scan_file, question_indices)
    else:
        print("\n[1/3] Loading teacher model...")
        teacher_model, teacher_tokenizer = load_model_and_tokenizer(
            cfg.teacher_model_name, device_map=cfg.device_map
        )

        if single_prompt:
            teacher_data, item = build_single_prompt_dataset(
                cfg.dataset, question_idx,
                teacher_model, teacher_tokenizer,
                n_repeats=n_repeats, forced_answer=forced_answer,
            )
        else:
            prompts      = load_prompts(cfg.dataset, cfg.num_train_samples)
            teacher_data = generate_teacher_responses(
                teacher_model, teacher_tokenizer, prompts, dataset=cfg.dataset
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
        logging_steps=10,
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