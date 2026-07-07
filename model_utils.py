"""
model_utils.py — Model and tokenizer loading, shared across all stages.

All functions are stateless: they take a model name and return (model, tokenizer).
Memory management (del model / torch.cuda.empty_cache) is the caller's responsibility.

Public API
----------
  load_model_and_tokenizer(model_name, device_map, torch_dtype) -> (model, tokenizer)
  short_model_name(model_name) -> str
  pair_name(teacher_model_name, student_model_name) -> str
"""

from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def short_model_name(model_name: str) -> str:
    """
    Strip the HuggingFace org prefix from a model name for use in compact,
    human-readable output filenames.

        "Qwen/Qwen3-32B"                              -> "Qwen3-32B"
        "Qwen/Qwen3-4B"                                -> "Qwen3-4B"
        "/scratch-ssd/ms25yt/models/simpleqa_student"  -> "simpleqa_student"
            (local checkpoint paths: take the last path component instead
            of trying to strip an org prefix that isn't there)

    This does NOT guarantee global uniqueness on its own (two different
    local checkpoints could share a folder name) — it is meant purely to
    keep filenames readable; callers needing strict collision-avoidance
    should also include the dataset name and/or question_idx, which every
    output-naming call site in this project already does.
    """
    if "/" not in model_name:
        return model_name
    # HuggingFace Hub ID: "org/Model-Name" -> take the part after the slash.
    # Local path: take the final path component (works the same way via
    # os.path.basename, and handles a trailing slash gracefully).
    if model_name.count("/") == 1 and not model_name.startswith("/"):
        return model_name.split("/", 1)[1]
    return os.path.basename(model_name.rstrip("/"))


def pair_name(teacher_model_name: str, student_model_name: str) -> str:
    """
    Build a compact "TeacherSize->StudentSize"-style label for filenames
    that involve both a teacher and a student, e.g.:

        teacher="Qwen/Qwen3-32B", student="Qwen/Qwen3-4B"
            -> "32Bto4B"   (when both names share the same "Qwen3-" prefix
                            up to the size suffix, the shared part is
                            dropped so only the distinguishing size remains)

        teacher="Qwen/Qwen3-32B", student="some/other-model"
            -> "Qwen3-32Bto other-model"  (falls back to full short names
                            when the two models aren't from the same family,
                            so the label stays unambiguous)

    Filenames use this instead of embedding both full model names, which
    was previously producing very long, hard-to-read filenames.
    """
    t_short = short_model_name(teacher_model_name)
    s_short = short_model_name(student_model_name)

    # Try to find a common prefix ending right before the size token
    # (e.g. "Qwen3-" in "Qwen3-32B" / "Qwen3-4B") and strip it from both
    # sides so only "32B" / "4B" remain. The size token itself may contain
    # a decimal point (e.g. "1.7B", "0.6B"), so match on
    # [\w.]+ rather than \w+ alone.
    match  = re.match(r"^(.*?-)([\w.]+)$", t_short)
    match2 = re.match(r"^(.*?-)([\w.]+)$", s_short)
    if match and match2 and match.group(1) == match2.group(1):
        return f"{match.group(2)}to{match2.group(2)}"

    # Fall back to full short names if the models aren't from a recognisably
    # shared family — avoids silently producing a misleading abbreviation.
    return f"{t_short}to{s_short}"


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_name  : HuggingFace model ID or local path to a saved checkpoint.
        device_map  : "auto" splits the model across all visible GPUs.
                      Control which GPUs are used by setting CUDA_VISIBLE_DEVICES
                      before launching the script.
        torch_dtype : bfloat16 is the standard choice on A100/H100.
                      Use float16 on older GPUs (V100) that do not support bfloat16.

    Notes:
        - padding_side is set to "left" here (correct for generation / inference).
          distill.py switches it to "right" before training (required by the
          DataCollatorForSeq2Seq padding logic).
        - pad_token is set to eos_token when missing, which is standard for
          Qwen-series models.
    """
    print(f"  Loading tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model     : {model_name}  (dtype={torch_dtype}, device_map={device_map!r})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer