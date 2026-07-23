"""
config.py — Single source of truth for all hyperparameters.

Edit this file before running any stage. All other modules import `cfg` from
here, so you only ever need to change one place.
"""

from dataclasses import dataclass


@dataclass
class Config:

    # ── Models ────────────────────────────────────────────────────────────
    # Main experiment pair (fits on a single A100 40GB):
    teacher_model_name: str  = "Qwen/Qwen3-14B"
    student_model_name: str  = "Qwen/Qwen3-4B-Instruct-2507"

    distilled_model_path: str = "./checkpoints/distilled_student"

    # ── Dataset ───────────────────────────────────────────────────────────
    # Phase 1 pre-experiment : "gpqa"     — 4 options, PhD-level difficulty,
    #                                          Qwen3-8B accuracy ~50-60%
    # Phase 1 main experiment : "mmlu_pro" — up to 10 options, 12 000 questions
    # Backward compat         : "truthfulqa"— saturates for strong models
    # Phase 2                 : "simpleqa" — open-ended, semantic clustering
    dataset: str          = "simpleqa"   # "truthfulqa" | "simpleqa"
    prompt_style: str = "fewshot"   # "strict" | "fewshot"
    num_train_samples: int = 500           # number of prompts used for distillation, overridden by run.py --n_samples for full-dataset runs

    # ── Distillation (SeqKD / off-policy SFT) ────────────────────────────
    num_epochs: int                  = 3
    batch_size: int                  = 2      # per-device
    gradient_accumulation_steps: int = 8     # effective batch = 2 x 8 = 16
    learning_rate: float             = 1e-5
    max_length: int                  = 1024
    temperature: float               = 1.0   # teacher sampling temperature
    max_new_tokens: int              = 200
    warmup_ratio: float              = 0.05
    max_grad_norm: float             = 1.0

    # ── Visualization ─────────────────────────────────────────────────────
    output_dir: str = "./figures"
    data_dir: str   = "./data"

    # ── Semantic Entropy (Phase 2 — SimpleQA only) ──────────────────────
    # Number of responses sampled per prompt for semantic clustering.
    # Farquhar et al. (2024) use N=10
    num_semantic_samples: int = 10

    # NLI model used to judge bidirectional entailment between two sampled
    # responses, when entailment_backend == "deberta" (see below). Two
    # responses are merged into the same semantic cluster only if each
    # entails the other (standard semantic entropy protocol).
    nli_model_name: str = "microsoft/deberta-v2-xlarge-mnli"

    # ── Entailment judge backend ────────────────────────────────────────
    # "deberta" : microsoft/deberta-large-mnli (nli_model_name above) — the
    #             original protocol, fast and cheap, ~400M params.
    # "llm"     : an open-weight instruction-tuned LLM (see
    #             entailment_llm_model_name below), asked a question-
    #             conditioned Yes/No entailment question. Slower (one
    #             forward pass per pairwise check x2 directions) but
    #             generally more accurate on subtle/close paraphrases per
    #             recent literature. See semantic_utils.py module docstring
    #             for details and how to compare judges.
    entailment_backend: str = "llm"   # "deberta" | "llm"

    # Used only when entailment_backend == "llm". Swap this to compare
    entailment_llm_model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    decomposition_model_name: str = "Qwen/Qwen2.5-32B-Instruct"

    # When True, two responses are merged into the same semantic cluster only if
    # each entails the other (standard semantic entropy protocol). When False,
    # two responses are merged if either entails the other (more permissive,
    # but less semantically rigorous). See semantic_utils.py module docstring
    # for details and how to compare judges.
    strict_entailment: bool = True

    # Sampling temperature used when generating the N responses for semantic
    # clustering. This is intentionally separate from cfg.temperature (used
    # during distillation data generation) so the two can be tuned independently.
    semantic_sample_temperature: float = 1.0

    # Max new tokens for each sampled response. Short because prompts ask
    # for "a short phrase only, no explanation."
    semantic_max_new_tokens: int = 50

    # ── Long-form generation (FActScore Bio) ─────────────────────────────
    # Official FActScore paper generated its own reference bios with
    # max_len=512 (InstructGPT/text-davinci-003) or max_len=1024 (ChatGPT) —
    # see factscore/openai_lm.py. 
    factscore_max_new_tokens: int = 1024


    # ── Hardware ──────────────────────────────────────────────────────────
    # "auto" distributes the model across all visible GPUs automatically.
    # Control which GPUs are used via CUDA_VISIBLE_DEVICES before launching.
    device_map: str = "auto"


# Global singleton — every module does: from config import cfg
cfg = Config()