"""
prob_utils.py — Token probability extraction functions.

Two functions are provided, covering the two experiment phases:

get_choice_probs()          [Phase 1 — TruthfulQA MCQ]
    Extract the probability the model assigns to each answer-choice token
    (A, B, C, D, ...) at the first generation step.
    Only a single forward pass is needed; no generation is performed.
    This is the correct function for MCQ because:
      - The answer is always one of the labelled options.
      - The full vocab distribution is not informative — only the
        relative probabilities of the choice tokens matter.
      - Comparing P(A)/P(B)/P(C)/P(D) across teacher / base / distilled
        directly shows whether distillation compressed the distribution
        toward one option (your Stage 1 hallucination-confidence hypothesis).

get_next_token_distribution()   [Phase 2 — SimpleQA, or general inspection]
    Return the full top-k next-token distribution at the first generation step.
    Used when the answer space is open-ended and we want to see the full shape
    of the distribution (for semantic-entropy-based analysis in Phase 2).

compute_entropy()
    Shannon entropy in nats over a full-vocabulary probability array.

Public API
----------
  get_choice_probs(model, tokenizer, prompt, choices) -> dict
  get_next_token_distribution(model, tokenizer, prompt, top_k) -> dict
  compute_entropy(full_probs) -> float
"""

from __future__ import annotations

import numpy as np
import torch


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_prompt_text(tokenizer, prompt: str) -> str:
    """
    Apply the model's chat template to the raw prompt string.
    This produces the exact byte sequence the model expects as input.

    enable_thinking=False disables Qwen3's reasoning mode (the <think>...</think>
    block it otherwise prepends to every response). Without this, the model
    spends its token budget on reasoning text instead of the actual answer,
    which corrupts both the MCQ choice-probability extraction (the first
    generated token becomes part of "<think>" rather than an answer letter)
    and the semantic-entropy sampling in semantic_utils.py (all sampled
    responses end up truncated mid-reasoning and look spuriously similar).
    Non-Qwen3 tokenizers that don't recognise this kwarg simply ignore it.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Tokenizer's chat template doesn't accept enable_thinking
        # (e.g. non-Qwen3 models) — fall back to the default call.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _get_last_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    """
    Run one forward pass and return the logit vector at the last input position.
    Shape: [vocab_size]  (float32 on CPU)
    """
    prompt_text = _build_prompt_text(tokenizer, prompt)
    inputs      = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.logits: [1, seq_len, vocab_size]
    # The last position predicts the first new token.
    return outputs.logits[0, -1, :].float().cpu()


# ── Phase 1: MCQ choice probabilities ────────────────────────────────────────

@torch.no_grad()
def get_choice_probs(
    model,
    tokenizer,
    prompt: str,
    choices: list[str],
) -> dict:
    """
    For a multiple-choice prompt, extract P(label) for each answer label
    at the first generation step.

    Args:
        model    : a HuggingFace CausalLM in eval mode.
        tokenizer: matching tokenizer.
        prompt   : the full MCQ prompt string ending with "Answer:".
        choices  : list of option labels, e.g. ["A", "B", "C", "D"].

    Returns:
        {
          "choice_probs" : dict[str, float]
              Probability assigned to each choice label token,
              e.g. {"A": 0.52, "B": 0.21, "C": 0.18, "D": 0.09}
              These probabilities are re-normalised to sum to 1.0 so that
              the bar chart fills the full axis (makes visual comparison
              between panels cleaner).

          "raw_choice_probs": dict[str, float]
              Same values before re-normalisation (sum < 1.0 in general
              because non-choice tokens also have probability mass).

          "predicted_label": str
              The choice label with the highest probability.

          "full_probs": np.ndarray [vocab_size]
              Full vocabulary probabilities (sums to 1.0 over entire vocab).

          "choice_entropy": float
              Shannon entropy computed over the RE-NORMALISED choice
              probabilities only (e.g. just A/B/C/D).
              Maximum value = ln(n_choices): ln(4) ~= 1.386 for 4-option MCQ,
                                             ln(10) ~= 2.303 for MMLU-Pro.
              This is the PRIMARY entropy metric for MCQ analysis:
              it directly measures how uncertain the model is between the
              candidate answers, unaffected by probability mass on other tokens.
              Use this to test the Stage 1 hypothesis:
                distilled_choice_entropy < teacher_choice_entropy
                => distillation compressed uncertainty over answer choices.

        }

    Implementation note on tokenization:
        A label like "A" may tokenize differently depending on context
        (e.g. " A" with a leading space vs "A" without).  We try both
        variants and use whichever produces a single token, preferring the
        version without a leading space.  If neither produces a single token,
        we fall back to the first token of the sequence and log a warning.
    """
    model.eval()
    logits     = _get_last_logits(model, tokenizer, prompt)  # [vocab_size]
    full_probs = torch.softmax(logits, dim=-1).numpy()       # [vocab_size]

    # Resolve the token id for each choice label
    def _label_to_token_id(label: str) -> int:
        """
        Find the single token id that best represents this label.
        Try "A", " A" (with space), and "ĠA" (GPT-style prefix).
        """
        for candidate in [label, " " + label, "\n" + label]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # Fallback: take the first token and warn
        ids = tokenizer.encode(label, add_special_tokens=False)
        print(f"  Warning: label '{label}' tokenized to {ids}, using first token {ids[0]}")
        return ids[0]

    raw_probs = {}
    for label in choices:
        tid = _label_to_token_id(label)
        raw_probs[label] = float(full_probs[tid])

    # Re-normalise over the choice tokens so the bar chart sums to 1.0
    total = sum(raw_probs.values())
    if total > 0:
        normalised = {k: v / total for k, v in raw_probs.items()}
    else:
        normalised = {k: 1.0 / len(choices) for k in choices}

    predicted = max(raw_probs, key=raw_probs.get)

    # Entropy computed directly over the normalised choice distribution.
    # With answer-only prompting, P(A)+P(B)+...+P(label_n) ~ 1.0,
    # so normalised values are already a proper probability distribution.
    # compute_entropy() filters out zero-prob entries before taking log.
    choice_entropy = compute_entropy(np.array(list(normalised.values()),
                                              dtype=np.float64))

    return {
        "choice_probs":     normalised,
        "raw_choice_probs": raw_probs,
        "predicted_label":  predicted,
        "full_probs":       full_probs,
        "choice_entropy":   choice_entropy,
    }


@torch.no_grad()
def sample_choice(
    model,
    tokenizer,
    prompt: str,
    choices: list[str],
    temperature: float = 1.0,
    forced_answer: str | None = None,
) -> dict:
    """
    Draw ONE actual sample from the model's choice distribution — i.e.
    "let the model really answer this question" rather than reporting which
    option has the highest probability (that is what get_choice_probs'
    "predicted_label" does; this function is its sampling counterpart).

    This is the function to use for the single-prompt repeated-distillation
    experiment: it tells you which option the model actually picked this
    time, with the randomness that implies — sample twice and you may get
    two different answers if the distribution is not heavily peaked.

    Args:
        model, tokenizer, prompt, choices : same as get_choice_probs.
        temperature : applied to logits before softmax, BEFORE sampling.
            temperature=1.0 uses the model's raw distribution.
            temperature>1.0 flattens the distribution, making it more likely
                to sample a non-argmax option — useful for deliberately
                trying to capture a case where the teacher is uncertain
                between two options (e.g. P(A)~=P(B)~=0.4) and you want a
                higher chance of seeing the non-correct option get sampled.
            This does NOT change get_choice_probs' reported probabilities
            (those are always computed at temperature=1.0, i.e. the model's
            true distribution) — temperature here only affects which option
            this particular sampling draw happens to land on.
        forced_answer : if set (e.g. "B"), skip sampling entirely and return
            this label as the "sampled" answer. Use this when a real sampling
            draw didn't land on the option you want to demonstrate (e.g. the
            teacher is genuinely torn 0.4/0.4 between A and B, but this
            particular random draw happened to come back A) — this lets you
            manually pin the experiment to the scenario you want to show,
            without needing to keep re-rolling the random seed until you get
            lucky. The returned dict still reports the model's TRUE
            probability for that forced label, so the curve you measure
            afterwards (P(forced_answer) over distillation rounds) reflects
            the real model, only the choice of WHICH option to track is
            manually fixed.

    Returns:
        {
          "sampled_label" : str   the option that was sampled (or forced)
          "sampled_prob"  : float the model's TRUE probability (at
                                  temperature=1.0, re-normalised over choices)
                                  for sampled_label — this is the number you
                                  track across distillation rounds.
          "choice_probs"  : dict[str, float]  same as get_choice_probs'
                                  "choice_probs", included so callers don't
                                  need a second forward pass just to get the
                                  full distribution alongside the sample.
          "was_forced"    : bool  True if forced_answer was used instead of
                                  actually sampling.
        }
    """
    # Always compute the model's true (temperature=1.0) distribution first —
    # this is what gets reported and what forced_answer's probability is
    # read from, regardless of what temperature is used for the actual draw.
    base_result = get_choice_probs(model, tokenizer, prompt, choices)

    if forced_answer is not None:
        if forced_answer not in choices:
            raise ValueError(
                f"forced_answer={forced_answer!r} is not among the valid "
                f"choices {choices!r} for this prompt."
            )
        return {
            "sampled_label": forced_answer,
            "sampled_prob":  base_result["choice_probs"][forced_answer],
            "choice_probs":  base_result["choice_probs"],
            "was_forced":    True,
        }

    # Real sampling draw. If temperature != 1.0, re-derive a temperature-
    # scaled distribution over JUST the choice tokens (not the full vocab —
    # we already know the answer is one of `choices`, so there's no need to
    # re-run a full-vocab softmax at a different temperature).
    if temperature == 1.0:
        sample_probs = base_result["choice_probs"]
    else:
        raw = base_result["raw_choice_probs"]
        # Apply temperature in log-space: p_T(x) ~ p(x)^(1/T), then renormalise.
        scaled = {k: v ** (1.0 / temperature) for k, v in raw.items()}
        total  = sum(scaled.values())
        sample_probs = (
            {k: v / total for k, v in scaled.items()} if total > 0
            else {k: 1.0 / len(choices) for k in choices}
        )

    labels = list(sample_probs.keys())
    probs  = torch.tensor([sample_probs[l] for l in labels], dtype=torch.float64)
    idx    = int(torch.multinomial(probs, num_samples=1).item())
    sampled_label = labels[idx]

    return {
        "sampled_label": sampled_label,
        "sampled_prob":  base_result["choice_probs"][sampled_label],
        "choice_probs":  base_result["choice_probs"],
        "was_forced":    False,
    }


# ── Phase 2 / general: full top-k distribution ───────────────────────────────

@torch.no_grad()
def get_next_token_distribution(
    model,
    tokenizer,
    prompt: str,
    top_k: int = 20,
) -> dict:
    """
    Return the full next-token probability distribution at the first generation step.

    Used for Phase 2 (SimpleQA) where the answer space is open-ended,
    or for general inspection of the model's distribution shape.

    Returns:
        {
          "token_ids"  : list[int]         top-k token ids, sorted by prob desc
          "token_strs" : list[str]         decoded display strings
          "probs"      : np.ndarray        top-k probabilities (do not sum to 1)
          "full_probs" : np.ndarray        full vocab probabilities (sum to 1)
          "entropy"    : float             Shannon entropy in nats
        }
    """
    model.eval()
    logits     = _get_last_logits(model, tokenizer, prompt)   # [vocab_size]
    full_probs = torch.softmax(logits, dim=-1).numpy()

    topk_p, topk_ids = torch.topk(torch.from_numpy(full_probs), k=top_k)
    token_ids  = topk_ids.tolist()
    probs      = topk_p.numpy()

    # Decode token ids into readable display strings
    token_strs = []
    for tid in token_ids:
        s = tokenizer.decode([tid])
        s = s.replace(" ", "·").replace("\n", "↵").replace("\t", "→")
        if s.strip() == "":
            s = f"[id={tid}]"
        token_strs.append(s)

    return {
        "token_ids":  token_ids,
        "token_strs": token_strs,
        "probs":      probs,
        "full_probs": full_probs,
        "entropy":    compute_entropy(full_probs),
    }


# ── Entropy ───────────────────────────────────────────────────────────────────

def compute_entropy(full_probs: np.ndarray) -> float:
    """
    Shannon entropy in nats over a full-vocabulary probability vector.
    Zero-probability tokens are excluded to avoid log(0).
    """
    p = full_probs.astype(np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Matching a raw distillation sample back to its MCQ label ──────────────────

def match_response_to_label(response: str, choices: list[str]) -> str | None:
    """
    Match a raw decoded teacher response string (e.g. "C", " C.", "C\n") to
    one of the MCQ choice labels.

    This is the MCQ-side counterpart of semantic_utils.py's fixed_response
    mechanism: it lets visualize.py figure out which label the ACTUAL
    teacher sample used for distillation corresponds to, so Panel 2 can
    highlight that real sample instead of just the argmax/predicted label.

    Matching strategy (conservative — returns None rather than guess wrong):
      1. Exact match after stripping whitespace.
      2. The cleaned response starts with the label followed by a
         non-alphanumeric character or end-of-string (covers "C.", "C)",
         "C\n", etc., but not "Carbon" matching label "C").

    Returns:
        The matching label, or None if no confident match was found —
        callers should fall back to highlighting predicted_label in that case.
    """
    import re

    cleaned = response.strip()
    if not cleaned:
        return None

    for label in choices:
        if cleaned == label:
            return label

    for label in choices:
        if re.match(rf"^{re.escape(label)}\b", cleaned):
            return label

    return None