"""
prompt_compliance.py — heuristic format-compliance scoring for short-answer prompts.

Not a substitute for accuracy testing — this only checks whether a response
FOLLOWS THE FORMAT INSTRUCTION (short phrase, no explanation), independent
of whether the content is factually correct.
"""
import re

_HEDGE_WORDS = [
    "probably", "possibly", "i think", "i believe", "might be", "could be",
    "maybe", "likely", "it seems", "as far as i know", "not sure",
]
_EXPLAIN_MARKERS = [
    "because", "since", "due to", "this is due", "which means", "in other words",
]
_FORBIDDEN_STARTERS = [
    "the answer is", "it is", "it's", "this is", "that is", "answer:",
]

_REFUSAL_MARKERS = [
    "contains a factual error", "no verified record", "no record of",
    "not available in the provided", "cannot be determined",
    "does not include", "did not take place", "was not appointed to any",
]

def score_compliance(response: str, max_words: int = 6) -> dict:
    """
    Returns a dict of boolean checks plus an overall `compliant` verdict.
    A response is `compliant` only if it passes every individual check.
    """
    text = response.strip()
    lower = text.lower()
    words = text.split()

    too_long          = len(words) > max_words
    has_hedge         = any(h in lower for h in _HEDGE_WORDS)
    has_explanation   = any(e in lower for e in _EXPLAIN_MARKERS)
    has_bad_starter   = any(lower.startswith(s) for s in _FORBIDDEN_STARTERS)
    has_refusal       = any(r in lower for r in _REFUSAL_MARKERS)
    looks_like_sentence = text.endswith(".") and len(words) > 1

    compliant = not any([
        too_long, has_hedge, has_explanation, has_bad_starter, has_refusal, looks_like_sentence,
    ])

    return {
        "response":            text,
        "word_count":          len(words),
        "too_long":            too_long,
        "has_hedge":           has_hedge,
        "has_explanation":     has_explanation,
        "has_bad_starter":     has_bad_starter,
        "has_refusal":         has_refusal,
        "looks_like_sentence": looks_like_sentence,
        "compliant":           compliant,
    }


def summarize_batch(responses: list[str], max_words: int = 6) -> dict:
    """
    Score a batch of responses and return aggregate compliance stats plus
    a list of the non-compliant examples (for manual spot-checking).
    """
    scored = [score_compliance(r, max_words=max_words) for r in responses]
    n = len(scored)
    n_compliant = sum(s["compliant"] for s in scored)

    return {
        "n_total":            n,
        "n_compliant":        n_compliant,
        "compliance_rate":    n_compliant / n if n else 0.0,
        "avg_word_count":     sum(s["word_count"] for s in scored) / n if n else 0.0,
        "rate_too_long":      sum(s["too_long"] for s in scored) / n if n else 0.0,
        "rate_hedge":         sum(s["has_hedge"] for s in scored) / n if n else 0.0,
        "rate_explanation":   sum(s["has_explanation"] for s in scored) / n if n else 0.0,
        "rate_bad_starter":   sum(s["has_bad_starter"] for s in scored) / n if n else 0.0,
        "rate_refusal":       sum(s["has_refusal"] for s in scored) / n if n else 0.0,
        "rate_sentence":      sum(s["looks_like_sentence"] for s in scored) / n if n else 0.0,
        "non_compliant_examples": [s["response"] for s in scored if not s["compliant"]],
    }