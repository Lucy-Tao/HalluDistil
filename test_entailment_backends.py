"""
test_entailment_backends.py -- small-scale comparison: run the SAME sets
of candidate answers through both entailment backends (LLM judge vs
DeBERTa) and compare the resulting clusters/entropy side by side, before
deciding whether to switch compute_claim_entropy.py over to DeBERTa for
speed.

This does NOT run generation -- it uses a handful of FIXED, realistic
answer sets (drawn from actual outputs already seen in this project's
wandb runs, e.g. the Kang Ji-hwan case) so the comparison isolates just
the judge/backend difference, not generation variance on top of it.

Confirmed against the real semantic_utils.py (project owner shared the
actual file): load_nli_model(), load_local_llm_judge(), cluster_by_
entailment(), and compute_semantic_distribution() are all called below
exactly matching their real signatures -- in particular,
compute_semantic_distribution(responses, clusters) needs BOTH arguments
and returns a dict (key "semantic_entropy"), and cluster_by_entailment()
returns clusters as lists of INDICES into `responses`, not the response
text itself (mapped back to text below for readable printing).

Usage
-----
  python test_entailment_backends.py
"""
import numpy as np

from config import cfg
from model_utils import load_model_and_tokenizer
from semantic_utils import cluster_by_entailment, compute_semantic_distribution, load_nli_model


# Fixed test cases: (description, question, list_of_4_answers).
# Case 1 and 2 are drawn from the actual Kang Ji-hwan wandb run seen in
# this project -- Case 1 is the "everyone agrees" case (expected entropy
# ~0), Case 2 has real disagreement (expected entropy > 0). Case 3 and 4
# are hand-written to test edge cases: near-duplicate phrasing that
# SHOULD cluster together, and answers that are similar in topic but
# actually different facts (should NOT cluster together).
TEST_CASES = [
    (
        "easy/consistent (real case)",
        "What nationality is Kang Ji-hwan?",
        ["South Korean", "South Korean", "South Korean", "South Korean"],
    ),
    (
        "genuine disagreement (real case)",
        "Where did Kang Ji-hwan originate from?",
        ["Seoul, South Korea", "South Korea", "South Korea", "Busan, South Korea"],
    ),
    (
        "near-duplicate phrasing (should cluster together)",
        "What is his profession?",
        ["actor", "an actor", "He is an actor.", "Actor"],
    ),
    (
        "similar topic, different facts (should NOT fully cluster)",
        "Where did he study?",
        ["Seoul National University", "Hanyang University", "Seoul National University", "Yonsei University"],
    ),
]


def run_backend(backend_name, judge_model, judge_tokenizer):
    print(f"\n{'=' * 70}\nBackend: {backend_name}\n{'=' * 70}")
    for desc, question, answers in TEST_CASES:
        clusters = cluster_by_entailment(
            answers, judge_model, judge_tokenizer,
            strict_entailment=True, backend=backend_name, question=question,
        )
        result = compute_semantic_distribution(answers, clusters)
        # clusters is a list of INDEX lists into `answers`, not text --
        # map back to the actual strings for a readable printout.
        clusters_as_text = [[answers[i] for i in cluster] for cluster in clusters]
        print(f"\n[{desc}]")
        print(f"  question: {question}")
        print(f"  answers:  {answers}")
        print(f"  clusters: {clusters_as_text}")
        print(f"  entropy:  {result['semantic_entropy']:.4f}")


def main():
    print(f"Loading LLM judge: {cfg.entailment_llm_model_name}")
    llm_judge_model, llm_judge_tokenizer = load_model_and_tokenizer(
        cfg.entailment_llm_model_name, device_map=cfg.device_map
    )
    run_backend("llm", llm_judge_model, llm_judge_tokenizer)

    print(f"\nLoading DeBERTa entailment model...")
    deberta_model, deberta_tokenizer = load_nli_model()
    run_backend("deberta", deberta_model, deberta_tokenizer)


if __name__ == "__main__":
    main()