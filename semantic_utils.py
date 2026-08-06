"""
semantic_utils.py — Semantic entropy computation for open-ended QA (Phase 2).

This replaces the single-forward-pass choice-probability approach used for
MCQ datasets (gpqa, mmlu_pro, truthfulqa) with the standard semantic entropy
protocol from Farquhar et al. (2024), "Detecting hallucinations in large
language models using semantic entropy":

  1. Sample N complete responses from the model (full text generation,
     not a single forward pass — open-ended answers have no fixed token set).
  2. Cluster the N responses by meaning using an entailment judge: two
     responses belong to the same cluster only if each one entails the
     other (bidirectional entailment).
  3. The "probability" of each semantic cluster is its frequency among
     the N samples (cluster_size / N).
  4. Semantic entropy is the Shannon entropy of this cluster-frequency
     distribution — analogous to choice_entropy in the MCQ phase, but
     computed over meaning-clusters instead of fixed answer labels.

Why bidirectional entailment, not exact string match:
    Two responses like "Alexander Graham Bell" and "It was Bell" are the
    same answer in meaning but different strings. Exact match would treat
    them as different answers and overestimate uncertainty. Bidirectional
    entailment correctly merges them into one cluster.

Entailment judge — two interchangeable backends, selected via the
`backend` argument threaded through get_entailment_prob() /
cluster_by_entailment() / get_semantic_entropy() (default "deberta",
matching the original protocol):

  (A) "deberta" (default) — microsoft/deberta-large-mnli, a dedicated NLI
      classifier (see cfg.nli_model_name). Loaded via load_nli_model().
      Single forward pass per pairwise check, returns a clean 3-way
      softmax [contradiction, neutral, entailment]. Fast and cheap, but a
      ~400M-param model with known limitations on subtle/close paraphrases.

  (B) "llm" — an open-weight instruction-tuned LLM used as the judge
      instead of DeBERTa, loaded via load_local_llm_judge() (default
      "meta-llama/Llama-3.1-8B-Instruct", see cfg.entailment_llm_model_name).
      The judge is asked the same question-conditioned entailment question
      used in the literature (Kuhn et al. 2023; Farquhar et al. 2024;
      replicated in Nay Myat Min et al. 2026's "Propaganda AI" Appendix
      C.2): "Does Possible Answer 1 semantically entail Possible Answer 2?
      Respond with entailment, contradiction, or neutral." — generated as
      free text, then checked for "entailment" as a substring of the
      (lowercased) response, exactly as in Propaganda AI's
      CheckBidirectionalEntailment algorithm. Returns a discrete 1.0/0.0,
      not a calibrated probability — cluster_by_entailment()'s
      threshold=0.5 check still works correctly against this.

  To compare judges (e.g. Llama-3.1-8B-Instruct vs Qwen2.5-14B-Instruct vs
  DeBERTa), load each judge once via the matching load_*() function and
  re-run get_semantic_entropy() with a different `judge_backend` /
  judge model each time — see visualize.py's cfg.entailment_backend switch.

Public API
----------
  sample_responses(model, tokenizer, prompt, n_samples, ...) -> list[str]
      Generate N complete responses via repeated sampling.

  load_nli_model(model_name) -> (nli_model, nli_tokenizer)
      Load the DeBERTa-large-mnli NLI classifier (backend "deberta").

  load_local_llm_judge(model_name) -> (judge_model, judge_tokenizer)
      Load a local open-weight instruction-tuned LLM to use as the
      entailment judge instead of DeBERTa (backend "llm"). Drop-in
      replacement for load_nli_model() — pass the result into the same
      get_semantic_entropy() call, with judge_backend="llm".

  cluster_by_entailment(responses, nli_model, nli_tokenizer, threshold, backend, question) -> list[list[int]]
      Partition response indices into semantic clusters via bidirectional
      entailment. Returns a list of clusters, each a list of indices into
      the original `responses` list.

  compute_semantic_distribution(responses, clusters) -> dict
      Convert clusters into a frequency distribution and compute entropy.
"""

from __future__ import annotations

import re

import torch

from config import cfg

DEFAULT_STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]

# ══════════════════════════════════════════════════════════════
# Step 1 — Sample N complete responses
# ══════════════════════════════════════════════════════════════

def sampling_params_for(model_name: str) -> dict:
    """Published sampling recommendation for the model's family."""
    m = model_name.lower()
    if "qwen" in m:
        return {"top_p": 0.8, "top_k": 20}
    if "llama" in m:
        return {"top_p": 0.9, "top_k": None}
    return {"top_p": None, "top_k": None}


@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    max_new_tokens: int = 50,
    stop_sequences: list[str] | None = DEFAULT_STOP_SEQUENCES,
    system_prompt: str | None = None,
) -> list[str]:
    """
    Generate n_samples complete responses to the same prompt via repeated
    sampling at the given temperature.

    Unlike the MCQ phase (which only needs one forward pass), open-ended
    generation requires full autoregressive decoding for each sample,
    since the answer is not a single token from a known small set.

    Each call to model.generate() is independent — there is no shared
    state between samples, so results reflect the model's natural sampling
    variability at this temperature.

    Args:
        model         : HuggingFace CausalLM in eval mode.
        tokenizer     : matching tokenizer.
        prompt        : the full prompt string (already includes the
                        "answer with a short phrase" instruction).
        n_samples     : how many independent responses to generate.
        temperature   : sampling temperature; > 0 enables do_sample=True.
        max_new_tokens: cap on response length (short, since we expect
                        a short phrase per the prompt instruction).

    Returns:
        list[str] of length n_samples, decoded response texts
        (prompt stripped, special tokens removed, whitespace trimmed).
    """
    model.eval()
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs     = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        renormalize_logits=True,
    )
    # Truncation follows each family's own published recommendation
    # rather than one global setting, because the families disagree:
    # Qwen recommends 0.8/20 for non-thinking mode, Llama 3.1 ships
    # 0.9 with no top_k, and OLMo 2 publishes nothing at all. AUROC is
    # a rank statistic, so entropy scales need only be consistent
    # within a family, which is where teacher and student are compared.
    # repetition_penalty is deliberately left to each model's own
    # generation_config. 1.3 was tried earlier and produced degenerate
    # multilingual output.
    if top_p is not None:
        generate_kwargs["top_p"] = top_p
    if top_k is not None:
        generate_kwargs["top_k"] = top_k
    if stop_sequences:
        generate_kwargs["stop_strings"] = stop_sequences
        generate_kwargs["tokenizer"]    = tokenizer

    responses = []
    for _ in range(n_samples):
        output_ids = model.generate(**inputs, **generate_kwargs)
        new_ids  = output_ids[0][prompt_len:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if stop_sequences:
            for stop in stop_sequences:
                if response.endswith(stop.strip()) and stop.strip():
                    response = response[: -len(stop.strip())].strip()
                    break
        responses.append(response)

    return responses


# ══════════════════════════════════════════════════════════════
# Step 2 — Load the entailment judge
# ══════════════════════════════════════════════════════════════

def load_nli_model(model_name: str | None = None):
    """
    Load the DeBERTa-v2-xlarge-mnli NLI classifier (backend "deberta", the
    default and the original protocol from Kuhn et al. 2023 / Farquhar
    et al. 2024).

    Args:
        model_name: HF model id. Defaults to cfg.nli_model_name
            (typically "microsoft/deberta-v2-xlarge-mnli") if not given.

    Returns:
        (nli_model, nli_tokenizer)

    Note on label order:
        microsoft/deberta-v2-xlarge-mnli outputs 3-way logits in the order
        [contradiction, neutral, entailment] — this ordering is used by
        get_entailment_prob() below.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_name is None:
        model_name = cfg.nli_model_name

    print(f"  Loading NLI model: {model_name}")
    nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nli_model.to(device)
    nli_model.eval()
    print(f"  NLI model device: {device}")
    return nli_model, nli_tokenizer


def load_local_llm_judge(model_name: str | None = None):
    """
    Load a local, open-weight instruction-tuned LLM to use as the
    bidirectional-entailment judge instead of DeBERTa (backend "llm").

    Args:
        model_name: HF model id, e.g. "meta-llama/Llama-3.1-8B-Instruct"
            (the default — see cfg.entailment_llm_model_name) or
            "Qwen/Qwen2.5-14B-Instruct" for comparison.

    Returns:
        (judge_model, judge_tokenizer) via model_utils.load_model_and_tokenizer,
        so this stays consistent with how every other model in this
        pipeline is loaded (same dtype, device_map, padding behaviour).
        Pass the result into get_semantic_entropy(..., judge_backend="llm").

    To compare judges, just call this again with a different model_name
    and re-run the same question through get_semantic_entropy() with
    judge_backend="llm" — see cfg.entailment_backend / cfg.entailment_llm_model_name.
    """
    from model_utils import load_model_and_tokenizer

    if model_name is None:
        model_name = cfg.entailment_llm_model_name

    print(f"  Loading local LLM entailment judge: {model_name}")
    return load_model_and_tokenizer(model_name, device_map=cfg.device_map)


# Question-conditioned entailment prompt for the "llm" backend, following
# the protocol used by Kuhn et al. (2023) / Farquhar et al. (2024) and
# replicated verbatim (modulo variable names) in Nay Myat Min et al.
# (2026)'s "Propaganda AI" Appendix C.2. The question is included for
# context, since e.g. "Paris" alone doesn't entail "The capital of France
# is Paris" without knowing the question is about France's capital.
#
# This generates ONE of {entailment, contradiction, neutral} as free text
# rather than reading a Yes/No token probability — matching the literature
# exactly, including how it's parsed (substring match on "entailment" in
# the lowercased response, per Propaganda AI's Algorithm 2 / CheckBidirectionalEntailment).
_LLM_JUDGE_PROMPT_TEMPLATE = (
    "We are evaluating answers to the question {question}\n"
    "Here are two possible answers:\n"
    "Possible Answer 1: {premise}\n"
    "Possible Answer 2: {hypothesis}\n"
    "Does Possible Answer 1 semantically entail Possible Answer 2? "
    "Respond with entailment, contradiction, or neutral.\n"
    "Answer:"
)


@torch.no_grad()
def _get_entailment_prob_llm(
    judge_model,
    judge_tokenizer,
    question: str,
    premise: str,
    hypothesis: str,
    max_new_tokens: int = 10,
) -> float:
    """
    Prompt the local LLM judge with the question-conditioned entailment
    prompt above and generate a short free-text response, then check
    whether "entailment" appears in it (case-insensitive substring match,
    same as Propaganda AI's CheckBidirectionalEntailment: `"entailment" in
    lower(response)`).

    Returns 2 for entailment, 0 for contradiction, 1 for neutral —
    matching the DeBERTa-mnli label convention (0/1/2), so that
    cluster_by_entailment()'s `== 2` checks work identically for
    both backends.
    """
    prompt_text = _LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=question, premise=premise, hypothesis=hypothesis,
    )
    messages = [{"role": "user", "content": prompt_text}]
    try:
        text = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
    output_ids = judge_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(cfg.judge_temperature > 0),
        temperature=cfg.judge_temperature if cfg.judge_temperature > 0 else None,
        pad_token_id=judge_tokenizer.pad_token_id,
    )
    new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
    response = judge_tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
    if "entailment" in response:
        return 2
    if "contradiction" in response:
        return 0
    return 1


def get_entailment_prob(
    nli_model,
    nli_tokenizer,
    premise: str,
    hypothesis: str,
    backend: str = "deberta",
    question: str | None = None,
) -> float:
    """
    Return P(entailment) for "premise entails hypothesis" under the
    selected judge backend.

    Args:
        premise   : the first response (treated as the supporting text).
        hypothesis: the second response (treated as the claim being tested).
        backend   : "deberta" (default) or "llm" — see module docstring.
        question  : required when backend="llm" (the LLM judge prompt is
            question-conditioned, per Kuhn et al. 2023). Ignored for
            backend="deberta", which classifies the premise/hypothesis
            pair directly without question context (the original,
            unchanged DeBERTa behaviour).

    Returns:
        float in [0, 1].

    Note:
        Entailment is directional: P(premise -> hypothesis) is not
        generally equal to P(hypothesis -> premise). Bidirectional
        entailment (both directions above threshold) is checked by the
        caller in cluster_by_entailment().
    """
    # A string trivially entails itself — this should never depend on a
    # judge model's mood. Skipping the judge call here both (a) guarantees
    # exact-duplicate responses always end up in the same cluster instead
    # of occasionally being split apart by an unreliable judge call (the
    # "entailment" keyword not appearing verbatim in an LLM judge's free-
    # text response is enough to misfire this for the LLM backend — see
    # _get_entailment_prob_llm), and (b) saves a judge call entirely for
    # what is a common case (repeated sampling often produces the exact
    # same short answer multiple times).
    if premise.strip() == hypothesis.strip():
        return 2

    if backend == "llm":
        if question is None:
            raise ValueError(
                "backend='llm' requires `question` for context (the LLM "
                "judge prompt is question-conditioned, per Kuhn et al. "
                "2023) — pass it through from cluster_by_entailment() / "
                "get_semantic_entropy()."
            )
        return _get_entailment_prob_llm(
            nli_model, nli_tokenizer, question, premise, hypothesis
        )

    if backend == "deberta":
        inputs = nli_tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True
        ).to(nli_model.device)
        with torch.no_grad():
            logits = nli_model(**inputs).logits[0]
        return int(torch.argmax(logits).item())

    raise ValueError(f"Unknown backend: {backend!r}. Use 'deberta' or 'llm'.")


# ══════════════════════════════════════════════════════════════
# Step 3 — Cluster responses by bidirectional entailment
# ══════════════════════════════════════════════════════════════

def cluster_by_entailment(
    responses: list[str],
    nli_model,
    nli_tokenizer,
    strict_entailment: bool = True,
    backend: str = "deberta",
    question: str | None = None,
) -> list[list[int]]:
    n = len(responses)
    semantic_set_ids = [-1] * n
    next_id = 0

    for i, resp_i in enumerate(responses):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id

            for j in range(i + 1, n):
                resp_j = responses[j]

                implication_1 = get_entailment_prob(
                    nli_model, nli_tokenizer, resp_i, resp_j,
                    backend=backend, question=question,
                )
                implication_2 = get_entailment_prob(
                    nli_model, nli_tokenizer, resp_j, resp_i,
                    backend=backend, question=question,
                )

                if strict_entailment:
                    equivalent = (implication_1 == 2) and (implication_2 == 2)
                else:
                    implications = [implication_1, implication_2]
                    equivalent = (0 not in implications) and ([1, 1] != implications)

                if equivalent:
                    semantic_set_ids[j] = next_id

            next_id += 1

    from collections import defaultdict
    clusters_map = defaultdict(list)
    for idx, cluster_id in enumerate(semantic_set_ids):
        clusters_map[cluster_id].append(idx)

    return list(clusters_map.values())


# ══════════════════════════════════════════════════════════════
# Step 4 — Convert clusters into a probability distribution + entropy
# ══════════════════════════════════════════════════════════════

def compute_semantic_distribution(
    responses: list[str],
    clusters: list[list[int]],
) -> dict:
    """
    Convert semantic clusters into a frequency distribution and compute
    the semantic entropy of that distribution.

    Args:
        responses: the original list of N sampled response strings.
        clusters : output of cluster_by_entailment() — list of index lists.

    Returns:
        {
          "cluster_probs"  : dict[str, float]
              Maps a representative response string (the most common exact
              string within that cluster) to its frequency (cluster_size / N).
              Sums to 1.0 across all clusters.

              NOTE: it is possible for two DIFFERENT clusters to pick the
              same representative string (e.g. clustering failed to merge
              two groups of identical-text responses into one cluster).
              When that happens their frequencies are ADDED together under
              that one key — never silently overwritten — so cluster_probs
              always sums to 1.0 regardless of how many clusters mapped to
              the same display string. (An earlier version of this
              function used plain dict assignment here, which let a later
              cluster's frequency silently replace an earlier one sharing
              the same representative string, undercounting the true
              distribution — e.g. exact-duplicate responses split across
              several singleton clusters by an imperfect judge call would
              each show up as their own 1/N entry, then collapse into a
              single 1/N entry in the final dict instead of summing to
              their true combined frequency.)

          "cluster_members": dict[str, list[str]]
              Maps the same representative string to the full list of raw
              responses that were merged into ANY cluster sharing that
              representative — concatenated across colliding clusters, not
              overwritten, for the same reason as cluster_probs above.

          "predicted_response": str
              The representative string of the largest cluster (the most
              frequent semantic answer).

          "semantic_entropy": float
              Shannon entropy (nats) of the cluster-frequency distribution.
              Maximum value = ln(n_samples) when every sample is its own
              singleton cluster (maximum disagreement / hallucination).
              Minimum value = 0 when all samples land in one cluster
              (the model is fully consistent across samples). Reported via
              abs() to avoid displaying IEEE754 negative zero ("-0.000")
              in the all-one-cluster case — entropy is never actually
              negative, that's purely a floating-point sign artifact of
              -sum(1.0 * log(1.0)).
        }
    """
    import math
    from collections import Counter

    n_total = len(responses)
    cluster_probs   = {}
    cluster_members = {}

    for cluster in clusters:
        member_texts = [responses[idx] for idx in cluster]
        # Representative label: the most common exact string in this cluster
        # (falls back to the first response if all strings are unique).
        most_common = Counter(member_texts).most_common(1)[0][0]

        freq = len(cluster) / n_total
        # Accumulate rather than overwrite — see the cluster_probs docstring
        # note above for why this matters whenever two separate clusters
        # happen to share the same representative string.
        cluster_probs[most_common] = cluster_probs.get(most_common, 0.0) + freq
        cluster_members.setdefault(most_common, []).extend(member_texts)

    predicted = max(cluster_probs, key=cluster_probs.get)

    probs_arr = list(cluster_probs.values())
    semantic_entropy = abs(-sum(p * math.log(p) for p in probs_arr if p > 0))

    return {
        "cluster_probs":      cluster_probs,
        "cluster_members":    cluster_members,
        "predicted_response": predicted,
        "semantic_entropy":   float(semantic_entropy),
    }


# ══════════════════════════════════════════════════════════════
# Convenience: full pipeline in one call
# ══════════════════════════════════════════════════════════════

def get_semantic_entropy(
    model,
    tokenizer,
    prompt: str,
    nli_model,
    nli_tokenizer,
    n_samples: int = 5,
    temperature: float = 1.0,
    max_new_tokens: int = 50,
    strict_entailment: bool = True,
    fixed_response: str | None = None,
    judge_backend: str = "deberta",
    question: str | None = None,
) -> dict:
    """
    Run the full semantic entropy pipeline for one prompt on one model:
    sample N responses -> cluster by entailment -> compute distribution + entropy.

    This is the single function visualize.py calls per model per prompt.
    The nli_model/nli_tokenizer are passed in (loaded once, shared across
    all three generation models) rather than reloaded here.

    Args:
        fixed_response : if set, this response is treated as ALREADY sampled
            (e.g. it's the teacher's response that was actually used as
            distillation training data — see distill.py's
            load_teacher_distill_response()). Only n_samples - 1 NEW
            responses are generated; fixed_response is combined with them
            before clustering, so the reported distribution/entropy reflects
            all n_samples responses together, and no compute is wasted
            re-sampling something we already have on disk.
            If None (default), all n_samples responses are freshly sampled,
            matching the previous behaviour exactly.
        judge_backend  : "deberta" (default) or "llm" — which entailment
            judge cluster_by_entailment() should use. See module docstring
            and load_nli_model() / load_local_llm_judge().
        question       : required when judge_backend="llm" (the LLM judge
            prompt is question-conditioned, per Kuhn et al. 2023). Ignored
            for judge_backend="deberta".

    Returns the same dict as compute_semantic_distribution(), with the raw
    sampled responses also included for transparency. When fixed_response
    is set, two extra keys are added:
        "fixed_response"         : the input fixed_response, echoed back.
        "fixed_response_cluster" : the representative string of whichever
            cluster fixed_response landed in — this is what Panel 2 in
            visualize.py highlights instead of the modal answer. None if
            fixed_response was somehow not found in any cluster (should not
            happen in practice, since it is always included as the first
            sample below).
    """
    if fixed_response is not None:
        n_to_sample = max(n_samples - 1, 0)
        new_responses = (
            sample_responses(
                model, tokenizer, prompt,
                n_samples=n_to_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            if n_to_sample > 0 else []
        )
        responses = [fixed_response] + new_responses
        fixed_idx = 0
    else:
        responses = sample_responses(
            model, tokenizer, prompt,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        fixed_idx = None

    clusters = cluster_by_entailment(
        responses, nli_model, nli_tokenizer, strict_entailment,
        backend=judge_backend, question=question,
    )
    result   = compute_semantic_distribution(responses, clusters)
    result["raw_responses"] = responses

    if fixed_idx is not None:
        from collections import Counter

        result["fixed_response"] = fixed_response
        result["fixed_response_cluster"] = None
        for cluster in clusters:
            if fixed_idx in cluster:
                member_texts = [responses[i] for i in cluster]
                result["fixed_response_cluster"] = Counter(member_texts).most_common(1)[0][0]
                break

    return result


# ══════════════════════════════════════════════════════════════
# Correctness judging — SimpleQA official grader (LLM only)
# ══════════════════════════════════════════════════════════════
#
# Correctness is graded with the official SimpleQA grader prompt
# (openai/simple-evals, simpleqa_eval.py GRADER_TEMPLATE): a predicted
# answer is classified against a gold target as one of
#     CORRECT / INCORRECT / NOT_ATTEMPTED   (returned as A / B / C).
# NOT_ATTEMPTED natively captures abstention/hedging, so downstream code
# can drop it before AUROC (grade == "NOT_ATTEMPTED") instead of relying
# on separate phrase-matching.
#
# Deliberately DECOUPLED from cfg.entailment_backend: the grader is ALWAYS
# an LLM (DeBERTa cannot run a generative A/B/C prompt). Two interchangeable
# backends — QwenGrader (local HF model) and GptGrader (OpenAI-compatible
# API). The clustering judge (entailment_backend) is chosen independently.
#
# Supersedes the old judge_correctness() / judge_correctness_llm(), which
# are removed.

import os

SIMPLEQA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

_GRADE_LETTER_TO_LABEL = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def _parse_grade_letter(text: str) -> str:
    """Parse the grader's reply into CORRECT / INCORRECT / NOT_ATTEMPTED.

    Official simple-evals just does re.search(r'(A|B|C)', reply); we make it
    slightly more robust because Qwen (non-thinking) is less letter-compliant
    than gpt-4.1 and may echo the word. Priority: a standalone A/B/C letter
    first (word-boundary avoids matching the 'A' inside 'ATTEMPTED'), then a
    full-word fallback (INCORRECT checked before CORRECT since the former
    contains the latter as a substring). Unparseable -> NOT_ATTEMPTED
    (conservative: an ungradeable reply is not a pass)."""
    t = text.strip().upper()
    m = re.search(r"\b([ABC])\b", t)
    if m:
        return _GRADE_LETTER_TO_LABEL[m.group(1)]
    for label in ("NOT_ATTEMPTED", "INCORRECT", "CORRECT"):
        if label in t:
            return label
    return "NOT_ATTEMPTED"

class QwenGrader:
    """SimpleQA grader backed by a local Qwen3 (or any local chat LLM).

    Qwen3 non-thinking mode, official recommended sampling
    (temperature=0.7, top_p=0.8, top_k=20). Decoupled from
    cfg.judge_temperature (that still governs the clustering LLM judge)."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.no_grad()
    def grade(self, question: str, target: str, predicted: str) -> str:
        prompt = SIMPLEQA_GRADER_TEMPLATE.format(
            question=question, target=target, predicted_answer=predicted)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)          # non-thinking
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=True,
            temperature=0.7, top_p=0.8, top_k=20,   # Qwen3 non-thinking recommended
            pad_token_id=self.tokenizer.pad_token_id)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return _parse_grade_letter(reply)


class GptGrader:
    """SimpleQA grader backed by an OpenAI-compatible API (Lagrange gateway).

    temperature=0.5 per your setting. Reads base_url/api_key from env
    (LAGRANGE_BASE_URL / LAGRANGE_API_KEY) unless passed explicitly."""

    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or os.environ.get(
                "LAGRANGE_BASE_URL",
                "https://lagrange.uksouth.cloudapp.azure.com/openai"),
            api_key=api_key or os.environ["LAGRANGE_API_KEY"])

    def grade(self, question: str, target: str, predicted: str) -> str:
        prompt = SIMPLEQA_GRADER_TEMPLATE.format(
            question=question, target=target, predicted_answer=predicted)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=8)
        return _parse_grade_letter(resp.choices[0].message.content or "")
