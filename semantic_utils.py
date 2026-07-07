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

import torch

from config import cfg


# ══════════════════════════════════════════════════════════════
# Step 1 — Sample N complete responses
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float = 1.0,
    max_new_tokens: int = 30,
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
    messages = [{"role": "user", "content": prompt}]
    # enable_thinking=False disables Qwen3's <think>...</think> reasoning
    # block. Without this, sampled responses are dominated by reasoning
    # text rather than the actual short-phrase answer, which corrupts the
    # semantic clustering step (responses look spuriously similar because
    # they're all truncated mid-reasoning, not because the model agrees
    # on the answer). See prob_utils._build_prompt_text for the same fix
    # applied to the MCQ path.
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

    responses = []
    for _ in range(n_samples):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
        )
        new_ids  = output_ids[0][prompt_len:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        responses.append(response)

    return responses


# ══════════════════════════════════════════════════════════════
# Step 2 — Load the entailment judge
# ══════════════════════════════════════════════════════════════

def load_nli_model(model_name: str | None = None):
    """
    Load the DeBERTa-large-mnli NLI classifier (backend "deberta", the
    default and the original protocol from Kuhn et al. 2023 / Farquhar
    et al. 2024).

    Args:
        model_name: HF model id. Defaults to cfg.nli_model_name
            (typically "microsoft/deberta-large-mnli") if not given.

    Returns:
        (nli_model, nli_tokenizer)

    Note on label order:
        microsoft/deberta-large-mnli outputs 3-way logits in the order
        [contradiction, neutral, entailment] — this ordering is used by
        get_entailment_prob() below.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_name is None:
        model_name = cfg.nli_model_name

    print(f"  Loading NLI model: {model_name}")
    nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    nli_model.eval()
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

    Returns 1.0 if the judge said entailment, 0.0 otherwise (i.e.
    contradiction or neutral) — a discrete decision, not a calibrated
    probability, matching how the literature uses this judge (cluster
    merging is a binary decision: both directions must say "entailment").
    cluster_by_entailment()'s threshold=0.5 check still works correctly
    against this 0.0/1.0 value.
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
        do_sample=False,
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
    """
    Partition response indices into semantic clusters.

    Algorithm: greedy union — for each response, check it against the
    first member of every existing cluster; if bidirectional entailment
    holds, join that cluster; otherwise start a new cluster. This is the
    standard approach used in semantic entropy implementations and is
    O(n_samples^2) in the number of judge calls, which is fine for the
    small n_samples (5-10) used here.

    Args:
        responses    : list of N sampled response strings.
        nli_model    : loaded judge model (see load_nli_model / load_local_llm_judge).
        nli_tokenizer: matching tokenizer.
        strict_entailment: when True, only merge responses if both directions of entailment hold.
        backend      : "deberta" (default) or "llm" — see module docstring.
        question     : required when backend="llm" (question-conditioned
            entailment prompt); ignored for backend="deberta".

    Returns:
        list of clusters, each a list of indices into `responses`.
        E.g. for responses = ["Bell", "It was Bell", "Meucci"], a typical
        result might be [[0, 1], [2]] — two responses merged, one separate.
    """
    clusters: list[list[int]] = []

    for i, resp_i in enumerate(responses):
        placed = False

        for cluster in clusters:
            # Compare against the first member of the cluster as the
            # representative — sufficient in practice for short phrases,
            # and keeps the algorithm at O(n^2) rather than O(n^3).
            j       = cluster[0]
            resp_j  = responses[j]

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
                cluster.append(i)
                placed = True
                break

        if not placed:
            clusters.append([i])

    return clusters


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
    max_new_tokens: int = 30,
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
# Correctness judging (separate from entailment-based clustering)
# ══════════════════════════════════════════════════════════════
#
# This is a DIFFERENT task from everything above. cluster_by_entailment()
# /get_entailment_prob() compare two SAMPLED responses against each other
# with no access to a gold answer — that's what semantic entropy itself
# needs. Correctness judging instead compares ONE response (typically the
# greedy/most-likely answer) against the KNOWN correct answer, and is what
# you need for AUROC analysis (Kuhn et al. 2023): correlating semantic
# entropy against whether the greedy answer was actually right.
#
# Did NOT exist anywhere in this codebase before this addition — if you
# already have correctness-judging logic in compute_auroc.py (or
# elsewhere), check it against this rather than assuming it's redundant;
# share that file and I'll reconcile the two rather than risk two
# different judging methods feeding the same AUROC numbers.

_CORRECTNESS_JUDGE_PROMPT_TEMPLATE = (
    "We are assessing the quality of answers to the following question: {question}\n"
    "The expected answer is: {reference_answer}\n"
    "The proposed answer is: {predicted_answer}\n"
    "Within the context of the question, does the proposed answer mean the "
    "same as the expected answer? Respond only with yes or no.\n"
    "Response:"
)


@torch.no_grad()
def judge_correctness_llm(
    judge_model,
    judge_tokenizer,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    max_new_tokens: int = 10,
) -> bool:
    """
    LLM-judge correctness check: does predicted_answer mean the same thing
    as reference_answer, in the context of question?

    Prompt follows Kossen et al. (2024) / Tjandra et al. (2024)'s
    correctness-assessment protocol (see also Farquhar et al. 2024's
    long-form correctness check) — generates "yes"/"no" as free text and
    checks whether the response starts with "yes", rather than reading
    token-level probabilities, matching how the literature implements
    this step.

    Args:
        judge_model, judge_tokenizer : from load_local_llm_judge() — reuse
            the same judge you use for entailment clustering, or load a
            different one; nothing here requires them to match.
        question          : the original question.
        reference_answer  : the known-correct (gold) answer.
        predicted_answer  : the model's response being graded (typically
            a greedy/T=0 generation, per the AUROC protocol in Kuhn et al.
            2023 — semantic entropy is computed from high-temperature
            samples, but correctness is judged on the low-temperature/
            greedy answer).

    Returns:
        bool — True if the judge says "yes" (correct), False otherwise.
    """
    prompt_text = _CORRECTNESS_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        predicted_answer=predicted_answer,
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
        do_sample=False,
        pad_token_id=judge_tokenizer.pad_token_id,
    )
    new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
    response = judge_tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
    return response.startswith("yes")


def judge_correctness(
    judge_model,
    judge_tokenizer,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    backend: str = "deberta",
) -> bool:
    """
    Backend-dispatching correctness judge — the correctness-checking
    counterpart of get_entailment_prob()'s backend dispatch, so callers
    (e.g. filter_questions.py's scan_model()) can use whichever judge
    backend cfg.entailment_backend points to for BOTH entailment
    clustering and correctness checking, without branching themselves.

    Args:
        backend: "llm" -> judge_correctness_llm() (generates yes/no as
            free text, per Kossen et al. 2024 / Tjandra et al. 2024).
            "deberta" (default) -> bidirectional NLI entailment between
            predicted_answer and reference_answer, matching the original
            Kuhn et al. (2023) protocol used in this codebase's previous
            compute_auroc.py: correct only if BOTH directions entail.
        strict_entailment: only used for backend="deberta".

    Returns:
        bool — True if predicted_answer is judged correct.
    """
    if backend == "llm":
        return judge_correctness_llm(
            judge_model, judge_tokenizer, question, reference_answer, predicted_answer,
        )

    if backend == "deberta":
        implication_pred_to_gold = get_entailment_prob(
            judge_model, judge_tokenizer, predicted_answer, reference_answer,
            backend="deberta",
        )
        implication_gold_to_pred = get_entailment_prob(
            judge_model, judge_tokenizer, reference_answer, predicted_answer,
            backend="deberta",
        )
        return implication_pred_to_gold == 2 and implication_gold_to_pred == 2

    raise ValueError(f"Unknown backend: {backend!r}. Use 'deberta' or 'llm'.")