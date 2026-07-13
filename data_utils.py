"""
data_utils.py — Dataset loading for all supported benchmarks.

Supported datasets
------------------
"truthfulqa"  TruthfulQA multiple-choice (817 questions, 4 options)
              Saturates for strong models like Qwen3-8B (>95% accuracy).
              Kept for reference / backward compatibility.

"gpqa"        Graduate-Level Google-Proof QA (448 questions, 4 options)
              Biology, physics, chemistry. PhD experts reach 65% accuracy.
              Qwen3-8B accuracy ~50-60% — ideal difficulty for observing
              genuine teacher uncertainty in the distribution plots.
              Recommended for Phase 1 pre-experiment.

"mmlu_pro"    MMLU-Pro (12 000 questions, up to 10 options)
              14 domains. Causes 16-33% accuracy drop vs original MMLU.
              10 options make distribution saturation much harder.
              Recommended for Phase 1 main experiment (large scale).

"simpleqa"    SimpleQA short-form open-ended QA (4326 questions)
              Phase 2 dataset. No fixed choice labels — full vocab matters.

Public API
----------
  load_dataset_items(dataset, num_samples)  ->  list[dict]
      Every dict contains at minimum:
        "prompt"  : str          full prompt string fed to the model
        "choices" : list[str]    option labels, e.g. ["A","B","C","D"]
                                 empty list for open-ended datasets
        "question": str          raw question text (for display / logging)

  load_prompts(dataset, num_samples)  ->  list[str]
      Convenience wrapper — returns only the prompt strings.
"""

from __future__ import annotations
import random


# ── Unified entry point ───────────────────────────────────────────────────────

def load_dataset_items(dataset: str, num_samples: int) -> list[dict]:
    """
    Dispatch to the correct loader based on the dataset name.

    Args:
        dataset    : one of "truthfulqa", "gpqa", "mmlu_pro", "simpleqa"
        num_samples: maximum number of items to return
    """
    loaders = {
        "truthfulqa": _load_truthfulqa,
        "gpqa":       _load_gpqa,
        "mmlu_pro":   _load_mmlu_pro,
        "simpleqa":   _load_simpleqa,
    }
    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Choose from: {list(loaders.keys())}"
        )
    return loaders[dataset](num_samples)


def load_prompts(dataset: str, num_samples: int) -> list[str]:
    """Return only the prompt strings (drops choice metadata)."""
    return [item["prompt"] for item in load_dataset_items(dataset, num_samples)]


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def _load_truthfulqa(num_samples: int) -> list[dict]:
    """
    Load TruthfulQA multiple-choice from HuggingFace.

    Dataset : truthfulqa/truthful_qa, config=multiple_choice, split=validation
    Size    : 817 questions, variable number of options per question
    Fields  : question, mc1_targets.choices (list of answer strings,
              first element is always the correct answer)

    Limitation:
        TruthfulQA was designed for early GPT-3-scale models.  Strong models
        like Qwen3-8B achieve >95% accuracy, making the distribution saturated
        (correct option probability always near 1.0).  Consider using GPQA
        or MMLU-Pro instead for observing genuine teacher uncertainty.

    Return format per item:
        {
          "prompt"  : "Question: ...\nA. ...\nB. ...\nAnswer:"
          "choices" : ["A", "B", "C", ...]   # labels in prompt order
          "question": str                     # raw question text
          "answer"  : str                     # correct label — always "A",
                                                # since mc1_targets.choices[0]
                                                # is defined to be the correct
                                                # answer and is never shuffled
        }
    """
    from datasets import load_dataset

    print(f"Loading TruthfulQA — {num_samples} samples...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

    items = []
    for row in ds:
        question    = row["question"]
        raw_choices = row["mc1_targets"]["choices"]
        labels      = [chr(65 + i) for i in range(len(raw_choices))]
        options_str = "\n".join(f"{l}. {c}" for l, c in zip(labels, raw_choices))
        # Instruct the model to reply with only the option letter.
        # This concentrates probability mass on A/B/C/D tokens,
        # making raw choice probabilities meaningful without normalisation.
        labels_str = ', '.join(labels)
        prompt     = (
            f"Question: {question}\n"
            f"{options_str}\n"
            f"Answer with only the option letter ({labels_str}), "
            f"no explanation.\nAnswer:"
        )
        items.append({
            "prompt":   prompt,
            "choices":  labels,
            "question": question,
            "answer":   "A",   # mc1_targets.choices[0] is always correct
        })
        if len(items) >= num_samples:
            break

    print(f"  Loaded {len(items)} TruthfulQA items.")
    print(f"  Example:\n    {items[0]['prompt'][:200]}...")
    return items


# ── GPQA ──────────────────────────────────────────────────────────────────────

def _load_gpqa(num_samples: int, subset: str = "gpqa_diamond",
               seed: int = 42) -> list[dict]:
    """
    Load GPQA from HuggingFace.

    Dataset : Idavidrein/gpqa
    Subsets : gpqa_diamond (198 hardest), gpqa_main (448), gpqa_extended (546)
    Split   : train  (the only split available)
    Fields  : "Question", "Correct Answer", "Incorrect Answer 1/2/3"

    Why GPQA is better than TruthfulQA for this study:
        PhD-level questions in biology, physics, chemistry.
        Expert accuracy ~65%, Qwen3-8B accuracy ~50-60%.
        This means the teacher's distribution is genuinely uncertain on many
        questions — the A/B/C/D probabilities are spread out rather than
        collapsing to ~1.0 on one option.  That uncertainty is exactly what
        we need to observe the distillation confidence-compression effect.

    Option shuffling:
        The raw dataset stores the correct answer separately from the three
        incorrect answers.  We shuffle all four into a random order and assign
        labels A/B/C/D so that the correct answer is not always in position A.
        A fixed random seed ensures reproducibility across runs.

    Return format per item:
        {
          "prompt"  : "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:"
          "choices" : ["A", "B", "C", "D"]
          "question": str    raw question text
          "answer"  : str    correct label after shuffling, e.g. "C"
        }
    """
    from datasets import load_dataset

    print(f"Loading GPQA ({subset}) — {num_samples} samples...")
    ds = load_dataset("Idavidrein/gpqa", subset, split="train",
                      trust_remote_code=True)

    rng   = random.Random(seed)
    items = []

    for row in ds:
        question       = row["Question"]
        correct_answer = row["Correct Answer"]
        wrong_answers  = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]

        # Combine and shuffle so correct answer is not always first
        all_options = [correct_answer] + wrong_answers
        rng.shuffle(all_options)

        labels      = ["A", "B", "C", "D"]
        options_str = "\n".join(f"{l}. {o}" for l, o in zip(labels, all_options))
        # Instruct the model to reply with only the option letter.
        labels_str = ', '.join(labels)
        prompt     = (
            f"Question: {question}\n"
            f"{options_str}\n"
            f"Answer with only the option letter ({labels_str}), "
            f"no explanation.\nAnswer:"
        )

        # Track which label the correct answer landed on after shuffling
        correct_label = labels[all_options.index(correct_answer)]

        items.append({
            "prompt":   prompt,
            "choices":  labels,          # always ["A", "B", "C", "D"]
            "question": question,
            "answer":   correct_label,   # ground-truth label, e.g. "C"
        })
        if len(items) >= num_samples:
            break

    print(f"  Loaded {len(items)} GPQA items from subset '{subset}'.")
    print(f"  Example:\n    {items[0]['prompt'][:300]}...")
    return items


# ── MMLU-Pro ──────────────────────────────────────────────────────────────────

def _load_mmlu_pro(num_samples: int,
                   category: str | None = None) -> list[dict]:
    """
    Load MMLU-Pro from HuggingFace.

    Dataset : TIGER-Lab/MMLU-Pro
    Split   : test  (12 032 questions across 14 domains)
    Fields  : question (str), options (list[str], up to 10 items),
              answer (str, correct label A-J), category (str)

    Why MMLU-Pro is better than TruthfulQA:
        10 answer options vs 4, making saturation much harder.
        Strong models still fail on ~15-40% of questions depending on category.
        Large scale (12 000 questions) supports robust Stage 1 statistics.

    Args:
        num_samples: number of items to return
        category   : if set, filter to only this domain, e.g. "physics",
                     "biology", "chemistry", "math", "computer science".
                     If None, sample uniformly across all categories.

    Return format per item:
        {
          "prompt"  : "Question: ...\nA. ...\nB. ...\n...\nAnswer:"
          "choices" : ["A", "B", ..., "J"]  # only as many as exist (up to 10)
          "question": str
          "answer"  : str   correct label, e.g. "D"
          "category": str   domain name
        }
    """
    from datasets import load_dataset

    filter_msg = f", category='{category}'" if category else ""
    print(f"Loading MMLU-Pro — {num_samples} samples{filter_msg}...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    items = []
    for row in ds:
        # Optionally filter by category
        if category and row["category"].lower() != category.lower():
            continue

        question = row["question"]
        # options is a list of answer strings; filter out any "N/A" padding
        raw_options = [o for o in row["options"] if o.strip() != "N/A"]
        labels      = [chr(65 + i) for i in range(len(raw_options))]   # A-J
        options_str = "\n".join(f"{l}. {o}" for l, o in zip(labels, raw_options))
        # Instruct the model to reply with only the option letter.
        labels_str = ', '.join(labels)
        prompt     = (
            f"Question: {question}\n"
            f"{options_str}\n"
            f"Answer with only the option letter ({labels_str}), "
            f"no explanation.\nAnswer:"
        )

        items.append({
            "prompt":    prompt,
            "choices":   labels,
            "question":  question,
            "answer":    row["answer"],     # correct label stored directly
            "category":  row["category"],
        })
        if len(items) >= num_samples:
            break

    print(f"  Loaded {len(items)} MMLU-Pro items.")
    if items:
        cats = {}
        for it in items:
            cats[it["category"]] = cats.get(it["category"], 0) + 1
        top = sorted(cats.items(), key=lambda x: -x[1])[:5]
        print(f"  Top categories: {top}")
        print(f"  Example:\n    {items[0]['prompt'][:300]}...")
    return items


# ── SimpleQA ──────────────────────────────────────────────────────────────────

_SIMPLEQA_N_FEWSHOT = 5

_STRICT_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer the question with only the minimal factual answer string.\n"
    "Do not write a full sentence.\n"
    "Do not include explanations, context, hedging, or punctuation.\n"
    "Do not start with phrases like 'The answer is' or 'It is'.\n"
    "Use the most common valid form of the answer.\n"
    "Answer:"
)


def _build_simpleqa_fewshot_prefix(ds) -> str:
    """
    5 (question, answer) pairs pulled from the LAST _SIMPLEQA_N_FEWSHOT rows
    of the dataset — disjoint from every row _load_simpleqa() actually uses
    as a teacher/student question (it only iterates over
    ds[:-_SIMPLEQA_N_FEWSHOT]). HuggingFace's load_dataset for split="test"
    returns rows in a fixed order (no shuffling), so this is deterministic
    across calls: teacher and student — and any future rerun — all get the
    exact same 5 examples without needing to persist a random seed.
    """
    tail = list(ds)[-_SIMPLEQA_N_FEWSHOT:]
    prefix = "Answer the following question as briefly as possible.\n\n"
    for row in tail:
        prefix += f"Question: {row['problem']}\nAnswer: {row['answer']}\n\n"
    return prefix


def _load_simpleqa(num_samples: int) -> list[dict]:
    """
    Load SimpleQA from HuggingFace.

    Dataset : basicv8vc/SimpleQA, split=test
    Size    : 4326 short factual questions with a unique correct answer.
              The last _SIMPLEQA_N_FEWSHOT=5 rows are ALWAYS reserved
              (regardless of cfg.prompt_style) and never used as an actual
              teacher/student question — so the usable pool is 4321, not
              4326. This is deliberate even for "strict" (which doesn't
              need a few-shot pool at all): keeping the same question set
              under both prompt styles means a strict-vs-fewshot
              comparison isn't confounded by also comparing different
              questions.
    Fields  : "problem" (question string)

    Prompt (cfg.prompt_style, set via config.py or --prompt_style CLI flag):
      "fewshot" (default) — Farquhar et al. (2024) / Kossen et al.
          (2024)'s short-phrase template: "answer briefly" instruction +
          5 few-shot (question, answer) pairs + the real question.
      "strict" — the original zero-shot, many-negative-constraints
          instruction (no few-shot). Kept available for direct comparison
          against "fewshot" on the identical 4321-question set — see
          conversation history for why "fewshot" is otherwise preferred
          (measurably better accuracy at similar-or-better format
          compliance, on the samples tested so far).

    Return format per item:
        {
          "prompt"  : str   the fully-constructed prompt for this question
          "choices" : []    empty — open-ended generation, no fixed labels
          "question": str   raw question text (instruction-free, for display)
          "answer"  : str   the reference (gold) answer string, used by
                            evaluate.py for NLI-based entailment scoring
        }
    """
    from datasets import load_dataset
    from config import cfg

    print(f"Loading SimpleQA — {num_samples} samples (prompt_style={cfg.prompt_style!r})...")
    ds = load_dataset("basicv8vc/SimpleQA", split="test")

    if cfg.prompt_style == "fewshot":
        fewshot_prefix = _build_simpleqa_fewshot_prefix(ds)
        usable_rows = list(ds)[:-_SIMPLEQA_N_FEWSHOT]
    elif cfg.prompt_style == "strict":
        fewshot_prefix = None
        usable_rows = list(ds)
    else:
        raise ValueError(
            f"Unknown cfg.prompt_style: {cfg.prompt_style!r}. Use 'strict' or 'fewshot'."
        )

    if num_samples > len(usable_rows):
        print(f"  WARNING: requested {num_samples} samples but only "
              f"{len(usable_rows)} are available — returning "
              f"{len(usable_rows)} items instead.")

    items = []
    for row in usable_rows:
        question = row["problem"]
        if cfg.prompt_style == "fewshot":
            prompt = fewshot_prefix + f"Question: {question}\nAnswer:"
        else:   # "strict"
            prompt = _STRICT_PROMPT_TEMPLATE.format(question=question)
        items.append({
            "prompt":   prompt,
            "choices":  [],
            "question": question,
            "answer":   row["answer"],
        })
        if len(items) >= num_samples:
            break

    print(f"  Loaded {len(items)} SimpleQA items.")
    print(f"  Example prompt:\n{items[0]['prompt']}")
    return items
