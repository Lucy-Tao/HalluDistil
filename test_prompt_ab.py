"""
test_prompt_ab.py — A/B test two SimpleQA prompt templates for format
compliance AND accuracy, side by side, on a small sample.

Usage (on the cluster, inside the haldist conda env):
    python3 test_prompt_ab.py --n 50

Writes a CSV to figures/prompt_ab_<timestamp>.csv with per-question results
for both prompts, plus a printed summary table.
"""
import argparse
import csv
import time
from math import comb

from config import cfg
from model_utils import load_model_and_tokenizer
from semantic_utils import sample_responses, judge_correctness, load_local_llm_judge
from prompt_compliance import summarize_batch
from datasets import load_dataset

PROMPT_A_OLD = (
    "Question: {question}\n"
    "Answer the question with only the minimal factual answer string.\n"
    "Do not write a full sentence.\n"
    "Do not include explanations, context, hedging, or punctuation.\n"
    "Do not start with phrases like 'The answer is' or 'It is'.\n"
    "Use the most common valid form of the answer.\n"
    "Answer:"
)

PROMPT_B_NEW = (
    "Answer the following question as briefly as possible.\n"
    "Question: {question}\n"
    "Answer:"
)

# Farquhar et al. 2024 / Kossen et al. 2024's short-phrase template:
# instruction + N few-shot (question, answer) pairs + the real question.
FEWSHOT_INSTRUCTION = "Answer the following question as briefly as possible.\n\n"
FEWSHOT_PAIR_TEMPLATE = "Question: {q}\nAnswer: {a}\n\n"
FEWSHOT_FINAL_TEMPLATE = "Question: {question}\nAnswer:"


def build_fewshot_examples(dataset, n_shot: int = 5, n_eval: int = 0):
    """
    Pull n_shot (question, answer) pairs from the TAIL of the dataset,
    guaranteed not to overlap with the first n_eval rows used for
    evaluation.
    """
    rows = list(dataset)
    tail = rows[-n_shot:]
    assert len(tail) == n_shot
    tail_indices = set(range(len(rows) - n_shot, len(rows)))
    eval_indices = set(range(n_eval))
    assert not (tail_indices & eval_indices), "few-shot pool overlaps with eval set!"
    return [(r["problem"], r["answer"]) for r in tail]

def mcnemar_test(correct_x: list[bool], correct_y: list[bool]) -> dict:
    """
    Exact (binomial-based) McNemar's test for two paired binary
    correctness sequences aligned by question index.
    """
    assert len(correct_x) == len(correct_y)
    x_win = sum(1 for x, y in zip(correct_x, correct_y) if x and not y)
    y_win = sum(1 for x, y in zip(correct_x, correct_y) if y and not x)
    n_discordant = x_win + y_win
    if n_discordant == 0:
        p_value = 1.0
    else:
        k = min(x_win, y_win)
        p_value = sum(comb(n_discordant, i) for i in range(k + 1)) * 2 / (2 ** n_discordant)
        p_value = min(p_value, 1.0)
    return {"x_win": x_win, "y_win": y_win, "n_discordant": n_discordant, "p_value": p_value}


def print_mcnemar_comparisons(results: dict):
    """results: {label: {"correctness": [...bool...], ...}} for each of the 3 prompts."""
    print("\n" + "=" * 60)
    print("PAIRWISE McNEMAR'S TEST (accuracy, per-question paired)")
    print("=" * 60)
    labels = list(results.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            lx, ly = labels[i], labels[j]
            m = mcnemar_test(results[lx]["correctness"], results[ly]["correctness"])
            sig = "significant (p<0.05)" if m["p_value"] < 0.05 else "not significant"
            print(f"\n  {lx} vs {ly}:")
            print(f"    discordant pairs: {m['n_discordant']} "
                  f"({lx} wins {m['x_win']}, {ly} wins {m['y_win']})")
            print(f"    p-value: {m['p_value']:.3f}  -> {sig}")

def build_fewshot_prompt(fewshot_examples: list[tuple[str, str]], question: str) -> str:
    prefix = FEWSHOT_INSTRUCTION
    for q, a in fewshot_examples:
        prefix += FEWSHOT_PAIR_TEMPLATE.format(q=q, a=a)
    return prefix + FEWSHOT_FINAL_TEMPLATE.format(question=question)


def run_prompt_test(model, tokenizer, judge_model, judge_tok, questions, answers, prompt_fn, label):
    responses = []
    correctness = []
    for q, gold in zip(questions, answers):
        prompt = prompt_fn(q)
        r = sample_responses(model, tokenizer, prompt, n_samples=1, temperature=0.1,
                              max_new_tokens=cfg.semantic_max_new_tokens)[0]
        responses.append(r)
        is_correct = judge_correctness(judge_model, judge_tok, q, gold, r,
                                        backend=cfg.entailment_backend)
        correctness.append(is_correct)

    compliance = summarize_batch(responses)
    accuracy = sum(correctness) / len(correctness) if correctness else 0.0

    print(f"\n=== {label} ===")
    print(f"accuracy:          {accuracy:.1%}")
    print(f"compliance_rate:   {compliance['compliance_rate']:.1%}")
    print(f"avg_word_count:    {compliance['avg_word_count']:.1f}")
    print(f"rate_too_long:     {compliance['rate_too_long']:.1%}")
    print(f"rate_hedge:        {compliance['rate_hedge']:.1%}")
    print(f"rate_explanation:  {compliance['rate_explanation']:.1%}")
    print(f"rate_bad_starter:  {compliance['rate_bad_starter']:.1%}")
    print(f"rate_refusal:      {compliance['rate_refusal']:.1%}")
    print(f"rate_sentence:     {compliance['rate_sentence']:.1%}")
    if compliance["non_compliant_examples"]:
        print("non-compliant examples (first 5):")
        for ex in compliance["non_compliant_examples"][:5]:
            print(f"  - {ex!r}")

    return {
        "label": label, "responses": responses, "correctness": correctness,
        "accuracy": accuracy, **compliance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of questions to test")
    parser.add_argument("--model", type=str, default=cfg.student_model_name,
                         help="which model to test the prompts on")
    parser.add_argument("--judge_backend", type=str, default=cfg.entailment_backend,
                         choices=["llm", "deberta"],
                         help="override cfg.entailment_backend for this run, "
                              "to compare judges without editing config.py")
    parser.add_argument("--judge_model", type=str, default=cfg.entailment_llm_model_name,
                        help="override cfg.entailment_llm_model_name for this run "
                            "(only used when --judge_backend=llm)")
    args = parser.parse_args()
    cfg.entailment_backend = args.judge_backend

    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    questions = [row["problem"] for row in ds][:args.n]
    answers   = [row["answer"]  for row in ds][:args.n]

    fewshot_examples = build_fewshot_examples(ds, n_shot=5, n_eval=args.n)
    print("Few-shot pool (from the tail of the dataset, disjoint from eval set):")
    for q, a in fewshot_examples:
        print(f"  Q: {q}\n  A: {a}")

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print(f"Judge backend: {cfg.entailment_backend}")
    if cfg.entailment_backend == "llm":
        print(f"Judge model: {args.judge_model}")
        judge_model, judge_tok = load_local_llm_judge(args.judge_model)
    else:
        from semantic_utils import load_nli_model
        judge_model, judge_tok = load_nli_model(cfg.nli_model_name)

    result_a = run_prompt_test(model, tokenizer, judge_model, judge_tok,
                                questions, answers,
                                lambda q: PROMPT_A_OLD.format(question=q),
                                "Prompt A (old, strict)")
    result_b = run_prompt_test(model, tokenizer, judge_model, judge_tok,
                                questions, answers,
                                lambda q: PROMPT_B_NEW.format(question=q),
                                "Prompt B (new, loose)")
    result_c = run_prompt_test(model, tokenizer, judge_model, judge_tok,
                                questions, answers,
                                lambda q: build_fewshot_prompt(fewshot_examples, q),
                                "Prompt C (few-shot, loose instruction + 5 examples)")
    print_mcnemar_comparisons({
        "A": result_a, "B": result_b, "C": result_c,
    })

    ts = time.strftime("%Y%m%d_%H%M%S")
    judge_tag = args.judge_model.split("/")[-1] if cfg.entailment_backend == "llm" else "deberta"
    out_path = f"figures/prompt_ab_{judge_tag}_{ts}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "gold",
            "response_A", "correct_A",
            "response_B", "correct_B",
            "response_C", "correct_C",
        ])
        for q, gold, ra, ca, rb, cb, rc, cc in zip(
            questions, answers,
            result_a["responses"], result_a["correctness"],
            result_b["responses"], result_b["correctness"],
            result_c["responses"], result_c["correctness"],
        ):
            writer.writerow([q, gold, ra, ca, rb, cb, rc, cc])
    print(f"\nWrote per-question results to {out_path}")


if __name__ == "__main__":
    main()