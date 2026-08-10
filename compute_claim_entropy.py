"""
compute_claim_entropy.py -- Phase 4: compute long-form semantic entropy for
each atomic fact, following the naive long-form SE algorithm
(jlko/long_hallucinations).

INPUT: the output of run_factscore_eval.py, i.e. one entity per line with a
nested "claims" list of {atom, is_supported}. This script flattens that into
one record per atom (atom -> claim, is_supported -> is_true), then computes
semantic entropy per atom. Abstained/empty entities carry no claims and so
contribute nothing.

For each claim:
  1. Generate questions: call the question-gen prompt N_STOCHASTIC_QUESTIONS
     times (default 2), each asking for N_QUESTIONS (default 3) in
     "{question} -- {answer}" format => 6 questions, each with its own
     expected_answer.
  2. For each question: get_semantic_entropy() with fixed_response=
     expected_answer, n_samples=4 -- generates N_REGENERATE=3 new answers
     from the model under test, clusters with expected_answer via
     entailment, computes discrete semantic entropy over the 4-answer
     distribution.
  3. Refusal handling: if >= half of the 4 answers contain a refusal string,
     override that question's entropy to ln(4).
  4. Average the 6 per-question entropies for the claim's final entropy.

text_so_far (context for both prompts) is the claims that precede this one
(same question_idx, smaller claim_idx), concatenated -- matching jlko's
' '.join(propositions[:pidx]).

Entailment judge defaults to deberta (--judge_backend). Checkpointed per
(question_idx, claim_idx), safe to resume.

Usage
-----
  python compute_claim_entropy.py \
      --input gen_longform_data/factscore_Qwen3-32B.jsonl \
      --model_role teacher \
      --output gen_longform_data/entropy_Qwen3-32B.jsonl

  # distilled student: pass 'student' and point --model_name_override at
  # the checkpoint.
  python compute_claim_entropy.py \
      --input gen_longform_data/factscore_distilled_ep3.jsonl \
      --model_role student \
      --model_name_override /scratch-ssd/ms25yt/models/factscore_bio_distilled_student_ep3 \
      --output gen_longform_data/entropy_distilled_ep3.jsonl

Output: one jsonl line per claim:
  {question_idx, entity, claim_idx, claim, is_true,
   semantic_entropy, per_question_entropies}
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from config import cfg
from model_utils import load_model_and_tokenizer
from semantic_utils import (
    sample_responses, load_local_llm_judge, load_nli_model, get_semantic_entropy,
)

N_QUESTIONS = 3
N_STOCHASTIC_QUESTIONS = 1
N_REGENERATE = 5
N_SAMPLES_FOR_ENTROPY = N_REGENERATE + 1  # expected_answer + 3 regenerated
REFUSAL_STRINGS = ['not available', 'not provided', 'unknown', 'unclear']

_GEN_QUESTIONS_INSTRUCTION = (
    "Generate a list of {n} questions, that might have generated the "
    "sentence in the context of the preceding original text, as well as "
    "their answers. Please do not use specific facts that appear in the "
    "follow-up sentence when formulating the question.\n"
    "Make the questions and answers diverse. Avoid yes-no questions.\n"
    "The answers should not be a full sentence and as short as possible, "
    "e.g. only a name, place, or thing. Use the format "
    "\"1. {{question}} -- {{answer}}\""
)


def load_factscore_claims(path):
    """Read run_factscore_eval.py output (one entity per line, nested claims)
    and flatten to one record per atom, matching the fields the rest of this
    script expects: question_idx, claim_idx, entity, claim, is_true.
    Abstained/empty entities have an empty claims list and contribute nothing."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for claim_idx, atom in enumerate(r.get("claims", [])):
                records.append({
                    "question_idx": r["question_idx"],
                    "claim_idx": claim_idx,
                    "entity": r["entity"],
                    "claim": atom["atom"],
                    "is_true": atom["is_supported"],
                })
    return records


def build_gen_questions_prompt(proposition, text_so_far):
    instruction = _GEN_QUESTIONS_INSTRUCTION.format(n=N_QUESTIONS)
    if not text_so_far:
        return f"You see the sentence:\n\n{proposition}\n\n{instruction}"
    return (f"Following this text:\n\n{text_so_far}\n\n"
            f"You see the sentence:\n\n{proposition}\n\n{instruction}")


def build_answer_question_prompt(user_question, question, text_so_far):
    instruction = ('Please answer this question. Do not answer in a full '
                   'sentence. Answer with as few words as possible, e.g. '
                   'only a name, place, or thing.')
    if not text_so_far:
        return (f'We are writing an answer to the question '
                f'"{user_question}". First, we observe the following '
                f'question:\n\n{question}\n\n{instruction}')
    return (f'We are writing an answer to the question "{user_question}". '
            f'So far we have written:\n\n{text_so_far}\n\n'
            f'The next sentence should be the answer to the following '
            f'question:\n\n{question}\n\n{instruction}')


def parse_gen_questions(raw_output):
    pairs = []
    for line in raw_output.split('\n'):
        line = line.strip()
        if not line or ' -- ' not in line:
            continue
        q_part, a_part = line.split(' -- ', 1)
        q_part = q_part.strip()
        for i in range(1, N_QUESTIONS + 1):
            prefix = f"{i}."
            if q_part.startswith(prefix):
                q_part = q_part[len(prefix):].strip()
                break
        pairs.append((q_part, a_part.strip()))
    return pairs


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    # temperature=1.0: the two question-gen calls must produce DIFFERENT
    # question sets for coverage; at 0.1 they came back near-identical.
    # stop_sequences=None is required.
    return sample_responses(
        model, tokenizer, prompt,
        n_samples=1, temperature=temperature, max_new_tokens=max_new_tokens,
        stop_sequences=None,
    )[0]


def compute_entropy_for_claim(
    claim, entity, text_so_far,
    qgen_model, qgen_tokenizer,
    test_model, test_tokenizer,
    judge_model, judge_tokenizer,
    judge_backend="deberta",
    question_idx=None, claim_idx=None, qa_table=None,
    is_true=None,
):
    user_question = f"Tell me a bio of {entity}."
    all_questions = []  # (question, expected_answer)
    gen_prompt = build_gen_questions_prompt(claim, text_so_far)
    for call_idx in range(N_STOCHASTIC_QUESTIONS):
        raw = generate_text(qgen_model, qgen_tokenizer, gen_prompt, max_new_tokens=500)
        parsed = parse_gen_questions(raw)
        if len(parsed) != N_QUESTIONS:
            print(f"  WARNING: qgen call {call_idx + 1} for claim_idx={claim_idx} "
                  f"parsed {len(parsed)}/{N_QUESTIONS} questions.")
        all_questions.extend(parsed)

    per_question_entropies = []
    for question, expected_answer in all_questions:
        answer_prompt = build_answer_question_prompt(user_question, question, text_so_far)
        result = get_semantic_entropy(
            test_model, test_tokenizer, answer_prompt,
            judge_model, judge_tokenizer,
            n_samples=N_SAMPLES_FOR_ENTROPY,
            temperature=1.0,
            max_new_tokens=30,
            fixed_response=expected_answer,
            strict_entailment=False,
            judge_backend=judge_backend,
            question=question,
        )
        answers = result["raw_responses"]
        unknown_count = sum(
            any(sw in a.lower() for sw in REFUSAL_STRINGS) for a in answers
        )
        if unknown_count >= len(answers) // 2:
            entropy = -np.log(1 / len(answers))
        else:
            entropy = result["semantic_entropy"]
        per_question_entropies.append(float(entropy))

        if qa_table is not None:
            regen_answers = list(answers)
            regen_answers.remove(expected_answer)
            qa_table.add_data(
                question_idx, claim_idx, claim, is_true,
                gen_prompt, question, expected_answer,
                answer_prompt, " | ".join(regen_answers), float(entropy),
            )

    claim_entropy = float(np.mean(per_question_entropies)) if per_question_entropies else None
    return {
        "semantic_entropy": claim_entropy,
        "per_question_entropies": per_question_entropies,
    }


def load_done_keys(ckpt_path):
    done = set()
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["question_idx"], rec["claim_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def build_context_map(claims):
    by_qidx = defaultdict(list)
    for c in claims:
        by_qidx[c["question_idx"]].append(c)
    context_map = {}
    for qidx, group in by_qidx.items():
        group_sorted = sorted(group, key=lambda c: c["claim_idx"])
        running = []
        for c in group_sorted:
            text_so_far = " ".join(running) if running else None
            context_map[(qidx, c["claim_idx"])] = text_so_far
            running.append(c["claim"])
    return context_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="factscore_*.jsonl from run_factscore_eval.py")
    parser.add_argument("--model_role", required=True, choices=["teacher", "student"],
                        help="which local model to base the resampling model on. "
                             "For a distilled student, pass 'student' and use "
                             "--model_name_override.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--question_gen_model", default=None,
                        help="model for question generation "
                             "(default cfg.decomposition_model_name)")
    parser.add_argument("--model_name_override", default=None,
                        help="override the resampling model path, e.g. a "
                             "distilled checkpoint")
    parser.add_argument("--wandb_project", default=None,
                        help="if given, log a live (claim, question, answers, "
                             "entropy) table to this wandb project")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--test_model_device", type=int, default=None,
                        help="GPU index to pin the model-under-test to. "
                             "Pass explicitly on 2+ GPU runs to avoid "
                             "layer-splitting/CPU-offload.")
    parser.add_argument("--qgen_judge_device", type=int, default=None,
                        help="GPU index for the question-gen model (and the "
                             "llm judge, when reused). deberta judge is also "
                             "moved here if given.")
    parser.add_argument("--judge_backend", choices=["llm", "deberta"], default="deberta",
                        help="entailment judge backend (see semantic_utils.py)")
    args = parser.parse_args()

    claims = load_factscore_claims(args.input)
    print(f"Loaded {len(claims)} claims (flattened) from {args.input}")
    context_map = build_context_map(claims)

    done_keys = load_done_keys(args.output)
    todo = [c for c in claims if (c["question_idx"], c["claim_idx"]) not in done_keys]
    print(f"Checkpoint: {len(done_keys)} claims done, {len(todo)} remaining.")
    if not todo:
        print("Nothing to do.")
        return

    qa_table = None
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "input": args.input,
                "model_role": args.model_role,
                "model_name_override": args.model_name_override,
                "n_questions": N_QUESTIONS,
                "n_stochastic_questions": N_STOCHASTIC_QUESTIONS,
                "n_regenerate": N_REGENERATE,
            },
        )
        qa_table = wandb.Table(columns=[
            "question_idx", "claim_idx", "claim", "is_true",
            "qgen_prompt", "generated_question", "expected_answer",
            "answer_prompt", "regenerated_answers", "entropy",
        ])
        print(f"wandb enabled -- project={args.wandb_project!r}, run={wandb.run.name!r}")

    default_test_model_name = (cfg.teacher_model_name if args.model_role == "teacher"
                               else cfg.student_model_name)
    test_model_name = args.model_name_override or default_test_model_name
    qgen_model_name = args.question_gen_model or cfg.decomposition_model_name

    # Explicit single-GPU pinning ({"": "cuda:N"}) keeps the model-under-test
    # and the qgen/judge model on separate whole GPUs. Falls back to
    # cfg.device_map ("auto") on single-GPU runs.
    test_device_map = (
        {"": f"cuda:{args.test_model_device}"}
        if args.test_model_device is not None else cfg.device_map
    )
    qgen_judge_device_map = (
        {"": f"cuda:{args.qgen_judge_device}"}
        if args.qgen_judge_device is not None else cfg.device_map
    )

    print(f"Loading model under test ({args.model_role}"
          f"{' [override]' if args.model_name_override else ''}): {test_model_name}")
    # Pre-bind so that a failure inside load_model_and_tokenizer surfaces
    # as itself rather than as an UnboundLocalError from the `del` at the
    # end of the function, which hid the real cause on several runs.
    test_model = test_tokenizer = None
    test_model, test_tokenizer = load_model_and_tokenizer(test_model_name, device_map=test_device_map)

    if qgen_model_name == test_model_name:
        qgen_model, qgen_tokenizer = test_model, test_tokenizer
        print("Question-gen model matches model under test -- reusing.")
    else:
        print(f"Loading question-gen model: {qgen_model_name}")
        qgen_model, qgen_tokenizer = load_model_and_tokenizer(qgen_model_name, device_map=qgen_judge_device_map)

    if args.judge_backend == "deberta":
        print(f"Loading entailment judge (deberta): {cfg.nli_model_name}")
        judge_model, judge_tokenizer = load_nli_model(cfg.nli_model_name)
        if args.qgen_judge_device is not None:
            judge_model = judge_model.to(f"cuda:{args.qgen_judge_device}")
    else:
        judge_model_name = cfg.entailment_llm_model_name
        if judge_model_name == test_model_name:
            judge_model, judge_tokenizer = test_model, test_tokenizer
            print("Entailment judge matches model under test -- reusing.")
        elif judge_model_name == qgen_model_name:
            judge_model, judge_tokenizer = qgen_model, qgen_tokenizer
            print("Entailment judge matches question-gen model -- reusing.")
        else:
            print(f"Loading entailment judge (llm): {judge_model_name}")
            judge_model, judge_tokenizer = load_local_llm_judge(judge_model_name)

    with open(args.output, "a", encoding="utf-8") as ckpt_f:
        logged_count = 0
        for c in todo:
            question_idx, claim_idx, entity = c["question_idx"], c["claim_idx"], c["entity"]
            text_so_far = context_map.get((question_idx, claim_idx))
            print(f"[{question_idx}.{claim_idx}] {entity}: {c['claim'][:60]!r}")
            result = compute_entropy_for_claim(
                c["claim"], entity, text_so_far,
                qgen_model, qgen_tokenizer,
                test_model, test_tokenizer,
                judge_model, judge_tokenizer,
                judge_backend=args.judge_backend,
                question_idx=question_idx, claim_idx=claim_idx, qa_table=qa_table,
                is_true=c["is_true"],
            )
            logged_count += 1
            # Log the (cumulative) table only every 500 claims. wandb.Table
            # is snapshot-semantics: each log stores the ENTIRE accumulated
            # table, so logging every claim writes 1+2+...+N rows total and
            # bloats to tens of GB. Every-500 cuts that ~500x while still
            # giving a live (if coarser) view.
            if qa_table is not None and logged_count % 500 == 0:
                import wandb
                wandb.log({"qa_samples": wandb.Table(
                    columns=qa_table.columns, data=qa_table.data
                )})

            record = {
                "question_idx": question_idx,
                "entity": entity,
                "claim_idx": claim_idx,
                "claim": c["claim"],
                "is_true": c["is_true"],
                "semantic_entropy": result["semantic_entropy"],
                "per_question_entropies": result["per_question_entropies"],
            }
            ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            se = result["semantic_entropy"]
            print(f"  -> semantic_entropy = {se:.4f}" if se is not None else "  -> semantic_entropy = None")

    del test_model
    if qgen_model is not test_model:
        del qgen_model
    if judge_model not in (test_model, qgen_model):
        del judge_model
    torch.cuda.empty_cache()
    if qa_table is not None:
        import wandb
        wandb.finish()
    print(f"\nDone. Wrote to {args.output}")


if __name__ == "__main__":
    main()