"""
diagnose_teacher_generation.py — isolate the real cause of teacher
(Qwen3-14B) generation crashing with "probability tensor contains inf,
nan or element < 0", by precisely replicating run_scan()'s actual loading
ORDER: judge (Qwen2.5-32B-Instruct) loaded FIRST via load_judge(), THEN
teacher loaded SECOND via device_map='auto' — exactly what scan_teacher.sh
does.

Usage:
    python diagnose_teacher_generation.py
"""
import gc

import torch

from config import cfg
from model_utils import load_model_and_tokenizer
from semantic_utils import sample_responses, DEFAULT_STOP_SEQUENCES
from filter_questions import load_judge

TEST_PROMPT = (
    "Question: Who received the IEEE Frank Rosenblatt Award in 2010?\n"
    "Answer the question with only the minimal factual answer string.\n"
    "Do not write a full sentence.\n"
    "Do not include explanations, context, hedging, or punctuation.\n"
    "Do not start with phrases like 'The answer is' or 'It is'.\n"
    "Use the most common valid form of the answer.\n"
    "Answer:"
)


def try_generate(model, tokenizer, label):
    print(f"\n--- {label} ---")
    try:
        r = sample_responses(
            model, tokenizer, TEST_PROMPT,
            n_samples=1, temperature=0.1,
            stop_sequences=DEFAULT_STOP_SEQUENCES,
        )
        print(f"  SUCCESS: {r[0]!r}")
        return True
    except RuntimeError as e:
        print(f"  CRASHED: {type(e).__name__}: {e}")
        return False


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    results = {}

    print("\n" + "=" * 60 + "\n(A) TEACHER ONLY\n" + "=" * 60)
    print(f"Loading teacher: {cfg.teacher_model_name}...")
    model, tokenizer = load_model_and_tokenizer(cfg.teacher_model_name)
    results["A_teacher_only"] = try_generate(model, tokenizer, "(A) teacher only")
    del model, tokenizer
    free_gpu()

    print("\n" + "=" * 60 + "\n(B) JUDGE FIRST, THEN TEACHER (matches scan_model() exactly)\n" + "=" * 60)
    print(f"Loading judge: {cfg.entailment_llm_model_name}...")
    judge_model, judge_tok = load_judge()
    print(f"Loading teacher: {cfg.teacher_model_name}...")
    model, tokenizer = load_model_and_tokenizer(cfg.teacher_model_name)
    results["B_judge_then_teacher"] = try_generate(model, tokenizer, "(B) judge-then-teacher")
    del model, tokenizer, judge_model, judge_tok
    free_gpu()

    print("\n" + "=" * 60 + "\n(C) TEACHER FIRST, THEN JUDGE (reverse order)\n" + "=" * 60)
    print(f"Loading teacher: {cfg.teacher_model_name}...")
    model, tokenizer = load_model_and_tokenizer(cfg.teacher_model_name)
    print(f"Loading judge: {cfg.entailment_llm_model_name}...")
    judge_model, judge_tok = load_judge()
    results["C_teacher_then_judge"] = try_generate(model, tokenizer, "(C) teacher-then-judge")
    del model, tokenizer, judge_model, judge_tok
    free_gpu()

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'SUCCESS' if v else 'CRASHED'}")

    if results["A_teacher_only"] and not results["B_judge_then_teacher"]:
        print("\nCONCLUSION: loading order matters. Judge-before-teacher "
              "(the real scan_model() order) reproduces the crash.")
        if results["C_teacher_then_judge"]:
            print("Loading teacher FIRST avoids the crash — confirms it's "
                  "specifically about what's already resident in GPU "
                  "memory when device_map='auto' places the teacher.")
    elif not results["A_teacher_only"]:
        print("\nCONCLUSION: even teacher-alone crashed this time — the "
              "issue may be intermittent (sampling randomness) rather "
              "than deterministically tied to loading order.")
    else:
        print("\nCONCLUSION: none of these reproduced the crash here.")


if __name__ == "__main__":
    main()