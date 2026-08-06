"""
judge_responses.py — Phase 2 of the decoupled pipeline: read a
generate_responses.py output file (raw responses only, no judging) and
produce a fully-judged record for each question:
  - grade correctness of the low_temp_response (CORRECT/INCORRECT/NOT_ATTEMPTED)
  - semantic_entropy / clusters: computed from the 10 pure high-temperature
    raw_responses via cluster_by_entailment() + compute_semantic_distribution().

    NOTE: unlike the older filter_questions.py scan_model(), which folded
    the low-temp sample into the same pool used for entropy (n_high_temp+1
    total, via get_semantic_entropy's fixed_response mechanism), this
    keeps low_temp_response and raw_responses conceptually and numerically
    separate — entropy is computed from the 10 raw_responses ONLY. This
    matches generate_responses.py's Phase 1 design (see conversation
    history for why merging them biases entropy downward: the low-temp
    sample is disproportionately likely to land in the modal cluster).

Loads ONE judge model for the whole run (cfg.entailment_backend /
cfg.entailment_llm_model_name — set via config.py or the CLI overrides
below). Checkpointed and resumable, same convention as
generate_responses.py.

Output per record: question_idx, question, prompt, gold_answer,
low_temp_response, grade, semantic_entropy, raw_responses, and
clusters — the raw index-based clustering (list of lists, e.g.
[[0,1,3],[2],[4,5,6,7,8,9]]) straight from cluster_by_entailment(),
showing exactly which raw_responses indices were judged to be the same
answer. (compute_semantic_distribution() also returns cluster_probs /
cluster_members / predicted_response, but those are intentionally
dropped here — not needed downstream, and easy to recompute from
`clusters` + `raw_responses` later if that changes.)

Usage:
    python judge_responses.py \
        --input ~/SimpleQA/gen_data/gen_simpleqa_Qwen3-14B_strict.jsonl \
        --output_dir ~/SimpleQA/judged_data \
        --entailment_backend llm \
        --entailment_llm_model_name Qwen/Qwen2.5-32B-Instruct \
        --strict_entailment
"""
import argparse
import json
import os
import re

from config import cfg
from semantic_utils import (
    cluster_by_entailment,
    compute_semantic_distribution,
    QwenGrader,
    GptGrader,
    load_local_llm_judge,
    load_nli_model,
)


def load_judge():
    """Load whichever entailment judge cfg.entailment_backend points to."""
    if cfg.entailment_backend == "llm":
        return load_local_llm_judge(cfg.entailment_llm_model_name)
    if cfg.entailment_backend == "deberta":
        return load_nli_model(cfg.nli_model_name)
    raise ValueError(f"Unknown cfg.entailment_backend={cfg.entailment_backend!r}")


def load_done_indices(ckpt_path: str) -> set[int]:
    """Same convention as generate_responses.py: read an existing output
    file (if any) and return the set of question_idx already judged, so a
    resumed run skips them."""
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
                done.add(rec["question_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _shorten(base: str) -> str:
    """Strip duplicated segments from distilled-checkpoint filenames.

    Distilled models reach generate_responses.py as a filesystem path whose
    basename already encodes dataset, role and prompt_style, so the gen
    filename repeats them. Plain HF model ids match neither pattern and pass
    through unchanged.
    """
    base = re.sub(r"^(judged_)([a-z0-9]+)_\2_", r"\1\2_", base)
    base = re.sub(r"_student_(strict|fewshot)_", "_", base)
    return base


def derive_output_path(input_path: str, output_dir: str) -> str:
    """gen_simpleqa_Qwen3-14B_strict.jsonl -> judged_simpleqa_Qwen3-14B_strict.jsonl"""
    base = os.path.basename(input_path)
    if base.startswith("gen_"):
        base = "judged_" + base[len("gen_"):]
    else:
        base = "judged_" + base
    return os.path.join(output_dir, _shorten(base))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True,
                         help="a generate_responses.py output .jsonl file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--entailment_backend", type=str, default=None,
                         choices=["llm", "deberta"],
                         help="override cfg.entailment_backend for this run")
    parser.add_argument("--entailment_llm_model_name", type=str, default=None,
                         help="override cfg.entailment_llm_model_name for this run "
                              "(only used when entailment_backend='llm')")
    parser.add_argument("--strict_entailment", action="store_true", default=None,
                         help="use strict bidirectional entailment (both directions "
                              "must classify as entailment) — default matches cfg.strict_entailment")
    parser.add_argument("--no_strict_entailment", dest="strict_entailment",
                         action="store_false")
    parser.add_argument("--grader", type=str, default="qwen",
                         choices=["qwen", "gpt"],
                         help="which LLM grades correctness (SimpleQA grader). "
                              "Always an LLM, independent of entailment_backend.")
    parser.add_argument("--grader_model", type=str, default=None,
                         help="grader model id. qwen: HF model id (default = "
                              "entailment_llm_model_name if that's an LLM, else must be set); "
                              "gpt: API model id e.g. gpt-4.1.")
    args = parser.parse_args()

    if args.entailment_backend is not None:
        cfg.entailment_backend = args.entailment_backend
    if args.entailment_llm_model_name is not None:
        cfg.entailment_llm_model_name = args.entailment_llm_model_name
    strict_entailment = (
        cfg.strict_entailment if args.strict_entailment is None else args.strict_entailment
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = derive_output_path(args.input, args.output_dir)

    print(f"{'='*60}")
    print(f"PHASE 2: JUDGING (correctness + semantic entropy)")
    print(f"  input:               {args.input}")
    print(f"  output:              {output_path}")
    print(f"  entailment_backend:  {cfg.entailment_backend}")
    if cfg.entailment_backend == "llm":
        print(f"  judge model:         {cfg.entailment_llm_model_name}")
    print(f"  strict_entailment:   {strict_entailment}")
    print(f"{'='*60}\n")

    with open(args.input, "r", encoding="utf-8") as f:
        input_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(input_records)} generated record(s) from input.")

    done_indices = load_done_indices(output_path)
    remaining = [r for r in input_records if r["question_idx"] not in done_indices]
    print(f"Checkpoint: {len(done_indices)} question(s) already judged, "
          f"{len(remaining)} remaining.")

    if not remaining:
        print("Nothing to do — all questions already judged.")
        return

    print("Loading cluster judge...")
    cluster_model, cluster_tok = load_judge()  

    print(f"Loading grader ({args.grader})...")
    if args.grader == "gpt":
        grader = GptGrader(args.grader_model or "gpt-4.1")
    else:  # qwen
        gname = args.grader_model or cfg.entailment_llm_model_name
        if cfg.entailment_backend == "llm" and gname == cfg.entailment_llm_model_name:
            grader = QwenGrader(cluster_model, cluster_tok)
        else:
            gm, gt = load_local_llm_judge(gname)
            grader = QwenGrader(gm, gt)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for rec in remaining:
            question       = rec["question"]
            gold           = rec["answer"]
            low_temp       = rec["low_temp_response"]
            raw_responses  = rec["raw_responses"]

            grade = grader.grade(question, gold, low_temp)   # CORRECT/INCORRECT/NOT_ATTEMPTED

            clusters = cluster_by_entailment(
                raw_responses, cluster_model, cluster_tok, strict_entailment,
                backend=cfg.entailment_backend, question=question,
            )
            dist = compute_semantic_distribution(raw_responses, clusters)

            out_rec = {
                "question_idx":       rec["question_idx"],
                "question":           question,
                "prompt":             rec["prompt"],
                "gold_answer":        gold,
                "low_temp_response":  low_temp,
                "grade":              grade,
                "semantic_entropy":   dist["semantic_entropy"],
                "raw_responses":      raw_responses,
                "clusters":           clusters,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"[{rec['question_idx']}] grade={grade}  "
                  f"entropy={dist['semantic_entropy']:.3f}")

    print(f"\nDone. Wrote to {output_path}")


if __name__ == "__main__":
    main()