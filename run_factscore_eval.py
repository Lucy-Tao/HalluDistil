"""
run_factscore_eval.py -- long-form (FActScore Bio) correctness scoring for
one model's generations, with abstention handling by BLANKING (option B).

Pipeline
--------
  1. Load the model's generation jsonl (kept as-is on disk; original refusal
     text is never overwritten).
  2. For each entity, apply the SHARED abstention rule (abstention_rule.py).
     If abstained, replace the response with "" IN MEMORY ONLY before
     scoring. An empty string yields zero atomic facts downstream, so an
     abstained entity contributes nothing to the claim pool / AUROC.
  3. Report the abstention rate separately.
  4. Run FActScore on the (possibly blanked) generations and write out the
     per-claim decisions.

The original jsonl is untouched, so edge-case refusals (e.g. Ronaldo,
Gonzalo Fonseca) remain available for qualitative inspection.

Requires the factscorer.py edit that makes it skip only truly EMPTY strings
(replace `or is_non_answer(gen)` with `or (not gen or not gen.strip())`),
so that the abstention decision is owned solely by abstention_rule.py.

Usage
-----
  python run_factscore_eval.py \\
      --gen ~/gen_longform/gen_factscore_bio_Qwen3-32B.jsonl \\
      --out ~/SimpleQA/gen_longform_data/factscore_Qwen3-32B.jsonl
"""
import argparse
import json
import os
import sys

# shared abstention rule -- same file used by compute_abstention_rate.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstention_rule import is_abstained  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True,
                    help="generation jsonl (from generate_longform_responses.py)")
    ap.add_argument("--out", required=True,
                    help="where to write per-claim decisions jsonl")
    ap.add_argument("--factscore_dir",
                    default=os.path.expanduser("~/SimpleQA/third_eval/FActScore"),
                    help="path to the vendored FActScore package root")
    ap.add_argument("--data_dir",
                    default=os.path.expanduser("~/SimpleQA/factscore_data"))
    ap.add_argument("--cache_dir",
                    default=os.path.expanduser("~/SimpleQA/factscore_data/cache"))
    ap.add_argument("--db_path",
                    default=os.path.expanduser(
                        "~/SimpleQA/factscore_data/enwiki-20230401-subset.db"))
    ap.add_argument("--knowledge_name", default="enwiki-20230401")
    ap.add_argument("--gamma", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, args.factscore_dir)
    from factscore.factscorer import FactScorer  # noqa: E402

    # ---- load generations; blank abstained ones in memory only ----
    topics, generations, qidx_list = [], [], []
    abstained_topics = []
    with open(args.gen, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            entity = r["entity"]
            resp = r.get("response", "")
            qidx = r.get("question_idx")
            topics.append(entity)
            qidx_list.append(qidx)
            if is_abstained(resp):
                abstained_topics.append((qidx, entity))
                generations.append("")          # blank -> zero claims downstream
            else:
                generations.append(resp)

    n_total = len(topics)
    n_abstained = len(abstained_topics)
    print(f"Loaded {n_total} entities from {args.gen}")
    print(f"Abstained (blanked to ''): {n_abstained}/{n_total} "
          f"({100 * n_abstained / n_total:.1f}%)")
    for qi, ent in abstained_topics:
        print(f"    abstained: question_idx={qi}  {ent}")

    # ---- FActScore ----
    fs = FactScorer(
        model_name="retrieval+ChatGPT",
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        abstain_detection_type=None,   # official detector off; blanking is ours
    )
    fs.register_knowledge_source(
        args.knowledge_name,
        db_path=args.db_path,
        data_path=None,
    )

    out = fs.get_score(topics, generations, gamma=args.gamma)
    print(f"\nFActScore (over scored entities): {out['score']:.4f}")

    # ---- write per-claim decisions ----
    n_claims = 0
    n_scored_entities = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for qidx, topic, decision in zip(qidx_list, topics, out["decisions"]):
            if decision is None:
                # abstained or empty -> no claims; record the skip explicitly
                f.write(json.dumps({
                    "question_idx": qidx,
                    "entity": topic,
                    "abstained_or_empty": True,
                    "claims": [],
                }, ensure_ascii=False) + "\n")
                continue
            n_scored_entities += 1
            claims = [{"atom": d["atom"], "is_supported": bool(d["is_supported"])}
                      for d in decision]
            n_claims += len(claims)
            f.write(json.dumps({
                "question_idx": qidx,
                "entity": topic,
                "abstained_or_empty": False,
                "claims": claims,
            }, ensure_ascii=False) + "\n")

    print(f"Scored entities: {n_scored_entities}/{n_total}")
    print(f"Total claims: {n_claims}")
    print(f"Wrote per-claim decisions to {args.out}")


if __name__ == "__main__":
    main()