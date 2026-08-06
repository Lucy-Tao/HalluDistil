"""
diagnose_judged.py -- Assess whether corruption in raw_responses
distorts semantic entropy and AUROC in judged files.

For each judged file: accuracy, entropy distribution split by
correctness, AUROC (all records), and AUROC excluding records whose
raw_responses trigger the corruption heuristics. A large gap between
the two AUROCs means conclusions are corruption-driven.

Usage:
  python diagnose_judged.py --files \
      judged_data/judged_simpleqa_Qwen3-14B_strict.jsonl \
      judged_data/judged_simpleqa_Qwen3-4B-Instruct-2507_strict.jsonl \
      judged_data_distilled/judged_simpleqa_simpleqa_Qwen3-4B-Instruct-2507_student_strict_full_gradacc8_strict.jsonl
"""
import argparse, json, re, statistics


def is_corrupt_field(text, long_threshold=300):
    if any((0x1F300 <= ord(c) <= 0x1FFFF) or (0x2600 <= ord(c) <= 0x27BF)
           or ord(c) in (0x2705, 0x274C, 0x274E, 0x2714, 0x2717) for c in text):
        return True
    if re.search(r'[0-9a-f]{8,}', text, re.I):
        return True
    if len(text) > long_threshold:
        return True
    return False


def record_is_corrupt(rec):
    fields = rec.get("raw_responses", [])
    n_bad = sum(1 for t in fields if is_corrupt_field(t))
    # >=2 corrupted samples out of 10 -> entropy likely distorted
    return n_bad >= 2


def auroc(labels, scores):
    """AUROC of scores predicting labels==1 (incorrect), rank-based."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, rank = 0.0, 1
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += (j - i)
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()

    for path in args.files:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        n = len(recs)
        # label 1 = incorrect (the thing entropy should detect)
        labels = [0 if r["is_correct"] else 1 for r in recs]
        scores = [r["semantic_entropy"] for r in recs]
        acc = 1 - sum(labels) / n

        ent_correct = [s for s, l in zip(scores, labels) if l == 0]
        ent_wrong = [s for s, l in zip(scores, labels) if l == 1]

        corrupt_mask = [record_is_corrupt(r) for r in recs]
        n_corrupt = sum(corrupt_mask)
        clean = [(l, s) for l, s, c in zip(labels, scores, corrupt_mask) if not c]

        a_all = auroc(labels, scores)
        a_clean = auroc([l for l, _ in clean], [s for _, s in clean]) if clean else None

        corrupt_wrong = sum(1 for l, c in zip(labels, corrupt_mask) if c and l == 1)

        print(f"\n===== {path} =====")
        print(f"  n={n}  accuracy={acc:.3f}")
        print(f"  entropy correct : mean={statistics.mean(ent_correct):.3f}  median={statistics.median(ent_correct):.3f}  (n={len(ent_correct)})")
        print(f"  entropy wrong   : mean={statistics.mean(ent_wrong):.3f}  median={statistics.median(ent_wrong):.3f}  (n={len(ent_wrong)})")
        print(f"  corrupted records (>=2 bad raw samples): {n_corrupt} ({100*n_corrupt/n:.1f}%), of which wrong: {corrupt_wrong}")
        print(f"  AUROC all           : {a_all:.4f}" if a_all else "  AUROC all: undefined")
        if a_clean is not None:
            print(f"  AUROC excl. corrupt : {a_clean:.4f}  (n={len(clean)})")
            print(f"  delta               : {a_clean - a_all:+.4f}")


if __name__ == "__main__":
    main()