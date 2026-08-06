"""
analyze_sweep.py -- Unified four-part analysis across distillation configs:
target reproduction rate, corruption rate, entropy stats, AUROC (all &
excluding corrupted records). Prints one summary table per prompt style.

Usage:
  python analyze_sweep.py --tags gradacc8 warmup15 clip05 epochs5 epochs10 epochs20
"""
import argparse, json, re, statistics

GEN_DIR = "/users/ms25yt/SimpleQA/gen_data_distilled"
JUDGED_DIR = "/users/ms25yt/SimpleQA/judged_data_distilled"
TEACHER_DIR = "/users/ms25yt/SimpleQA/gen_data_subset500"

NUMERIC_HINTS = re.compile(r'\b(id|doi|how much|how many|population|number|votes?|price|salary|cc|mhz)\b', re.I)


def is_corrupt_field(text, question, long_threshold=300):
    if any((0x1F300 <= ord(c) <= 0x1FFFF) or (0x2600 <= ord(c) <= 0x27BF)
           or ord(c) in (0x2705, 0x274C, 0x274E, 0x2714, 0x2717) for c in text):
        return True
    # UUID rule waived for questions expecting numeric/ID answers
    if not NUMERIC_HINTS.search(question) and re.search(r'[0-9a-f]{8,}', text, re.I):
        return True
    if len(text) > long_threshold:
        return True
    return False


def record_is_corrupt(rec):
    q = rec.get("question", "")
    return sum(1 for t in rec.get("raw_responses", []) if is_corrupt_field(t, q)) >= 2


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, rank, i = 0.0, 1, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (2 * rank + (j - i) - 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def analyze(tag, style):
    gen_path = f"{GEN_DIR}/gen_simpleqa_simpleqa_Qwen3-4B-Instruct-2507_student_{style}_full_{tag}_{style}.jsonl"
    judged_path = f"{JUDGED_DIR}/judged_simpleqa_simpleqa_Qwen3-4B-Instruct-2507_student_{style}_full_{tag}_{style}.jsonl"
    teacher_path = f"{TEACHER_DIR}/gen_simpleqa_Qwen3-14B_{style}.jsonl"

    targets = {}
    for l in open(teacher_path):
        r = json.loads(l)
        targets[r["question_idx"]] = r["raw_responses"][0].strip()

    gen = [json.loads(l) for l in open(gen_path) if l.strip()]
    judged = [json.loads(l) for l in open(judged_path) if l.strip()]

    # 1. reproduction rate (low_temp vs training target)
    exact = near = 0
    for r in gen:
        t = targets.get(r["question_idx"])
        if t is None: continue
        s = r["low_temp_response"].strip()
        if s == t: exact += 1
        elif s.lower().rstrip('.') == t.lower().rstrip('.'): near += 1

    # 2. corruption (field- and record-level, on gen file)
    n_fields = bad_fields = 0
    bad_records = 0
    for r in gen:
        q = r.get("question", "")
        fields = [r["low_temp_response"]] + r["raw_responses"]
        bad = [is_corrupt_field(t, q) for t in fields]
        n_fields += len(fields); bad_fields += sum(bad)
        if sum(bad[1:]) >= 2:  # raw_responses only, >=2 threshold
            bad_records += 1

    # 3+4. entropy & AUROC (judged file)
    labels = [0 if r["is_correct"] else 1 for r in judged]
    scores = [r["semantic_entropy"] for r in judged]
    acc = 1 - sum(labels) / len(labels)
    ent_c = [s for s, l in zip(scores, labels) if l == 0]
    ent_w = [s for s, l in zip(scores, labels) if l == 1]
    corrupt_mask = [record_is_corrupt(r) for r in judged]
    clean = [(l, s) for l, s, c in zip(labels, scores, corrupt_mask) if not c]
    a_all = auroc(labels, scores)
    a_clean = auroc([l for l, _ in clean], [s for _, s in clean]) if clean else None

    return {
        "tag": tag, "n": len(gen),
        "repro": 100 * exact / len(gen), "near": 100 * near / len(gen),
        "field_pct": 100 * bad_fields / n_fields, "rec_pct": 100 * bad_records / len(gen),
        "acc": 100 * acc, "n_correct": len(ent_c),
        "ent_c": statistics.mean(ent_c) if ent_c else float("nan"),
        "ent_w": statistics.mean(ent_w) if ent_w else float("nan"),
        "auroc": a_all, "auroc_clean": a_clean,
        "n_clean": len(clean),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    for style in ["strict", "fewshot"]:
        print(f"\n{'='*105}")
        print(f"PROMPT STYLE: {style}")
        print(f"{'='*105}")
        hdr = (f"{'tag':<10} {'repro%':>7} {'near%':>6} {'fld%':>6} {'rec%':>6} "
               f"{'acc%':>5} {'nC':>4} {'entC':>6} {'entW':>6} {'AUROC':>7} {'AUROCcl':>8} {'nClean':>7}")
        print(hdr); print("-" * len(hdr))
        for tag in args.tags:
            try:
                r = analyze(tag, style)
            except FileNotFoundError as e:
                print(f"{tag:<10} MISSING: {e.filename}")
                continue
            print(f"{r['tag']:<10} {r['repro']:>7.1f} {r['near']:>6.1f} {r['field_pct']:>6.2f} {r['rec_pct']:>6.1f} "
                  f"{r['acc']:>5.1f} {r['n_correct']:>4} {r['ent_c']:>6.3f} {r['ent_w']:>6.3f} "
                  f"{r['auroc']:>7.4f} {(r['auroc_clean'] if r['auroc_clean'] else float('nan')):>8.4f} {r['n_clean']:>7}")


if __name__ == "__main__":
    main()