"""
check_alignment.py -- 验证 prepare_sft_dataset 的 label mask 对齐。
直接调用 distill.py 的真实函数，全量检查 500 条样本。
用法: python check_alignment.py [gen_file路径]
"""
import json
import sys

from transformers import AutoTokenizer

from config import cfg
from distill import build_dataset_from_filtered_questions, prepare_sft_dataset

GEN_FILE = sys.argv[1] if len(sys.argv) > 1 else \
    "gen_data_subset500/gen_simpleqa_Qwen3-14B_strict.jsonl"

tok = AutoTokenizer.from_pretrained(
    cfg.student_model_name, trust_remote_code=True)
tok.padding_side = "right"

# 和 filtered 模式一样: 读全部 question_idx, 目标是 raw_responses[0]
with open(GEN_FILE, encoding="utf-8") as f:
    all_idx = sorted(json.loads(l)["question_idx"] for l in f if l.strip())
data = build_dataset_from_filtered_questions(GEN_FILE, all_idx)
ds = prepare_sft_dataset(data, tok)

assert len(ds) == len(data), \
    f"样本数不一致: dataset={len(ds)} vs data={len(data)} (有样本被截断跳过?)"

bad_target, bad_prefix = [], []
for i in range(len(ds)):
    ex, rec = ds[i], data[i]

    # 检查1: 未被mask的目标 == response + <|im_end|>
    target_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    decoded = tok.decode(target_ids, skip_special_tokens=False)
    expected = rec["response"] + tok.eos_token
    if decoded != expected:
        bad_target.append((rec["question_idx"], decoded, expected))

    # 检查2: prompt前缀的token与单独tokenize的prompt逐位一致
    messages = [{"role": "user", "content": rec["prompt"]}]
    try:
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if ex["input_ids"][:len(prompt_ids)] != prompt_ids:
        bad_prefix.append(rec["question_idx"])

print(f"\n共检查 {len(ds)} 条样本")
print(f"目标不匹配 : {len(bad_target)} 条")
print(f"前缀不对齐 : {len(bad_prefix)} 条  {bad_prefix[:20]}")

for qidx, dec, exp in bad_target[:10]:
    print(f"\n--- question_idx={qidx} ---")
    print(f"  实际学习目标: {dec!r}")
    print(f"  期望目标    : {exp!r}")

# 肉眼抽查一条完整样本
ex, rec = ds[0], data[0]
print(f"\n===== 抽查 question_idx={rec['question_idx']} =====")
print("完整输入:")
print(repr(tok.decode(ex["input_ids"], skip_special_tokens=False)))
print("\n模型实际学习的目标(label != -100):")
print(repr(tok.decode(
    [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])))