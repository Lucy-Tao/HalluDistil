#!/bin/bash
hostname
source ~/.bashrc
conda activate haldist
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=gpu_bus_id,memory.free --format=csv,noheader
python -u << 'PYEOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

m = "meta-llama/Meta-Llama-3.1-70B-Instruct"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map="auto")
print("device_map:", model.hf_device_map)

msgs = [{"role": "user", "content": "Who wrote Pride and Prejudice? Answer in one short phrase."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    lg = model(**inp).logits
print("logits finite:", torch.isfinite(lg).all().item(),
      "min", lg.min().item(), "max", lg.max().item())

n = inp["input_ids"].shape[1]
g = model.generate(**inp, max_new_tokens=20, do_sample=False)
print("greedy :", repr(tok.decode(g[0][n:], skip_special_tokens=True)))

g2 = model.generate(**inp, max_new_tokens=20, do_sample=True,
                    temperature=1.0, top_p=0.9)
print("sampled:", repr(tok.decode(g2[0][n:], skip_special_tokens=True)))

g3 = model.generate(**inp, max_new_tokens=20, do_sample=True,
                    temperature=1.0, top_p=0.9, renormalize_logits=True)
print("renorm :", repr(tok.decode(g3[0][n:], skip_special_tokens=True)))
PYEOF
