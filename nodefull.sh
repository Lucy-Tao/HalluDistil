#!/bin/bash
hostname
echo "--- disk"
df -h /scratch-ssd | tail -1
echo "--- group cache Qwen3"
G=/scratch-ssd/oatml/huggingface/hub
for m in Qwen3-32B Qwen3-4B-Instruct-2507; do
  p=$G/models--Qwen--$m
  if [ -d "$p" ]; then
    echo "$(du -sh $p | cut -f1)  $(find $p/snapshots -name '*.safetensors' 2>/dev/null | wc -l) shards  $m"
  else
    echo "MISSING  $m"
  fi
done
echo "--- personal cache"
ls -d /scratch-ssd/ms25yt/hf/hub/models--* 2>/dev/null || echo "none"
echo "--- locks"
ls -ld $G/.locks 2>/dev/null
touch $G/.locks/testwrite_$$ 2>/dev/null && echo "LOCKS WRITABLE" && rm -f $G/.locks/testwrite_$$ || echo "LOCKS NOT WRITABLE"
echo "--- offline load"
source ~/.bashrc
conda activate haldist
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c "
from transformers import AutoTokenizer, AutoConfig
for m in ['Qwen/Qwen3-32B', 'Qwen/Qwen3-4B-Instruct-2507']:
    try:
        c = AutoConfig.from_pretrained(m)
        t = AutoTokenizer.from_pretrained(m)
        print('OK', m, c.num_hidden_layers, 'layers')
    except Exception as e:
        print('FAIL', m, type(e).__name__, str(e)[:120])
"
