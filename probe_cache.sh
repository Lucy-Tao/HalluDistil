#!/bin/bash
#SBATCH --partition=msc
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=probe_cache
#SBATCH --output=logs/probe_cache_%A_%a.out

NODES=(oat10 oat11 oat13 oat14 oat15 oat16 oat17 oat18)
H=/scratch-ssd/oatml/huggingface/hub

echo "node=$(hostname -s)"
df -h /scratch-ssd | tail -1 | awk '{print "  free="$4"  used="$5}'
for m in Qwen--Qwen3-32B Qwen--Qwen3-4B-Instruct-2507 microsoft--deberta-v2-xlarge-mnli; do
    d="$H/models--$m"
    if [ ! -d "$d" ]; then
        echo "  $m ABSENT"
        continue
    fi
    best=0
    for s in "$d"/snapshots/*/; do
        n=$(ls "$s" 2>/dev/null | grep -c 'safetensors$\|\.bin$')
        [ "$n" -gt "$best" ] && best=$n
    done
    echo "  $m shards=$best size=$(du -shL "$d" 2>/dev/null | cut -f1)"
done
