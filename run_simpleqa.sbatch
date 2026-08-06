#!/bin/bash
#SBATCH --job-name=simpleqa_eval
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# 用法: sbatch run_simpleqa.sbatch <模型HF路径> <端口> <tensor_parallel_size>
MODEL=${1:?"用法: sbatch run_simpleqa.sbatch <model> <port> <tp_size>"}
PORT=${2:-8001}
TP_SIZE=${3:-1}
SAFE_NAME=$(echo "$MODEL" | tr '/' '_')

OUTPUT_DIR=~/SimpleQA/results
mkdir -p "$OUTPUT_DIR"

source /scratch-ssd/oatml/miniconda3/etc/profile.d/conda.sh
conda activate haldist-vllm
export OPENAI_API_KEY=$(cat ~/.openai_key)

cd ~/SimpleQA

echo "启动 vllm server: $MODEL (port=$PORT, tp=$TP_SIZE)"
vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --reasoning-parser qwen3 --enforce-eager \
  > "${OUTPUT_DIR}/${SAFE_NAME}_vllm.log" 2>&1 &
VLLM_PID=$!

echo "等待server就绪..."
for i in $(seq 1 120); do
  if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "server已就绪"
    break
  fi
  sleep 5
done

python simpleqa_runner.py \
  --model "$MODEL" \
  --port "$PORT" \
  --output "${OUTPUT_DIR}/${SAFE_NAME}_results.jsonl" \
  --max-workers 16

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "全部完成"
