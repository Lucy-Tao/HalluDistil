export HOME=/scratch-ssd/ms25yt/fakehome
mkdir -p $HOME/.cache
export PATH=/users/ms25yt/.conda/envs/vllm70b/bin:$PATH
export HF_HOME=/scratch-ssd/oatml/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch-ssd/ms25yt/datasets
export VLLM_CACHE_ROOT=/scratch-ssd/ms25yt/cache/vllm
export XDG_CACHE_HOME=/scratch-ssd/ms25yt/cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
