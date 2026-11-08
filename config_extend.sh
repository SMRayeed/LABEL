# Settings for the experiments added in November 2026. Sourced by
# run_all.sh right after config.sh, so config.sh stays as you edited it.
# Every value can also be set in the environment.

# ---------------------------------------------------------------- extensions
# Added for the 8-page version. `bash run_all.sh --only extend` fetches what
# these need; `--only models` then submits only the jobs not yet done.

# Scale curve: more Qwen2.5 sizes on the same closed-book probe.
SCALE_MODELS="${SCALE_MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-14B-Instruct}"
TIME_XL="${TIME_XL:-12:00:00}"           # 14B-class jobs on CPU; resubmit to resume

# Optional GPU. Models listed here run on a GPU node in bfloat16 instead of
# on CPU in float32, e.g. GPU_MODELS="Qwen/Qwen2.5-14B-Instruct".
# The GPU jobs use their own environment with a CUDA build of torch.
GPU_MODELS="${GPU_MODELS:-}"
GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_ENV_DIR="${GPU_ENV_DIR:-$HOME/gpu_probe_env}"
GPU_TORCH_INDEX="${GPU_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
TIME_GPU="${TIME_GPU:-08:00:00}"

# FDA labels from openFDA. A key is optional (without one: 1,000 requests a day).
OPENFDA_API_KEY="${OPENFDA_API_KEY:-}"
export OPENFDA_API_KEY
