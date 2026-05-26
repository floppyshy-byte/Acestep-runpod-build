# =============================================================================
# ACE-Step 1.5 — RunPod Serverless Worker (Slim)
# =============================================================================
# No model weights baked in. We rely on RunPod's cached model storage
#
# To use RunPod cached models, add this HF repo ID when creating the endpoint:
#   Floppyshy/Ace-Step1.5-Custom
#
# Layer ordering (large / stable -> small / volatile):
#   1. Base image
#   2. System runtime libraries
#   3. Copy uv + /app (source code, .venv)
#   4. Copy handler.py  <-- changes often; kept last for cache friendliness
#   5. CMD
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 0: uv sync needs
# ---------------------------------------------------------------------------
FROM alpine:latest AS fetcher
WORKDIR /app

ARG ACESTEP_COMMIT=6adf5f1382096d757de11ce20afc86ba746e2100
ADD https://raw.githubusercontent.com/floppyshy-byte/ACE-Step-1.5/${ACESTEP_COMMIT}/pyproject.toml /app/pyproject.toml
ADD https://raw.githubusercontent.com/floppyshy-byte/ACE-Step-1.5/${ACESTEP_COMMIT}/uv.lock /app/uv.lock

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile Python deps only
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.2.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg libsndfile1 \
    python3.11 python3.11-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY --from=fetcher /app /app

WORKDIR /app

ARG ACESTEP_COMMIT=6adf5f1382096d757de11ce20afc86ba746e2100
RUN git clone https://github.com/floppyshy-byte/ACE-Step-1.5.git /tmp/ace-step \
    && cd /tmp/ace-step && git checkout ${ACESTEP_COMMIT} \
    && rm -rf /tmp/ace-step/.git \
    && cp -a /tmp/ace-step/. /app/ \
    && rm -rf /tmp/ace-step
RUN uv sync
RUN uv pip install runpod accelerate

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what's needed to run inference
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.2.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    python3.11 python3.11-venv python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# Environment defaults (override at runtime in RunPod console).
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
ENV ACESTEP_MAIN_MODEL_REPO=Floppyshy/Ace-Step1.5-Custom
ENV ACESTEP_MAIN_MODEL_COMPONENTS=acestep-v15-xl-turbo,vae,Qwen3-Embedding-0.6B,acestep-5Hz-lm-4B
ENV ACESTEP_CHECKPOINTS_DIR=/runpod-volume/checkpoints
ENV HF_HOME=/runpod-volume/huggingface-cache/hub
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub
ENV ACESTEP_DISABLE_DOWNLOAD=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV ACESTEP_LM_BACKEND=pt
ENV ACESTEP_VLLM_ENFORCE_EAGER=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1
ENV PYTHONFAULTHANDLER=1
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV PYTORCH_CUDA_ALLOC_CONF=backend:native

# Copy setup helper and handler LAST so code changes don't invalidate dep layers.
COPY setup_models.py /app/setup_models.py
COPY handler.py /app/handler.py
COPY gpu_config.py /app/acestep/gpu_config.py
COPY audio_utils.py /app/acestep/audio_utils.py
COPY llm_inference.py /app/acestep/llm_inference.py

CMD ["python3", "-X", "faulthandler", "-u", "/app/handler.py"]
