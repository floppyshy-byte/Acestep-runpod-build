# =============================================================================
# ACE-Step 1.5 — RunPod Serverless Worker (Slim)
# =============================================================================
# No model weights baked in. We rely on RunPod's cached model storage or
# let ACE-Step download on first cold start. Image is ~2-3 GB vs ~15 GB.
#
# To use RunPod cached models, add these HF repo IDs when creating the endpoint:
#   ACE-Step/Ace-Step1.5
#   ACE-Step/acestep-v15-sft
#
# Layer ordering (large / stable -> small / volatile):
#   1. Base image
#   2. System runtime libraries
#   3. Copy uv + /app (source code, .venv)
#   4. Copy handler.py  <-- changes often; kept last for cache friendliness
#   5. CMD
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile Python deps only
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg libsndfile1 \
    python3.11 python3.11-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
ARG CACHEBUST=1
RUN git clone --depth 1 https://github.com/floppyshy-byte/ACE-Step-1.5.git . && echo "cache-bust: ${CACHEBUST}"

RUN uv sync
RUN uv pip install runpod

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what's needed to run inference
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# Environment defaults (override at runtime in RunPod console).
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-sft
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
ENV ACESTEP_MAIN_MODEL_REPO=Floppyshy/Ace-Step1.5-Custom
ENV ACESTEP_MAIN_MODEL_COMPONENTS=acestep-v15-xl-sft,vae,Qwen3-Embedding-0.6B,acestep-5Hz-lm-4B
ENV ACESTEP_CHECKPOINTS_DIR=/runpod-volume/checkpoints
ENV HF_HOME=/runpod-volume/huggingface-cache/hub
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub
ENV ACESTEP_DISABLE_DOWNLOAD=1

# Copy setup helper and handler LAST so code changes don't invalidate dep layers.
COPY setup_models.py /app/setup_models.py
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
