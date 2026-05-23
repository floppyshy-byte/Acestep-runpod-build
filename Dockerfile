# =============================================================================
# ACE-Step 1.5 — RunPod Serverless Worker
# =============================================================================
# Multi-stage build for space efficiency:
#   - Builder stage compiles deps and downloads models (heavy, slow).
#   - Runtime stage keeps only what's needed for inference (lean, fast pull).
#
# Layer ordering (large / stable -> small / volatile):
#   1. Base image
#   2. System runtime libraries
#   3. Copy uv + /app (source, .venv, checkpoints)
#   4. Copy handler.py  <-- changes often; kept last for cache friendliness
#   5. CMD
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — dependencies, compilation, model download
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# Install build-time dependencies in a single layer so they can be discarded later.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg libsndfile1 \
    python3.11 python3.11-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (small single binary).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone ACE-Step 1.5 source (stable — rarely changes).
WORKDIR /app
RUN git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git .

# Install Python dependencies. uv will use python3.11 because of requires-python.
RUN uv sync
RUN uv pip install runpod

# Pre-download model weights so they are baked into the image.
# This layer is VERY LARGE (~10–15 GB). Keep it before handler code.
ENV ACESTEP_CONFIG_PATH=acestep-v15-sft
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_CHECKPOINTS_DIR=/app/checkpoints
RUN mkdir -p /app/checkpoints /app/models /tmp/outputs

COPY download_models.py /app/download_models.py
RUN uv run python3 /app/download_models.py

# ---------------------------------------------------------------------------
# Stage 2: Runtime — only what is needed to run inference
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install only runtime system libraries (no compilers, no git).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary and the full /app tree (source code, .venv, checkpoints).
# The .venv was created against /usr/bin/python3.11 which exists in this stage too.
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# Environment defaults (can be overridden at runtime in RunPod console).
ENV ACESTEP_CONFIG_PATH=acestep-v15-sft
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_CHECKPOINTS_DIR=/app/checkpoints
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Copy handler LAST — code changes here won't invalidate the huge model layer above.
COPY handler.py /app/handler.py

# RunPod serverless entrypoint with unbuffered output.
CMD ["python3", "-u", "/app/handler.py"]
