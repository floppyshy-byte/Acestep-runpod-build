FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone ACE-Step 1.5
WORKDIR /app
RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git .

# Install project dependencies
RUN uv sync

# Install RunPod serverless SDK in the uv environment
RUN uv pip install runpod

# Create directories for models and outputs
RUN mkdir -p /app/checkpoints /app/models /tmp/outputs

# Environment defaults (override at runtime)
ENV ACESTEP_CONFIG_PATH=acestep-v15-sft
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_CHECKPOINTS_DIR=/app/checkpoints
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Pre-download model weights so they are baked into the image.
# This avoids downloading on every cold start but makes the image large (~10-15 GB).
COPY download_models.py /app/download_models.py
RUN uv run python3 /app/download_models.py

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["uv", "run", "python3", "/app/handler.py"]
