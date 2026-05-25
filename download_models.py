#!/usr/bin/env python3
"""Pre-download ACE-Step 1.5 model weights during Docker build."""
import os
import sys

# Ensure /app is on path so acestep imports work
sys.path.insert(0, "/app")

from acestep.model_downloader import (
    download_main_model,
    download_submodel,
    get_checkpoints_dir,
)

CHECKPOINTS_DIR = "/app/checkpoints"
os.environ["ACESTEP_CHECKPOINTS_DIR"] = CHECKPOINTS_DIR

# Which DiT model to bake in? Default to what the handler uses.
DIT_MODEL = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
# Which LM model? Default to what the handler uses.
LM_MODEL = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")

print(f"[download_models] Downloading main model to {CHECKPOINTS_DIR} ...")
success, msg = download_main_model(checkpoints_dir=CHECKPOINTS_DIR)
print(f"[download_models] Main model: {msg}")
if not success:
    sys.exit(1)

# The main model bundle includes acestep-v15-turbo, vae, Qwen3-Embedding, and acestep-5Hz-lm-1.7B.
# If the chosen DiT is NOT the turbo that ships with the main bundle, download it separately.
if DIT_MODEL != "acestep-v15-turbo":
    print(f"[download_models] Downloading DiT submodel: {DIT_MODEL} ...")
    success, msg = download_submodel(DIT_MODEL, checkpoints_dir=CHECKPOINTS_DIR)
    print(f"[download_models] DiT submodel: {msg}")
    if not success:
        sys.exit(1)

# If the chosen LM is NOT the one that ships with the main bundle, download it separately.
# (The main bundle ships acestep-5Hz-lm-1.7B, so this is usually a no-op.)
if LM_MODEL != "acestep-5Hz-lm-1.7B":
    print(f"[download_models] Downloading LM submodel: {LM_MODEL} ...")
    success, msg = download_submodel(LM_MODEL, checkpoints_dir=CHECKPOINTS_DIR)
    print(f"[download_models] LM submodel: {msg}")
    if not success:
        sys.exit(1)

print("[download_models] All models ready.")
