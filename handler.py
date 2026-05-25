import os
import base64
import signal
import runpod
from contextlib import contextmanager
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music, format_sample
from acestep.constants import TASK_INSTRUCTIONS
from setup_models import setup_checkpoints_from_cache


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


@contextmanager
def timeout(seconds: int, label: str = "operation"):
    """Signal-based timeout for Unix. Raises TimeoutError if exceeded."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    old_alarm = signal.alarm(seconds)
    try:
        yield
    except TimeoutError:
        raise TimeoutError(f"{label} timed out after {seconds}s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if old_alarm:
            signal.alarm(old_alarm)


GENERATION_TIMEOUT = int(os.getenv("ACESTEP_GENERATION_TIMEOUT", "300"))
INIT_TIMEOUT = int(os.getenv("ACESTEP_INIT_TIMEOUT", "180"))

# ---------------------------------------------------------------------------
# Bridge RunPod HF cache to ACE-Step checkpoint layout
# ---------------------------------------------------------------------------
with timeout(INIT_TIMEOUT, "checkpoint setup"):
    setup_checkpoints_from_cache()

# ---------------------------------------------------------------------------
# Initialize once at container startup (warm-start friendly)
# ---------------------------------------------------------------------------
print("[ACE-Step] Initializing handlers...")

dit_handler = AceStepHandler()
llm_handler = LLMHandler()

with timeout(INIT_TIMEOUT, "DiT handler init"):
    status, ok = dit_handler.initialize_service(
        project_root="/app",
        config_path=os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo"),
        device="cuda",
    )
if not ok:
    raise RuntimeError(f"AceStepHandler initialization failed: {status}")

with timeout(INIT_TIMEOUT, "LLM handler init"):
    status, ok = llm_handler.initialize(
        checkpoint_dir="/app/models",
        lm_model_path=os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-4B"),
        backend="pt",
        device="cuda",
    )
if not ok:
    raise RuntimeError(f"LLMHandler initialization failed: {status}")

print("[ACE-Step] Handlers ready!")

# ---------------------------------------------------------------------------
# RunPod serverless handler
# ---------------------------------------------------------------------------

def handler(event):
    """
    Expected input (RunPod serverless format):
    {
      "input": {
        "task_type": "text2music",     # text2music | cover | repaint | lego | extract | complete
        "prompt": "upbeat pop song",
        "lyrics": "...",               # or "[Instrumental]"
        "duration": 120,               # seconds, 10-600
        "bpm": 128,                    # optional
        "keyscale": "C Major",         # optional (also accepts "key_scale")
        "inference_steps": 8,          # 8 for turbo, 50 for xl-sft
        "guidance_scale": 7.0,
        "shift": 3.0,                  # 3.0 for turbo, 1.0 for xl-sft
        "seed": -1,
        "batch_size": 2,               # 1-8 (default 2, matching official API)
        "audio_format": "mp3",         # mp3 | flac | wav
        "thinking": false,             # use LM planner for structure
        "lm_temperature": 0.85,
        "lm_cfg_scale": 2.5,           # LLM CFG scale (official API default)
        "use_format": false,           # use format_sample() to enhance caption/lyrics
        "track_name": "...",           # for extract/lego tasks

        # For cover / repaint / style-transfer
        "reference_audio_base64": "...",  # base64-encoded reference audio (style influence)
        "src_audio_base64": "...",        # base64-encoded source audio (audio-to-audio)
        "audio_cover_strength": 0.7,      # 0.1 (loose) - 1.0 (strict)
        "repainting_start": 10.0,
        "repainting_end": 20.0,
      }
    }
    """
    job_input = event.get("input", {})

    task_type = job_input.get("task_type", "text2music")
    caption = job_input.get("prompt") or job_input.get("caption", "")
    lyrics = job_input.get("lyrics", "[Instrumental]")
    duration = job_input.get("duration", 30)
    bpm = job_input.get("bpm")
    keyscale = job_input.get("keyscale") or job_input.get("key_scale", "N/A")
    inference_steps = job_input.get("inference_steps", 8)
    guidance_scale = job_input.get("guidance_scale", 7.0)
    shift = job_input.get("shift", 3.0)
    seed = job_input.get("seed", -1)
    batch_size = job_input.get("batch_size", 2)
    audio_format = job_input.get("audio_format", "mp3")
    thinking = job_input.get("thinking", False)
    lm_temperature = job_input.get("lm_temperature", 0.85)
    lm_cfg_scale = job_input.get("lm_cfg_scale", 2.5)
    use_format = job_input.get("use_format", False)

    # Resolve task-specific instruction
    instruction = TASK_INSTRUCTIONS.get(task_type, "Fill the audio semantic mask based on the given conditions:")
    if task_type in ("extract", "lego"):
        track_name = job_input.get("track_name", "")
        if track_name:
            instruction = instruction.format(TRACK_NAME=track_name.upper())
        else:
            instruction = TASK_INSTRUCTIONS.get(f"{task_type}_default", instruction)

    # Format sample enhancement (optional)
    if use_format and caption:
        user_metadata = {}
        if bpm is not None:
            user_metadata["bpm"] = bpm
        if duration is not None and float(duration) > 0:
            user_metadata["duration"] = float(duration)
        if keyscale and keyscale != "N/A":
            user_metadata["keyscale"] = keyscale
        try:
            fmt_result = format_sample(
                llm_handler=llm_handler,
                caption=caption,
                lyrics=lyrics,
                user_metadata=user_metadata if user_metadata else None,
                temperature=lm_temperature,
                use_constrained_decoding=True,
            )
            if fmt_result.success:
                caption = fmt_result.caption or caption
                lyrics = fmt_result.lyrics or lyrics
                if fmt_result.bpm:
                    bpm = fmt_result.bpm
                if fmt_result.duration:
                    duration = fmt_result.duration
                if fmt_result.keyscale:
                    keyscale = fmt_result.keyscale
                print(f"[ACE-Step] format_sample applied: caption={caption[:60]}...")
            else:
                print(f"[ACE-Step] format_sample failed: {fmt_result.error}")
        except Exception as exc:
            print(f"[ACE-Step] format_sample error: {exc}")

    # Build params
    params = GenerationParams(
        task_type=task_type,
        instruction=instruction,
        caption=caption,
        lyrics=lyrics,
        duration=duration,
        bpm=bpm,
        keyscale=keyscale,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        seed=seed,
        lm_cfg_scale=lm_cfg_scale,
    )

    # Disable DCW for SFT models — PR #1207; DCW at 50 steps causes garbage audio
    if "sft" in os.getenv("ACESTEP_CONFIG_PATH", "").lower():
        params.dcw_enabled = False

    # LM thinking mode
    if thinking:
        params.thinking = True
        params.lm_temperature = lm_temperature
        params.dcw_scaler = 0.02
        params.dcw_high_scaler = 0.06
    else:
        params.dcw_scaler = 0.05
        params.dcw_high_scaler = 0.02

    # Handle reference audio for style transfer / cover
    ref_audio_b64 = job_input.get("reference_audio_base64")
    if ref_audio_b64:
        audio_bytes = base64.b64decode(ref_audio_b64)
        ref_path = f"/tmp/ref_audio_{abs(hash(ref_audio_b64)) % 100000}.mp3"
        with open(ref_path, "wb") as f:
            f.write(audio_bytes)
        params.reference_audio = ref_path

    # Handle source audio for cover / repaint
    src_audio_b64 = job_input.get("src_audio_base64")
    if src_audio_b64 and task_type in ("cover", "repaint"):
        audio_bytes = base64.b64decode(src_audio_b64)
        src_path = f"/tmp/src_audio_{abs(hash(src_audio_b64)) % 100000}.mp3"
        with open(src_path, "wb") as f:
            f.write(audio_bytes)
        params.src_audio = src_path

        if task_type == "cover":
            params.audio_cover_strength = job_input.get("audio_cover_strength", 0.7)
            # Auto-extract audio semantic codes from src_audio so the cover actually
            # preserves the melodic structure of the original. Without codes, cover
            # degrades to text2music with only padding/repaint masks.
            print(f"[ACE-Step] Extracting audio codes from source audio...")
            codes = dit_handler.convert_src_audio_to_codes(src_path)
            if codes and not codes.startswith("❌"):
                params.audio_codes = codes
                print(f"[ACE-Step] Extracted {codes.count('audio_code_')} audio codes")
            else:
                print(f"[ACE-Step] Audio code extraction failed: {codes}")
        elif task_type == "repaint":
            params.repainting_start = job_input.get("repainting_start", 0.0)
            params.repainting_end = job_input.get("repainting_end", 10.0)

    config = GenerationConfig(
        batch_size=batch_size,
        audio_format=audio_format,
        allow_lm_batch=True,
    )

    save_dir = "/tmp/outputs"
    os.makedirs(save_dir, exist_ok=True)

    print(f"[ACE-Step] Generating: task={task_type}, caption={caption[:60]}...")
    try:
        with timeout(GENERATION_TIMEOUT, "music generation"):
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=save_dir)
    except TimeoutError as exc:
        return {
            "error": "Generation timed out",
            "details": str(exc),
        }

    if not result.success:
        return {
            "error": "Generation failed",
            "details": str(result),
        }

    outputs = []
    for audio in result.audios:
        path = audio["path"]
        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        outputs.append({
            "filename": os.path.basename(path),
            "audio_base64": audio_b64,
            "seed": audio["params"]["seed"],
            "format": audio_format,
        })

    return {
        "output": outputs,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
