#!/usr/bin/env python3
"""
Local test for the ACE-Step 1.5 RunPod handler.
Run this inside the container (or locally if you have ACE-Step installed)
to verify the handler works before deploying to RunPod serverless.
"""

import json
import base64
import os
import sys

# Add /app to path if running locally with ACE-Step installed there
sys.path.insert(0, "/app")

from handler import handler


def test_text2music():
    """Test basic text-to-music generation."""
    event = {
        "input": {
            "task_type": "text2music",
            "prompt": "80s synthpop, bittersweet melancholic, high energy",
            "lyrics": "[Verse]\nWho are we playing for\nIn this empty concert hall",
            "duration": 30,
            "bpm": 118,
            "inference_steps": 8,   # turbo speed for testing
            "shift": 3.0,
            "seed": 42,
            "batch_size": 1,
            "audio_format": "mp3",
        }
    }

    print("=" * 60)
    print("TEST: text2music")
    print("=" * 60)
    result = handler(event)

    if "error" in result:
        print(f"FAILED: {result['error']}")
        print(f"Details: {result.get('details', 'N/A')}")
        return False

    outputs = result["output"]
    print(f"SUCCESS! Generated {len(outputs)} audio file(s)")
    for out in outputs:
        print(f"  - {out['filename']} (seed={out['seed']})")
        # Save to disk for listening
        out_path = f"/tmp/{out['filename']}"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(out["audio_base64"]))
        print(f"    Saved to: {out_path}")
    return True


def test_cover():
    """Test audio-to-audio style cover (requires source audio)."""
    # Create a dummy audio file or use an existing one
    dummy_path = "/tmp/dummy_src.mp3"
    if not os.path.exists(dummy_path):
        print("SKIP: No source audio for cover test. Place an MP3 at /tmp/dummy_src.mp3")
        return None

    with open(dummy_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    event = {
        "input": {
            "task_type": "cover",
            "prompt": "jazz piano version",
            "src_audio_base64": audio_b64,
            "audio_cover_strength": 0.7,
            "duration": 30,
            "inference_steps": 8,
            "seed": 42,
        }
    }

    print("=" * 60)
    print("TEST: cover (audio-to-audio)")
    print("=" * 60)
    result = handler(event)

    if "error" in result:
        print(f"FAILED: {result['error']}")
        return False

    outputs = result["output"]
    print(f"SUCCESS! Generated {len(outputs)} cover(s)")
    for out in outputs:
        out_path = f"/tmp/{out['filename']}"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(out["audio_base64"]))
        print(f"  Saved to: {out_path}")
    return True


if __name__ == "__main__":
    ok = test_text2music()
    if ok is False:
        sys.exit(1)

    test_cover()
    print("\nAll tests completed!")
