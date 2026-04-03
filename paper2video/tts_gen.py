"""
Stage 4: Text-to-speech generation.

Supports two engines:
  - orpheus: Local Orpheus 3B model (GPU, ~6GB VRAM)
  - polly:   Amazon Polly neural/long-form voices (AWS credentials required)
"""

import os
import wave


# ---------------------------------------------------------------------------
# Orpheus TTS (local GPU)
# ---------------------------------------------------------------------------

_orpheus_model = None


def _get_orpheus():
    global _orpheus_model
    if _orpheus_model is None:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from orpheus_tts import OrpheusModel

        _orig_setup = OrpheusModel._setup_engine
        def _patched_setup(self):
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                dtype=self.dtype,
                max_model_len=4096,
                gpu_memory_utilization=0.55,
            )
            return AsyncLLMEngine.from_engine_args(engine_args)
        OrpheusModel._setup_engine = _patched_setup

        print("[TTS] Loading Orpheus TTS model...")
        _orpheus_model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        print("[TTS] Model loaded.")
    return _orpheus_model


def _generate_orpheus(narration, audio_path, voice="tara"):
    model = _get_orpheus()
    audio_chunks = model.generate_speech(
        prompt=narration,
        voice=voice,
        temperature=0.4,
        top_p=0.9,
        max_tokens=2000,
        repetition_penalty=1.1,
        stop_token_ids=[49158],
    )

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for chunk in audio_chunks:
            total_frames += len(chunk) // 2
            wf.writeframes(chunk)

    return total_frames / 24000.0


# ---------------------------------------------------------------------------
# Amazon Polly
# ---------------------------------------------------------------------------

# Polly voices suited for narration (neural or long-form engine)
POLLY_VOICES = {
    # Long-form engine (best for narration)
    "Ruth":     {"engine": "long-form", "lang": "en-US"},
    "Matthew":  {"engine": "long-form", "lang": "en-US"},
    # Generative engine
    "Danielle": {"engine": "generative", "lang": "en-US"},
    "Gregory":  {"engine": "generative", "lang": "en-US"},
    "Amy":      {"engine": "generative", "lang": "en-GB"},
    # Neural engine (wider selection)
    "Joanna":   {"engine": "neural", "lang": "en-US"},
    "Kendra":   {"engine": "neural", "lang": "en-US"},
    "Salli":    {"engine": "neural", "lang": "en-US"},
    "Joey":     {"engine": "neural", "lang": "en-US"},
    "Stephen":  {"engine": "neural", "lang": "en-US"},
    "Brian":    {"engine": "neural", "lang": "en-GB"},
    "Olivia":   {"engine": "neural", "lang": "en-AU"},
}

_polly_client = None


def _get_polly():
    global _polly_client
    if _polly_client is None:
        import boto3
        _polly_client = boto3.client("polly")
        print("[TTS] Using Amazon Polly")
    return _polly_client


def _generate_polly(narration, audio_path, voice="Ruth"):
    client = _get_polly()

    voice_cfg = POLLY_VOICES.get(voice)
    if not voice_cfg:
        print(f"  [Warning] Unknown Polly voice '{voice}', falling back to Ruth")
        voice = "Ruth"
        voice_cfg = POLLY_VOICES["Ruth"]

    resp = client.synthesize_speech(
        Text=narration,
        OutputFormat="pcm",
        VoiceId=voice,
        Engine=voice_cfg["engine"],
        SampleRate="24000",
    )

    pcm_data = resp["AudioStream"].read()

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(24000)
        wf.writeframes(pcm_data)

    total_frames = len(pcm_data) // 2
    return total_frames / 24000.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_tts(script_chunks, output_dir, voice="tara", speed=1.0, engine="orpheus"):
    """
    Generate TTS audio for each narration chunk.

    Args:
        script_chunks: List of dicts with "narration" and "chunk_id".
        output_dir: Directory to save WAV files.
        voice: Voice name (engine-specific).
        speed: Speed multiplier (Orpheus only).
        engine: "orpheus" or "polly".

    Returns:
        List of dicts: [{"chunk_id": int, "audio_path": str, "duration": float}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    if engine == "polly":
        _gen = lambda text, path: _generate_polly(text, path, voice=voice)
    else:
        _gen = lambda text, path: _generate_orpheus(text, path, voice=voice)

    results = []
    for chunk in script_chunks:
        chunk_id = chunk["chunk_id"]
        narration = chunk["narration"]
        audio_path = os.path.join(output_dir, f"{chunk_id}.wav")

        try:
            duration = _gen(narration, audio_path)
            results.append({
                "chunk_id": chunk_id,
                "audio_path": audio_path,
                "duration": duration,
            })
            print(f"  [TTS] Chunk {chunk_id}: {duration:.2f}s → {audio_path}")
        except Exception as e:
            print(f"  [Warning] TTS failed for chunk {chunk_id}: {e}")

    print(f"[TTS] Generated {len(results)} audio files ({engine})")
    return results
