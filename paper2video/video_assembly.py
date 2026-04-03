"""
Stage 5: Assemble final video from matched images + audio chunks.

Uses ffmpeg for image-to-clip conversion and concatenation,
and Whisper for subtitle generation.
"""

import os
import subprocess


def _make_image_clip(image_path, audio_path, output_path, output_width=1920, output_height=1080):
    """
    Create a single video clip: static image scaled to fit output + audio.
    """
    # Get audio duration
    dur_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(dur_cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-filter_complex",
        (
            f"[0:v]scale={output_width}:{output_height}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"format=yuv420p[v]"
        ),
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return duration


def assemble_video(
    script_chunks,
    matches,
    image_records,
    tts_results,
    output_dir,
    final_output_path,
    output_width=1920,
    output_height=1080,
):
    """
    Assemble the final video from matched images and audio.

    Args:
        script_chunks: List of narration chunk dicts.
        matches: List of lists of image indices per chunk (from image_matcher).
        image_records: List of image record dicts with "path".
        tts_results: List of dicts with "chunk_id", "audio_path", "duration".
        output_dir: Working directory for intermediate files.
        final_output_path: Path for the final output video.
        output_width, output_height: Video resolution.

    Returns:
        Path to the final video file.
    """
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Build a map from chunk_id to tts_result
    tts_map = {r["chunk_id"]: r for r in tts_results}

    # Generate individual clips
    clip_paths = []
    for i, chunk in enumerate(script_chunks):
        chunk_id = chunk["chunk_id"]
        if chunk_id not in tts_map:
            print(f"  [Warning] No audio for chunk {chunk_id}, skipping")
            continue

        tts = tts_map[chunk_id]
        image_idx = matches[i][0]  # Best matching image
        image_path = image_records[image_idx]["path"]
        audio_path = tts["audio_path"]
        clip_path = os.path.join(clips_dir, f"clip_{chunk_id:03d}.mp4")

        print(f"  [Assembly] Chunk {chunk_id}: {os.path.basename(image_path)} + audio → clip")
        try:
            _make_image_clip(
                image_path, audio_path, clip_path,
                output_width=output_width, output_height=output_height,
            )
            clip_paths.append(clip_path)
        except Exception as e:
            print(f"  [Warning] Failed to create clip for chunk {chunk_id}: {e}")

    if not clip_paths:
        raise RuntimeError("No clips were generated. Check image/audio files.")

    # Write concat file
    concat_path = os.path.join(output_dir, "concat.txt")
    with open(concat_path, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp)}'\n")

    # Concatenate all clips
    no_subs_path = os.path.join(output_dir, "no_subs.mp4")
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        no_subs_path,
    ]
    subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
    print(f"[Assembly] Concatenated {len(clip_paths)} clips → {no_subs_path}")

    # Add subtitles using the existing subtitle_render module
    from subtitle_render import add_subtitles
    add_subtitles(no_subs_path, final_output_path, font_size=36)

    print(f"[Assembly] Final video → {final_output_path}")
    return final_output_path
