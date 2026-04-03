"""
Editorial Video Pipeline

Transforms a PDF document into a ~3 minute narrated video summary by:
1. Extracting real images from the document + captioning with VLM
2. Writing an editorial narration script (LLM sees the available images)
3. Matching each chunk to the best image (direct ID + CLIP/BM25 fallback)
4. Generating TTS audio per chunk (Orpheus or Amazon Polly)
5. Assembling the final video with subtitles

Usage:
    python pipeline_editorial.py \
        --pdf_path paper.pdf \
        --result_dir ./result/my_video \
        --model_name claude-sonnet \
        --voice tara

    # AWS mode: Bedrock for inference, Polly for TTS
    python pipeline_editorial.py \
        --pdf_path paper.pdf \
        --bedrock --tts_engine polly --voice Ruth
"""

import os
import json
import time
import argparse
from os import path

from wei_utils import get_agent_config
from image_extractor import extract_images_from_pdf, extract_text_from_pdf, caption_images
from script_gen import generate_script, save_script, load_script
from image_matcher import match_chunks_to_images
from tts_gen import generate_tts
from video_assembly import assemble_video


def run_pipeline(args):
    os.makedirs(args.result_dir, exist_ok=True)

    timings = {}
    token_usage = {}

    timings_path = path.join(args.result_dir, "timings.json")
    tokens_path = path.join(args.result_dir, "token_usage.json")
    if path.exists(timings_path):
        with open(timings_path, "r") as f:
            timings = json.load(f)
    if path.exists(tokens_path):
        with open(tokens_path, "r") as f:
            token_usage = json.load(f)

    stages = json.loads(args.stage)
    print(f"[Pipeline] Stages: {stages}")
    print(f"[Pipeline] PDF: {args.pdf_path}")
    print(f"[Pipeline] Output: {args.result_dir}")

    # Paths
    images_dir = path.join(args.result_dir, "extracted_images")
    image_records_path = path.join(args.result_dir, "image_records.json")
    script_path = path.join(args.result_dir, "script.json")
    matches_path = path.join(args.result_dir, "matches.json")
    audio_dir = path.join(args.result_dir, "audio")
    tts_results_path = path.join(args.result_dir, "tts_results.json")
    final_video_path = path.join(args.result_dir, "final.mp4")

    # ── Stage 1: Extract images + caption ──────────────────────────
    if "1" in stages or "0" in stages:
        print("\n" + "=" * 60)
        print("STAGE 1: Image Extraction + Captioning")
        print("=" * 60)
        t0 = time.time()

        image_records = extract_images_from_pdf(
            args.pdf_path, images_dir,
            min_size=args.min_image_size,
            page_dpi=args.page_dpi,
        )

        if not args.skip_captions:
            vlm_config = get_agent_config(args.model_name_vlm)
            image_records, caption_usage = caption_images(image_records, vlm_config)
            token_usage["captioning"] = caption_usage
        else:
            for r in image_records:
                r.setdefault("caption", "")

        with open(image_records_path, "w") as f:
            json.dump(image_records, f, indent=2)

        timings["image_extraction"] = time.time() - t0
        print(f"[Stage 1] Done in {timings['image_extraction']:.1f}s — {len(image_records)} images")

    # ── Stage 2: Script generation (sees available images) ─────────
    if "2" in stages or "0" in stages:
        print("\n" + "=" * 60)
        print("STAGE 2: Script Generation")
        print("=" * 60)
        t0 = time.time()

        # Load image records so the LLM knows what's available
        with open(image_records_path, "r") as f:
            image_records = json.load(f)

        image_captions = [
            {"id": r["id"], "caption": r.get("caption", "(no caption)")}
            for r in image_records
        ]

        full_text, _ = extract_text_from_pdf(args.pdf_path)
        agent_config = get_agent_config(args.model_name)
        script_chunks, script_usage = generate_script(
            full_text, agent_config, image_captions=image_captions,
        )
        save_script(script_chunks, script_path)
        token_usage["script_gen"] = script_usage

        timings["script_gen"] = time.time() - t0
        print(f"[Stage 2] Done in {timings['script_gen']:.1f}s — {len(script_chunks)} chunks")

    # ── Stage 3: Image matching ────────────────────────────────────
    if "3" in stages or "0" in stages:
        print("\n" + "=" * 60)
        print("STAGE 3: Image-Text Matching")
        print("=" * 60)
        t0 = time.time()

        script_chunks = load_script(script_path)
        with open(image_records_path, "r") as f:
            image_records = json.load(f)

        matches = match_chunks_to_images(
            script_chunks, image_records,
            clip_model_name=args.clip_model,
            clip_weight=args.clip_weight,
            caption_weight=1.0 - args.clip_weight,
            max_consecutive=args.max_consecutive,
        )

        with open(matches_path, "w") as f:
            json.dump(matches, f, indent=2)

        timings["matching"] = time.time() - t0
        print(f"[Stage 3] Done in {timings['matching']:.1f}s")

    # ── Stage 4: TTS ───────────────────────────────────────────────
    if "4" in stages or "0" in stages:
        print("\n" + "=" * 60)
        print("STAGE 4: TTS Generation")
        print("=" * 60)
        t0 = time.time()

        script_chunks = load_script(script_path)
        tts_results = generate_tts(
            script_chunks, audio_dir,
            voice=args.voice,
            speed=args.speed,
            engine=args.tts_engine,
        )

        with open(tts_results_path, "w") as f:
            json.dump(tts_results, f, indent=2)

        timings["tts"] = time.time() - t0
        print(f"[Stage 4] Done in {timings['tts']:.1f}s")

    # ── Stage 5: Video assembly ────────────────────────────────────
    if "5" in stages or "0" in stages:
        print("\n" + "=" * 60)
        print("STAGE 5: Video Assembly")
        print("=" * 60)
        t0 = time.time()

        script_chunks = load_script(script_path)
        with open(image_records_path, "r") as f:
            image_records = json.load(f)
        with open(matches_path, "r") as f:
            matches = json.load(f)
        with open(tts_results_path, "r") as f:
            tts_results = json.load(f)

        assemble_video(
            script_chunks, matches, image_records, tts_results,
            output_dir=args.result_dir,
            final_output_path=final_video_path,
            output_width=args.width,
            output_height=args.height,
        )

        timings["assembly"] = time.time() - t0
        print(f"[Stage 5] Done in {timings['assembly']:.1f}s")

    # ── Save tracking ──────────────────────────────────────────────
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    with open(tokens_path, "w") as f:
        json.dump(token_usage, f, indent=2)

    total = sum(timings.values())
    print(f"\n[Pipeline] Complete! Total time: {total:.1f}s")
    print(f"[Pipeline] Output: {final_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Editorial Video Pipeline: PDF → narrated video summary")

    # Input
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to source PDF document")
    parser.add_argument("--result_dir", type=str, default="./result/editorial", help="Output directory")

    # Models
    parser.add_argument("--model_name", type=str, default="claude-sonnet", help="LLM for script generation")
    parser.add_argument("--model_name_vlm", type=str, default="claude-sonnet", help="VLM for image captioning")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14", help="CLIP model for semantic fallback")

    # Image extraction
    parser.add_argument("--min_image_size", type=int, default=100, help="Minimum image dimension in px")
    parser.add_argument("--page_dpi", type=int, default=200, help="DPI for full-page renders")
    parser.add_argument("--skip_captions", action="store_true", help="Skip VLM captioning (forces semantic-only matching)")

    # Matching
    parser.add_argument("--clip_weight", type=float, default=0.6, help="CLIP vs caption weight for semantic fallback")
    parser.add_argument("--max_consecutive", type=int, default=2, help="Max consecutive chunks with same image")

    # TTS
    parser.add_argument("--tts_engine", type=str, default="orpheus", choices=["orpheus", "polly"],
                        help="TTS engine: orpheus (local GPU) or polly (Amazon Polly)")
    parser.add_argument("--voice", type=str, default="tara",
                        help="Voice name. Orpheus: tara, leah, jess, leo, etc. Polly: Ruth, Matthew, Danielle, etc.")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed multiplier (Orpheus only)")

    # Video
    parser.add_argument("--width", type=int, default=1920, help="Output video width")
    parser.add_argument("--height", type=int, default=1080, help="Output video height")

    # AWS
    parser.add_argument("--bedrock", action="store_true",
                        help="Route LLM/VLM through AWS Bedrock (remaps claude-* to bedrock-*)")

    # Pipeline control
    parser.add_argument("--stage", type=str, default='["0"]',
                        help='Stages to run: "0"=all, or subset like ["1","2","3"]')

    args = parser.parse_args()

    # --bedrock convenience: remap claude model names to bedrock equivalents
    if args.bedrock:
        _bedrock_map = {
            "claude-sonnet": "bedrock-sonnet",
            "claude-opus": "bedrock-opus",
            "claude-haiku": "bedrock-haiku",
        }
        args.model_name = _bedrock_map.get(args.model_name, args.model_name)
        args.model_name_vlm = _bedrock_map.get(args.model_name_vlm, args.model_name_vlm)

    run_pipeline(args)
