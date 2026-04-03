"""Subtitle rendering — Whisper transcription + MoviePy overlay."""

import whisper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

# Font search order — first match wins
_FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "arial.ttf",
]


def _load_font(font_size):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def create_subtitle_image(text, font_size=32):
    font = _load_font(font_size)

    dummy_img = Image.new("RGBA", (70, 70))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    box_w = text_w + 2 * padding
    box_h = text_h + 2 * padding
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 160))

    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, font=font, fill=(255, 255, 255, 255))

    return img


def generate_subtitle_clips(segments, video_w, video_h, font_size):
    clips = []
    for seg in segments:
        img = create_subtitle_image(seg["text"], font_size=font_size)
        img_array = np.array(img)
        clip = (ImageClip(img_array, ismask=False)
                .set_duration(seg["end"] - seg["start"])
                .set_start(seg["start"])
                .set_position(("center", video_h - font_size * 2)))
        clips.append(clip)
    return clips


def add_subtitles(video_path, output_path, font_size):
    print("[Subtitles] Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path, language="en")
    segments = result["segments"]

    print("[Subtitles] Generating subtitle clips...")
    video = VideoFileClip(video_path)
    subs = generate_subtitle_clips(segments, video.w, video.h, font_size)

    print("[Subtitles] Rendering final video...")
    final = CompositeVideoClip([video] + subs)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
