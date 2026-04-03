"""
Paper2Video API — thin FastAPI wrapper around the editorial pipeline.

Accepts PDF uploads, runs the pipeline in a background thread (single job queue),
and serves the resulting video.

    uvicorn api:app --host 0.0.0.0 --port 8001
"""

import io
import logging
import os
import re
import shutil
import threading
import time
import uuid
from argparse import Namespace
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

log = logging.getLogger(__name__)

app = FastAPI(title="Paper2Video")

RESULT_BASE = Path(os.environ.get("P2V_RESULT_DIR", "result"))

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()
_current_job: str | None = None

_STAGE_RE = re.compile(r"^STAGE\s+(\d+)")


def _get_job(job_id: str) -> dict | None:
    """Return a snapshot (copy) of the job dict, or None."""
    with _job_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------

def _run_job(job_id: str, pdf_path: str, result_dir: str, model: str, voice: str, tts_engine: str):
    global _current_job
    try:
        with _job_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["stage"] = "starting"
            _current_job = job_id

        from pipeline_editorial import run_pipeline

        args = Namespace(
            pdf_path=pdf_path,
            result_dir=result_dir,
            model_name=model,
            model_name_vlm=model,
            clip_model="openai/clip-vit-large-patch14",
            min_image_size=100,
            page_dpi=200,
            skip_captions=False,
            clip_weight=0.6,
            max_consecutive=2,
            voice=voice,
            speed=1.0,
            tts_engine=tts_engine,
            width=1920,
            height=1080,
            stage='["0"]',
        )

        # Capture stage updates via a stream wrapper instead of monkey-patching print
        _real_stdout = __import__("sys").stdout

        class _StageCapture(io.TextIOBase):
            def write(self, s):
                _real_stdout.write(s)
                m = _STAGE_RE.match(s)
                if m:
                    with _job_lock:
                        _jobs[job_id]["stage"] = f"stage_{m.group(1)}"
                return len(s)

            def flush(self):
                _real_stdout.flush()

        import sys
        sys.stdout = _StageCapture()
        try:
            run_pipeline(args)
        finally:
            sys.stdout = _real_stdout

        video_path = os.path.join(result_dir, "final.mp4")
        with _job_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["stage"] = "complete"
            _jobs[job_id]["video_path"] = video_path

    except Exception as e:
        log.exception("Job %s failed", job_id)
        with _job_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
    finally:
        with _job_lock:
            _current_job = None
        _process_queue()


def _process_queue():
    """Start the next queued job if nothing is running."""
    with _job_lock:
        if _current_job is not None:
            return
        for jid, job in _jobs.items():
            if job["status"] == "queued":
                # Capture values under the lock before starting thread
                args = (jid, job["pdf_path"], job["result_dir"], job["model"], job["voice"], job["tts_engine"])
                break
        else:
            return

    thread = threading.Thread(
        target=_run_job,
        args=args,
        name=f"p2v-{jid[:8]}",
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs")
async def create_job(
    pdf: UploadFile = File(...),
    model: str = Form("claude-sonnet"),
    voice: str = Form("tara"),
    tts_engine: str = Form("orpheus"),
):
    job_id = uuid.uuid4().hex[:12]
    result_dir = str(RESULT_BASE / job_id)
    os.makedirs(result_dir, exist_ok=True)

    pdf_path = os.path.join(result_dir, "input.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    with _job_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": None,
            "created_at": time.time(),
            "pdf_path": pdf_path,
            "result_dir": result_dir,
            "model": model,
            "voice": voice,
            "tts_engine": tts_engine,
            "video_path": None,
            "error": None,
        }

    _process_queue()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "error": job["error"],
    }


@app.get("/jobs/{job_id}/video")
async def get_video(job_id: str):
    job = _get_job(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    if job["status"] != "done" or not job.get("video_path"):
        return JSONResponse({"error": "not ready"}, status_code=409)
    if not os.path.exists(job["video_path"]):
        return JSONResponse({"error": "video file missing"}, status_code=404)
    return FileResponse(
        job["video_path"],
        media_type="video/mp4",
        filename=f"paper2video_{job_id}.mp4",
    )
