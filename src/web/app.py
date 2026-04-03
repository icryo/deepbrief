"""FastAPI web application — Research Intelligence Dashboard."""

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

from starlette.middleware.base import BaseHTTPMiddleware

from src.config import SCORING_CONFIGS
from src.db import (
    clear_preferences,
    count_events,
    count_github_projects,
    count_papers,
    create_video_job,
    delete_signal,
    get_all_runs,
    get_available_topics,
    get_events,
    get_github_languages,
    get_github_projects_page,
    get_latest_run,
    get_paper,
    get_paper_connections,
    get_paper_signal,
    get_papers_page,
    get_preferences_detail,
    get_preferences_updated_at,
    get_signal_counts,
    get_top_papers,
    get_video_job,
    init_db,
    insert_signal,
    load_preferences,
    search_papers_fts,
    update_video_job,
)
from src.cli_intel_db import (
    count_findings as count_cli_findings,
    count_new_findings as count_cli_new_findings,
    get_cli_findings_page,
    get_cli_repos,
    get_finding_repos,
    init_cli_intel_db,
)
from src.finboard import add_paper_to_finboard, is_in_finboard
from src.preferences import compute_preferences, enrich_papers_with_preferences

app = FastAPI(title="Research Intelligence")

# Static files & templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"
DATA_DIR = Path("data")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ---------------------------------------------------------------------------
# First-run redirect middleware
# ---------------------------------------------------------------------------


class FirstRunMiddleware(BaseHTTPMiddleware):
    """Redirect all non-setup requests to /setup when config.yaml is missing."""

    _ALLOWED_PREFIXES = ("/setup", "/static", "/api/setup", "/sw.js")

    async def dispatch(self, request: Request, call_next):
        from src.config import CONFIG_PATH
        if not CONFIG_PATH.exists():
            path = request.url.path
            if not any(path.startswith(p) for p in self._ALLOWED_PREFIXES):
                return RedirectResponse("/setup", status_code=302)
        return await call_next(request)


app.add_middleware(FirstRunMiddleware)


@app.get("/sw.js")
async def service_worker():
    """Serve SW from root scope for PWA."""
    from fastapi.responses import FileResponse
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )


def score_bar(value, max_val=10):
    """Render a visual score bar."""
    if value is None or max_val == 0:
        return "░" * 10
    filled = round(float(value) * 10 / max_val)
    filled = max(0, min(10, filled))
    return "█" * filled + "░" * (10 - filled)


def format_date(value, fmt="short"):
    """Format dates from various input formats (ISO, RFC 2822, etc.)."""
    if not value:
        return ""
    from email.utils import parsedate_to_datetime
    dt = None
    # Try ISO format first
    for pattern in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value[:26], pattern)
            break
        except (ValueError, TypeError):
            continue
    # Try RFC 2822 (RSS dates like "Wed, 18 Feb 2026 21:00:00 GMT")
    if dt is None:
        try:
            dt = parsedate_to_datetime(value)
        except (ValueError, TypeError):
            return value[:10] if len(value) >= 10 else value
    if fmt == "short":
        return dt.strftime("%Y-%m-%d")
    elif fmt == "medium":
        return dt.strftime("%b %d, %Y")
    elif fmt == "long":
        return dt.strftime("%a, %b %d %Y")
    return dt.strftime("%Y-%m-%d")


def abbreviate_label(label):
    """Abbreviate axis labels for table headers."""
    abbrevs = {
        "Code & Weights": "Code/Wt",
        "Novelty": "Novel",
        "Practical Applicability": "Practical",
        "Has Code/PoC": "Code/PoC",
        "Novel Attack Surface": "Attack",
        "Real-World Impact": "Impact",
    }
    return abbrevs.get(label, label[:10])


# Register as Jinja2 globals/filters
templates.env.globals["score_bar"] = score_bar
templates.env.globals["abbreviate_label"] = abbreviate_label
templates.env.filters["format_date"] = format_date


def _feature_enabled(feature: str) -> bool:
    """Check if a feature (github, events) is enabled in config."""
    from src.config import _cfg
    return _cfg.get(feature, {}).get("enabled", True)


templates.env.globals["github_enabled"] = lambda: _feature_enabled("github")
templates.env.globals["events_enabled"] = lambda: _feature_enabled("events")
templates.env.globals["cli_intel_enabled"] = lambda: _feature_enabled("cli_intel")


def _is_pipeline_enabled(pipeline: str) -> bool:
    """Wrapper for template use — checks if a pipeline is enabled."""
    from src.config import is_pipeline_enabled
    return is_pipeline_enabled(pipeline)


templates.env.globals["is_pipeline_enabled"] = _is_pipeline_enabled


@app.on_event("startup")
def startup():
    from src.config import validate_env
    validate_env()
    init_db()
    init_cli_intel_db()
    from src.scheduler import start_scheduler
    start_scheduler()
    log.info("Research Intelligence started")


@app.on_event("shutdown")
def shutdown():
    from src.scheduler import scheduler
    scheduler.shutdown(wait=False)
    # Snapshot thread list under lock before iterating
    with _pipeline_lock:
        threads = list(_pipeline_threads)
    for t in threads:
        if t.is_alive():
            log.info("Waiting for %s to finish ...", t.name)
            t.join(timeout=30)
    log.info("Research Intelligence stopped")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    now = datetime.now(timezone.utc)
    week_label = now.strftime("%b %d, %Y")

    aiml_top = get_top_papers("aiml", limit=5)
    security_top = get_top_papers("security", limit=5)

    # Enrich dashboard cards with preference data
    preferences = load_preferences()
    if preferences:
        aiml_top = enrich_papers_with_preferences(aiml_top, preferences)
        security_top = enrich_papers_with_preferences(security_top, preferences)

    aiml_run = get_latest_run("aiml")
    security_run = get_latest_run("security")

    last_run = None
    for r in [aiml_run, security_run]:
        if r and r.get("finished_at"):
            ts = r["finished_at"][:16]
            if last_run is None or ts > last_run:
                last_run = ts

    events = get_events(limit=50)
    today = now.strftime("%Y-%m-%d")
    # Deduplicate + filter past conference deadlines
    events_grouped = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    for e in events:
        cat = e.get("category", "other")
        title = e.get("title", "")
        if title in seen[cat]:
            continue
        # Skip past conference deadlines
        if cat == "conference" and (e.get("event_date") or "") < today:
            continue
        seen[cat].add(title)
        events_grouped[cat].append(e)

    with _pipeline_lock:
        running = list(_running_pipelines)

    # Show seed banner if few signals exist
    signal_counts = get_signal_counts()
    total_signals = sum(v for k, v in signal_counts.items() if k != "view")
    show_seed_banner = total_signals < 5

    aiml_count = count_papers("aiml", scored_only=True)
    security_count = count_papers("security", scored_only=True)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active": "dashboard",
        "week_label": week_label,
        "aiml_count": aiml_count,
        "security_count": security_count,
        "github_count": count_github_projects(),
        "event_count": count_events(),
        "last_run": last_run,
        "aiml_top": aiml_top,
        "security_top": security_top,
        "events": events,
        "events_grouped": dict(events_grouped),
        "running_pipelines": running,
        "show_seed_banner": show_seed_banner,
        "has_papers": (aiml_count + security_count) > 0,
        "cli_intel_count": count_cli_findings(),
        "cli_intel_new": count_cli_new_findings(),
    })


# ---------------------------------------------------------------------------
# Papers list
# ---------------------------------------------------------------------------


@app.get("/papers/{domain}", response_class=HTMLResponse)
async def papers_list(
    request: Request,
    domain: str,
    offset: int = 0,
    limit: int = 50,
    search: str | None = None,
    min_score: str | None = None,
    has_code: bool = False,
    topic: str | None = None,
    sort: str | None = None,
):
    if domain not in ("aiml", "security"):
        return RedirectResponse("/")

    # Convert min_score from string (empty string from blank input -> None)
    min_score_val: float | None = None
    if min_score:
        try:
            min_score_val = float(min_score)
        except ValueError:
            min_score_val = None

    config = SCORING_CONFIGS[domain]
    run = get_latest_run(domain) or {}

    # Load preferences to determine if personalized sort is available
    preferences = load_preferences()
    has_preferences = bool(preferences)

    # Default to personalized sort when preferences exist
    effective_sort = sort
    if sort == "adjusted" and not has_preferences:
        effective_sort = "score"

    papers, total = get_papers_page(
        domain, run_id=run.get("id"),
        offset=offset, limit=limit,
        min_score=min_score_val,
        has_code=has_code if has_code else None,
        search=search,
        topic=topic,
        sort=effective_sort if effective_sort != "adjusted" else "score",
    )

    # Enrich with preferences
    sort_adjusted = (sort == "adjusted") and has_preferences
    papers = enrich_papers_with_preferences(papers, preferences, sort_adjusted=sort_adjusted)

    # Get available topics for the filter dropdown
    available_topics = get_available_topics(domain, run.get("id", 0)) if run else []

    domain_label = "AI/ML" if domain == "aiml" else "Security"

    # Detect partial scoring: papers fetched but not all scored
    scoring_incomplete = False
    if run.get("id"):
        total_in_run = count_papers(domain, run_id=run["id"], scored_only=False)
        scored_in_run = count_papers(domain, run_id=run["id"], scored_only=True)
        if total_in_run > 0 and scored_in_run < total_in_run:
            scoring_incomplete = True

    context = {
        "request": request,
        "active": domain,
        "domain": domain,
        "domain_label": domain_label,
        "papers": papers,
        "total": total,
        "offset": offset,
        "limit": limit,
        "search": search,
        "min_score": min_score_val,
        "has_code": has_code,
        "topic": topic,
        "sort": sort,
        "available_topics": available_topics,
        "run": run,
        "axis_labels": config["axis_labels"],
        "has_preferences": has_preferences,
        "scoring_incomplete": scoring_incomplete,
    }

    # Return partial for HTMX requests (filter / pagination)
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/papers_results.html", context)

    return templates.TemplateResponse("papers.html", context)


# ---------------------------------------------------------------------------
# Paper detail
# ---------------------------------------------------------------------------


@app.get("/papers/{domain}/{paper_id}", response_class=HTMLResponse)
async def paper_detail(request: Request, domain: str, paper_id: int):
    paper = get_paper(paper_id)
    if not paper:
        return RedirectResponse(f"/papers/{domain}")

    config = SCORING_CONFIGS.get(domain, SCORING_CONFIGS["aiml"])
    domain_label = "AI/ML" if domain == "aiml" else "Security"

    connections = get_paper_connections(paper_id)

    # Record view signal (deduped by 5-min window)
    insert_signal(paper_id, "view")

    # Preference boost info
    preferences = load_preferences()
    papers_enriched = enrich_papers_with_preferences([paper], preferences)
    paper = papers_enriched[0]

    # Check finboard status
    arxiv_url = paper.get("arxiv_url", "")
    if not arxiv_url:
        arxiv_url = f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}"
    in_finboard = is_in_finboard(arxiv_url, title=paper.get("title", ""))

    video_job = get_video_job(paper_id)

    return templates.TemplateResponse("paper_detail.html", {
        "request": request,
        "active": domain,
        "domain": domain,
        "domain_label": domain_label,
        "paper": paper,
        "axis_labels": config["axis_labels"],
        "score_bar": score_bar,
        "connections": connections,
        "in_finboard": in_finboard,
        "video_job": video_job,
    })


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@app.get("/events", response_class=HTMLResponse)
async def events_page(request: Request):
    deadlines_raw = get_events(category="conference", limit=50)
    releases = get_events(category="release", limit=20)
    news_raw = get_events(category="news", limit=40)

    # Filter out past deadlines
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    deadlines = [d for d in deadlines_raw if (d.get("event_date") or "") >= today]

    # Deduplicate news by title and sort by date (RFC 2822 dates don't sort lexicographically)
    from email.utils import parsedate_to_datetime as _parse_rfc
    seen_titles: set[str] = set()
    news: list[dict] = []
    for n in news_raw:
        t = n.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            news.append(n)

    def _news_sort_key(item):
        d = item.get("event_date", "")
        try:
            return _parse_rfc(d)
        except (ValueError, TypeError):
            try:
                return datetime.fromisoformat(d[:19])
            except (ValueError, TypeError):
                return datetime.min

    news.sort(key=_news_sort_key, reverse=True)
    news = news[:20]

    return templates.TemplateResponse("events.html", {
        "request": request,
        "active": "events",
        "total": count_events(),
        "deadlines": deadlines,
        "releases": releases,
        "news": news,
    })


# ---------------------------------------------------------------------------
# GitHub Projects
# ---------------------------------------------------------------------------


@app.get("/github", response_class=HTMLResponse)
async def github_page(
    request: Request,
    offset: int = 0,
    limit: int = 50,
    search: str | None = None,
    language: str | None = None,
    domain: str | None = None,
    sort: str | None = None,
):
    run = get_latest_run("github") or {}

    projects, total = get_github_projects_page(
        run_id=run.get("id"),
        offset=offset,
        limit=limit,
        search=search,
        language=language,
        domain=domain,
        sort=sort,
    )

    available_languages = get_github_languages(run["id"]) if run else []

    context = {
        "request": request,
        "active": "github",
        "projects": projects,
        "total": total,
        "offset": offset,
        "limit": limit,
        "search": search,
        "language": language,
        "domain_filter": domain,
        "sort": sort,
        "available_languages": available_languages,
        "run": run,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/github_results.html", context)

    return templates.TemplateResponse("github.html", context)


# ---------------------------------------------------------------------------
# CLI Intelligence
# ---------------------------------------------------------------------------


@app.get("/cli-intel", response_class=HTMLResponse)
async def cli_intel_page(
    request: Request,
    offset: int = 0,
    limit: int = 50,
    search: str | None = None,
    repo: str | None = None,
    type: str | None = None,
    sort: str | None = None,
    new_only: bool = False,
):
    run = get_latest_run("cli_intel") or {}
    run_id = run.get("id")

    repos = get_cli_repos(run_id) if run_id else []
    available_repos = get_finding_repos(run_id) if run_id else []

    # Apply search filter at DB level via title matching
    findings, total = [], 0
    if run_id:
        findings, total = get_cli_findings_page(
            run_id=run_id,
            offset=offset,
            limit=limit,
            repo=repo,
            finding_type=type,
            sort=sort,
            new_only=new_only,
        )

    # If search is provided, filter in-memory (findings table has no FTS)
    if search and findings:
        search_lower = search.lower()
        filtered = [f for f in findings if search_lower in f.get("title", "").lower()
                    or search_lower in (f.get("description") or "").lower()
                    or search_lower in (f.get("summary") or "").lower()]
        total = len(filtered)
        findings = filtered

    new_count = count_cli_new_findings(run_id) if run_id else 0

    context = {
        "request": request,
        "active": "cli_intel",
        "findings": findings,
        "total": total,
        "offset": offset,
        "limit": limit,
        "search": search,
        "repo_filter": repo,
        "type_filter": type,
        "sort": sort,
        "new_only": new_only,
        "repos": repos,
        "available_repos": available_repos,
        "run": run,
        "new_count": new_count,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/cli_intel_results.html", context)

    return templates.TemplateResponse("cli_intel.html", context)


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


@app.get("/weeks", response_class=HTMLResponse)
async def weeks_page(request: Request, q: str = "", domain: str = "",
                     sort: str = "rank", page: int = 1):
    weeks_dir = DATA_DIR / "weeks"
    archives = []
    if weeks_dir.exists():
        for f in sorted(weeks_dir.glob("*.md"), reverse=True):
            parts = f.stem.rsplit("-", 1)
            file_domain = parts[-1] if len(parts) > 1 and parts[-1] in ("aiml", "security") else "unknown"
            date = parts[0] if len(parts) > 1 else f.stem
            archives.append({"filename": f.name, "date": date, "domain": file_domain})

    runs = get_all_runs(limit=20)

    search_results = []
    total = 0
    per_page = 50
    if q.strip():
        offset = (page - 1) * per_page
        search_results, total = search_papers_fts(
            q.strip(), domain=domain or None, sort=sort,
            limit=per_page, offset=offset,
        )

    ctx = {
        "request": request,
        "active": "weeks",
        "archives": archives,
        "runs": runs,
        "search_results": search_results,
        "total": total,
        "q": q,
        "domain": domain,
        "sort": sort,
        "page": page,
        "per_page": per_page,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/archive_results.html", ctx)

    return templates.TemplateResponse("weeks.html", ctx)


@app.get("/weeks/{filename}", response_class=HTMLResponse)
async def weeks_file(filename: str):
    import html as html_mod
    filepath = (DATA_DIR / "weeks" / filename).resolve()
    weeks_root = (DATA_DIR / "weeks").resolve()
    if not filepath.is_relative_to(weeks_root) or not filepath.exists() or not filepath.suffix == ".md":
        return RedirectResponse("/weeks")
    content = html_mod.escape(filepath.read_text())
    safe_name = html_mod.escape(filename)
    page = f"""<!DOCTYPE html><html><head><title>{safe_name}</title>
    <link rel="stylesheet" href="/static/style.css">
    <style>body {{ padding: 2rem; max-width: 900px; margin: 0 auto; }}
    pre, code {{ font-family: var(--font-mono); }} table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid var(--border); padding: 0.5rem; text-align: left; }}</style>
    </head><body><a href="/weeks">&larr; Back to archive</a>
    <pre style="white-space:pre-wrap; line-height:1.7">{content}</pre></body></html>"""
    return HTMLResponse(content=page)


# ---------------------------------------------------------------------------
# Pipeline triggers
# ---------------------------------------------------------------------------


_running_pipelines: set[str] = set()
_pipeline_lock = threading.Lock()
_pipeline_threads: list[threading.Thread] = []


def _enrich_s2(run_id: int, domain: str):
    """Run S2 enrichment (best-effort, failures don't break pipeline)."""
    try:
        from src.pipelines.semantic_scholar import enrich_run
        enrich_run(run_id, domain)
    except Exception as e:
        log.warning("S2 enrichment for %s run %d failed: %s", domain, run_id, e)


def _run_pipeline_bg(domain: str):
    """Run a pipeline in a background thread."""
    try:
        if domain == "aiml":
            from src.pipelines.aiml import run_aiml_pipeline
            from src.scoring import rescore_top, score_run
            run_id = run_aiml_pipeline()
            score_run(run_id, "aiml")
            rescore_top(run_id, "aiml")
            _enrich_s2(run_id, "aiml")
            _generate_report(run_id, "aiml")
        elif domain == "security":
            from src.pipelines.security import run_security_pipeline
            from src.scoring import rescore_top, score_run
            run_id = run_security_pipeline()
            score_run(run_id, "security")
            rescore_top(run_id, "security")
            _enrich_s2(run_id, "security")
            _generate_report(run_id, "security")
        elif domain == "github":
            from src.pipelines.github import run_github_pipeline
            run_github_pipeline()
        elif domain == "events":
            from src.pipelines.events import run_events_pipeline
            run_events_pipeline()
        elif domain == "cli_intel":
            from src.pipelines.cli_intel import run_cli_intel_pipeline
            run_cli_intel_pipeline()

        # Recompute preferences after scoring so adjusted rankings stay fresh
        if domain in ("aiml", "security"):
            try:
                from src.db import get_signal_counts as _get_counts
                if sum(_get_counts().values()) > 0:
                    compute_preferences()
            except Exception:
                log.warning("Post-pipeline preference recompute failed")
    except Exception:
        log.exception("Background pipeline %s failed", domain)
    finally:
        with _pipeline_lock:
            _running_pipelines.discard(domain)


def _cleanup_dead_threads():
    """Remove finished threads from tracking list. Must be called under _pipeline_lock."""
    _pipeline_threads[:] = [t for t in _pipeline_threads if t.is_alive()]


@app.post("/run/{domain}")
async def trigger_run(domain: str):
    if domain not in ("aiml", "security", "github", "events", "cli_intel"):
        return RedirectResponse("/", status_code=303)

    from src.config import is_pipeline_enabled
    if not is_pipeline_enabled(domain):
        return RedirectResponse("/", status_code=303)

    with _pipeline_lock:
        if domain in _running_pipelines:
            return RedirectResponse("/", status_code=303)
        _running_pipelines.add(domain)
        _cleanup_dead_threads()
        try:
            thread = threading.Thread(target=_run_pipeline_bg, args=(domain,), name=f"pipeline-{domain}")
            thread.start()
            _pipeline_threads.append(thread)
        except Exception:
            _running_pipelines.discard(domain)
            log.exception("Failed to start pipeline thread for %s", domain)
    return RedirectResponse("/", status_code=303)


# ---------------------------------------------------------------------------
# API status
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def api_status():
    aiml_run = get_latest_run("aiml")
    security_run = get_latest_run("security")
    github_run = get_latest_run("github")
    cli_intel_run = get_latest_run("cli_intel")
    with _pipeline_lock:
        running = list(_running_pipelines)
    return {
        "aiml": aiml_run,
        "security": security_run,
        "github": github_run,
        "cli_intel": cli_intel_run,
        "github_count": count_github_projects(),
        "events_count": count_events(),
        "cli_intel_count": count_cli_findings(),
        "running_pipelines": running,
    }


# ---------------------------------------------------------------------------
# Finboard integration
# ---------------------------------------------------------------------------


@app.post("/api/finboard/{paper_id}", response_class=HTMLResponse)
async def add_to_finboard(request: Request, paper_id: int):
    """Add a paper to finboard's reading list. Returns HTMX partial."""
    paper = get_paper(paper_id)
    if not paper:
        return HTMLResponse('<span style="color:var(--red)">Paper not found</span>')

    result = add_paper_to_finboard(paper)
    if result == "added":
        # Also record as a save signal for preference learning
        insert_signal(paper_id, "save")
        _maybe_recompute_preferences()
        return HTMLResponse(
            '<button class="btn btn-sm" disabled style="opacity:0.6">'
            '&#10003; Added to Finboard</button>'
        )
    elif result == "exists":
        return HTMLResponse(
            '<button class="btn btn-sm" disabled style="opacity:0.6">'
            'Already in Finboard</button>'
        )
    else:
        return HTMLResponse(
            '<span style="color:var(--red); font-size:0.82rem">'
            'Finboard DB not found</span>'
        )


# ---------------------------------------------------------------------------
# Paper2Video integration
# ---------------------------------------------------------------------------


def _video_generating_html(paper_id: int, stage_label: str) -> str:
    return (
        f'<div id="video-section" hx-get="/api/video/{paper_id}/status" '
        f'hx-trigger="every 10s" hx-swap="outerHTML">'
        f'<span class="badge badge--accent">Video generating... ({stage_label})</span>'
        f'</div>'
    )


def _video_done_html(paper_id: int) -> str:
    return (
        f'<div id="video-section">'
        f'<video controls width="100%" style="max-height:400px; border-radius:var(--radius); margin-top:0.75rem">'
        f'<source src="/api/video/{paper_id}/stream" type="video/mp4">'
        f'</video></div>'
    )


def _video_failed_html(paper_id: int, error: str) -> str:
    from markupsafe import escape
    return (
        f'<div id="video-section">'
        f'<span style="color:var(--red); font-size:0.82rem">Video failed: {escape(error)}</span> '
        f'<button class="btn btn-sm" hx-post="/api/video/{paper_id}" '
        f'hx-swap="outerHTML" hx-target="#video-section" hx-disabled-elt="this">'
        f'Retry</button></div>'
    )


@app.post("/api/video/{paper_id}", response_class=HTMLResponse)
async def generate_video(request: Request, paper_id: int):
    """Trigger video generation for a paper. Downloads PDF from arXiv, sends to Paper2Video API."""
    import asyncio
    import requests as req
    from src.config import PAPER2VIDEO_URL

    paper = get_paper(paper_id)
    if not paper:
        return HTMLResponse('<span style="color:var(--red)">Paper not found</span>')

    existing = get_video_job(paper_id)
    if existing and existing["status"] in ("queued", "running"):
        label = (existing.get("stage") or "queued").replace("_", " ").title()
        return HTMLResponse(_video_generating_html(paper_id, label))

    arxiv_id = paper.get("arxiv_id", "")
    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    loop = asyncio.get_event_loop()

    # Download PDF off the event loop
    try:
        pdf_resp = await loop.run_in_executor(None, lambda: req.get(pdf_url, timeout=60))
        pdf_resp.raise_for_status()
    except Exception as e:
        log.warning("Failed to download PDF for %s: %s", arxiv_id, e)
        return HTMLResponse('<span style="color:var(--red)">PDF download failed</span>')

    # Send to Paper2Video API off the event loop
    try:
        def _post():
            return req.post(
                f"{PAPER2VIDEO_URL}/jobs",
                files={"pdf": (f"{arxiv_id}.pdf", pdf_resp.content, "application/pdf")},
                data={"model": "bedrock-sonnet", "voice": "Ruth", "tts_engine": "polly"},
                timeout=30,
            )
        api_resp = await loop.run_in_executor(None, _post)
        api_resp.raise_for_status()
        job_id = api_resp.json()["job_id"]
    except Exception as e:
        log.warning("Paper2Video API call failed: %s", e)
        return HTMLResponse('<span style="color:var(--red)">Paper2Video service unavailable</span>')

    create_video_job(paper_id, job_id)

    return HTMLResponse(_video_generating_html(paper_id, "Queued"))


@app.get("/api/video/{paper_id}/status", response_class=HTMLResponse)
async def video_status(request: Request, paper_id: int):
    """Poll video job status. Returns HTMX partial."""
    import requests as req
    from src.config import PAPER2VIDEO_URL

    job = get_video_job(paper_id)
    if not job:
        return HTMLResponse(
            f'<div id="video-section">'
            f'<button class="btn btn-sm" hx-post="/api/video/{paper_id}" '
            f'hx-swap="outerHTML" hx-target="#video-section" hx-disabled-elt="this">'
            f'Generate Video</button></div>'
        )

    if job["status"] == "done":
        return HTMLResponse(_video_done_html(paper_id))

    if job["status"] == "failed":
        return HTMLResponse(_video_failed_html(paper_id, job.get("error", "unknown error")))

    # Active job — poll Paper2Video for latest status
    stage_label = (job.get("stage") or job["status"] or "running").replace("_", " ").title()
    try:
        resp = req.get(f"{PAPER2VIDEO_URL}/jobs/{job['job_id']}", timeout=5)
        if resp.ok:
            data = resp.json()
            status = data.get("status", job["status"])
            stage = data.get("stage", job.get("stage"))
            error = data.get("error")
            update_video_job(job["job_id"], status, stage, error)

            if status == "done":
                return HTMLResponse(_video_done_html(paper_id))
            if status == "failed":
                return HTMLResponse(_video_failed_html(paper_id, error or "unknown error"))

            stage_label = (stage or status or "running").replace("_", " ").title()
    except Exception:
        pass  # Use stage_label from DB fallback above

    return HTMLResponse(_video_generating_html(paper_id, stage_label))


@app.get("/api/video/{paper_id}/stream")
async def stream_video(paper_id: int):
    """Proxy the video from Paper2Video API."""
    import requests as req
    from src.config import PAPER2VIDEO_URL
    from starlette.responses import StreamingResponse

    job = get_video_job(paper_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "not ready"}, status_code=404)

    try:
        resp = req.get(f"{PAPER2VIDEO_URL}/jobs/{job['job_id']}/video", stream=True, timeout=30)
        resp.raise_for_status()

        def _iter_and_close():
            try:
                yield from resp.iter_content(chunk_size=65536)
            finally:
                resp.close()

        return StreamingResponse(
            _iter_and_close(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"inline; filename=paper2video_{paper_id}.mp4"},
        )
    except Exception as e:
        log.warning("Failed to stream video for paper %d: %s", paper_id, e)
        return JSONResponse({"error": "video stream failed"}, status_code=502)


# ---------------------------------------------------------------------------
# Preference signals
# ---------------------------------------------------------------------------


_pref_recompute_lock = threading.Lock()


def _maybe_recompute_preferences():
    """Recompute preferences if stale (>1 hour since last update)."""
    updated_at = get_preferences_updated_at()
    if updated_at:
        try:
            last = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            if age_hours < 1:
                return
        except (ValueError, AttributeError):
            pass
    # Skip if already recomputing
    if not _pref_recompute_lock.acquire(blocking=False):
        return

    def _run():
        try:
            compute_preferences()
        finally:
            _pref_recompute_lock.release()

    try:
        thread = threading.Thread(target=_run, name="pref-recompute")
        thread.start()
        with _pipeline_lock:
            _pipeline_threads.append(thread)
    except Exception:
        _pref_recompute_lock.release()
        log.exception("Failed to start preference recompute thread")


_SIGNAL_MESSAGES = {
    "upvote":   ("Upvoted \u2014 more like this",   "Removed upvote"),
    "downvote": ("Downvoted \u2014 less like this",  "Removed downvote"),
    "dismiss":  ("Dismissed",                         "Removed dismissal"),
    "save":     ("Saved",                             "Removed save"),
}


@app.post("/api/signal/{paper_id}/{action}", response_class=HTMLResponse)
async def record_signal(request: Request, paper_id: int, action: str):
    """Record a user signal. Returns HTMX partial with updated button state."""
    if action not in ("save", "upvote", "downvote", "dismiss"):
        return HTMLResponse("Invalid action", status_code=400)

    try:
        paper = get_paper(paper_id)
        if not paper:
            return HTMLResponse("Paper not found", status_code=404)

        # Toggle: if same signal exists, remove it
        current = get_paper_signal(paper_id)
        is_removal = current == action

        if is_removal:
            delete_signal(paper_id, action)
            new_signal = None
        else:
            # Remove conflicting signals (e.g., remove upvote if downvoting)
            for conflicting in ("upvote", "downvote", "dismiss"):
                if conflicting != action:
                    delete_signal(paper_id, conflicting)
            insert_signal(paper_id, action)
            new_signal = action

        _maybe_recompute_preferences()

        # Build toast message
        set_msg, remove_msg = _SIGNAL_MESSAGES.get(action, ("Signal set", "Signal removed"))
        toast_msg = remove_msg if is_removal else set_msg
        toast_type = "warning" if is_removal else "success"

        # Compute updated adjusted score for OOB swap
        preferences = load_preferences()
        from src.preferences import compute_paper_boost
        boost, reasons = compute_paper_boost(paper, preferences) if preferences else (0.0, [])
        composite = paper.get("composite") or 0
        adjusted_score = round(composite + boost, 2)

        ctx = {
            "request": request,
            "paper_id": paper_id,
            "user_signal": new_signal,
            "include_oob_score": True,
            "paper": {
                **paper,
                "preference_boost": boost,
                "adjusted_score": adjusted_score,
            },
        }

        response = templates.TemplateResponse("partials/signal_buttons.html", ctx)
        response.headers["HX-Trigger"] = json.dumps({
            "showSignalToast": {"message": toast_msg, "type": toast_type}
        })
        return response

    except Exception:
        log.exception("Error recording signal for paper %d", paper_id)
        error_response = templates.TemplateResponse("partials/signal_buttons.html", {
            "request": request,
            "paper_id": paper_id,
            "user_signal": get_paper_signal(paper_id),
        })
        error_response.headers["HX-Trigger"] = json.dumps({
            "showSignalToast": {"message": "Failed to record signal", "type": "error"}
        })
        return error_response


@app.get("/api/preferences")
async def api_preferences():
    """Return preference profile as JSON."""
    prefs = load_preferences()
    counts = get_signal_counts()
    return {"preferences": prefs, "signal_counts": counts}


@app.post("/api/preferences/recompute")
async def api_recompute_preferences():
    """Force recompute preferences."""
    prefs = compute_preferences()
    return {"status": "ok", "preference_count": len(prefs)}


@app.post("/api/preferences/reset")
async def api_reset_preferences():
    """Clear all signals and preferences."""
    clear_preferences()
    return {"status": "ok"}


@app.get("/preferences", response_class=HTMLResponse)
async def preferences_page(request: Request):
    """User preferences dashboard."""
    prefs_detail = get_preferences_detail()
    counts = get_signal_counts()
    updated_at = get_preferences_updated_at()

    # Group preferences by type
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in prefs_detail:
        prefix = p["pref_key"].split(":")[0]
        name = p["pref_key"].split(":", 1)[1] if ":" in p["pref_key"] else p["pref_key"]
        grouped[prefix].append({
            "name": name,
            "value": p["pref_value"],
            "count": p["signal_count"],
        })

    return templates.TemplateResponse("preferences.html", {
        "request": request,
        "active": "preferences",
        "grouped": dict(grouped),
        "signal_counts": counts,
        "updated_at": updated_at,
        "total_prefs": len(prefs_detail),
    })


# ---------------------------------------------------------------------------
# S2 enrichment trigger
# ---------------------------------------------------------------------------


@app.post("/run/enrich/{domain}")
async def trigger_enrich(domain: str):
    """Trigger Semantic Scholar enrichment for the latest run."""
    if domain not in ("aiml", "security"):
        return RedirectResponse("/", status_code=303)

    run = get_latest_run(domain)
    if not run:
        return RedirectResponse(f"/papers/{domain}", status_code=303)

    with _pipeline_lock:
        key = f"enrich-{domain}"
        if key in _running_pipelines:
            return RedirectResponse(f"/papers/{domain}", status_code=303)
        _running_pipelines.add(key)

        def _run():
            try:
                from src.pipelines.semantic_scholar import enrich_run
                enrich_run(run["id"], domain)
            except Exception as e:
                log.warning("S2 enrichment for %s failed: %s", domain, e)
            finally:
                with _pipeline_lock:
                    _running_pipelines.discard(key)

        try:
            thread = threading.Thread(target=_run, name=f"enrich-{domain}")
            thread.start()
            _pipeline_threads.append(thread)
        except Exception:
            _running_pipelines.discard(key)
            log.exception("Failed to start enrichment thread for %s", domain)
    return RedirectResponse(f"/papers/{domain}", status_code=303)


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    """First-time setup wizard."""
    return templates.TemplateResponse("setup.html", {
        "request": request,
    })


@app.post("/api/setup/validate-key")
async def validate_api_key(request: Request):
    """Validate an Anthropic API key with a test call."""
    try:
        body = await request.json()
        key = body.get("api_key", "").strip()
        if not key:
            return JSONResponse({"valid": False, "error": "No key provided"})

        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=15.0)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return JSONResponse({"valid": True})
    except Exception as e:
        import anthropic
        log.warning("API key validation failed: %s", e)
        if isinstance(e, anthropic.AuthenticationError):
            return JSONResponse({"valid": False, "error": "Invalid API key"})
        if isinstance(e, anthropic.APIConnectionError):
            return JSONResponse({"valid": False, "error": "Could not connect to Anthropic API"})
        return JSONResponse({"valid": False, "error": "Validation failed — check server logs"})


@app.post("/api/setup/save")
async def save_setup(request: Request):
    """Save setup wizard config to config.yaml and .env."""
    try:
        body = await request.json()
        api_key = body.get("api_key", "").strip()

        import src.config

        # Write API key to .env (never in config.yaml)
        if api_key:
            env_path = Path(".env")
            env_lines = []
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if not line.startswith("ANTHROPIC_API_KEY="):
                        env_lines.append(line)
            env_lines.append(f"ANTHROPIC_API_KEY={api_key}")
            env_path.write_text("\n".join(env_lines) + "\n")
            # Always set in current process
            os.environ["ANTHROPIC_API_KEY"] = api_key
            src.config.ANTHROPIC_API_KEY = api_key

        # Build config.yaml
        domains_data = body.get("domains", {})
        schedule_cron = body.get("schedule", "0 22 * * 0")

        # Scoring model settings
        scoring_data = body.get("scoring", {})
        scoring_block = {
            "model": scoring_data.get("model", "claude-haiku-4-5-20251001"),
            "rescore_model": scoring_data.get("rescore_model", "claude-sonnet-4-5-20250929"),
            "rescore_top_n": int(scoring_data.get("rescore_top_n", 15)),
            "batch_size": 20,
        }

        config_data = {
            "scoring": scoring_block,
            "domains": {
                "aiml": {
                    "enabled": domains_data.get("aiml", {}).get("enabled", True),
                    "label": "AI / ML",
                    "sources": ["huggingface", "arxiv"],
                    "arxiv_categories": ["cs.CV", "cs.CL", "cs.LG"],
                    "scoring_axes": _build_axes_config("aiml", domains_data),
                    "include_patterns": [],
                    "exclude_patterns": [],
                    "preferences": {"boost_topics": [], "penalize_topics": []},
                },
                "security": {
                    "enabled": domains_data.get("security", {}).get("enabled", True),
                    "label": "Security",
                    "sources": ["arxiv"],
                    "arxiv_categories": ["cs.CR"],
                    "scoring_axes": _build_axes_config("security", domains_data),
                    "include_patterns": [],
                    "exclude_patterns": [],
                    "preferences": {"boost_topics": [], "penalize_topics": []},
                },
            },
            "github": {"enabled": body.get("github", {}).get("enabled", True)},
            "events": {"enabled": body.get("events", {}).get("enabled", True)},
            "schedule": {"cron": schedule_cron} if schedule_cron else {"cron": ""},
            "database": {"path": str(src.config.DB_PATH)},
            "web": {"host": "0.0.0.0", "port": 8888},
        }

        from src.config import save_config
        save_config(config_data)

        # Update scheduler with new cron
        from src.scheduler import reschedule
        reschedule(schedule_cron)

        return JSONResponse({"status": "ok"})
    except Exception as e:
        log.exception("Setup save failed")
        return JSONResponse({"status": "error", "error": "Configuration save failed — check server logs"})


def _build_axes_config(domain: str, domains_data: dict) -> list[dict]:
    """Build scoring axes config from wizard form data."""
    d = domains_data.get(domain, {})
    weights = d.get("scoring_weights", [])

    if domain == "aiml":
        defaults = [
            {"name": "Code & Weights", "weight": 0.30, "description": "Open weights on HF, code on GitHub"},
            {"name": "Novelty", "weight": 0.35, "description": "Paradigm shifts over incremental"},
            {"name": "Practical Applicability", "weight": 0.35, "description": "Usable by practitioners soon"},
        ]
    else:
        defaults = [
            {"name": "Has Code/PoC", "weight": 0.25, "description": "Working tools, repos, artifacts"},
            {"name": "Novel Attack Surface", "weight": 0.40, "description": "First-of-kind research"},
            {"name": "Real-World Impact", "weight": 0.35, "description": "Affects production systems"},
        ]

    for i, ax in enumerate(defaults):
        if i < len(weights):
            ax["weight"] = round(weights[i], 2)

    return defaults


# ---------------------------------------------------------------------------
# Seed preferences
# ---------------------------------------------------------------------------


@app.get("/seed-preferences", response_class=HTMLResponse)
async def seed_preferences_page(request: Request):
    """Show seed papers for preference bootstrapping."""
    seed_path = Path("data/seed_papers.json")
    papers = []
    if seed_path.exists():
        papers = json.loads(seed_path.read_text())
    return templates.TemplateResponse("seed_preferences.html", {
        "request": request,
        "active": "preferences",
        "papers": papers,
    })


@app.post("/api/seed-preferences")
async def save_seed_preferences(request: Request):
    """Bulk-insert seed preference signals."""
    try:
        body = await request.json()
        ratings = body.get("ratings", {})

        # Filter to valid ratings only
        valid_ratings = {
            arxiv_id: action
            for arxiv_id, action in ratings.items()
            if action in ("upvote", "downvote")
        }

        if not valid_ratings:
            return JSONResponse({"status": "ok", "count": 0, "total_preferences": 0, "summary": {"boosted": [], "penalized": []}})

        # Load seed papers to get metadata for stub insertion
        seed_path = Path("data/seed_papers.json")
        seed_papers = []
        if seed_path.exists():
            seed_papers = json.loads(seed_path.read_text())

        # Build list of papers that were rated
        rated_papers = [p for p in seed_papers if p.get("arxiv_id") in valid_ratings]

        # Ensure all rated papers exist in DB (creates stubs if needed)
        from src.db import upsert_seed_papers
        id_map = upsert_seed_papers(rated_papers)

        # Insert signals using the paper IDs
        inserted = 0
        for arxiv_id, action in valid_ratings.items():
            paper_db_id = id_map.get(arxiv_id)
            if paper_db_id:
                insert_signal(paper_db_id, action)
                inserted += 1

        if inserted > 0:
            compute_preferences()

        # Return preference summary for observability
        prefs_detail = get_preferences_detail()
        summary = {"boosted": [], "penalized": []}
        for p in sorted(prefs_detail, key=lambda x: abs(x["pref_value"]), reverse=True):
            prefix = p["pref_key"].split(":")[0]
            name = p["pref_key"].split(":", 1)[1] if ":" in p["pref_key"] else p["pref_key"]
            if prefix in ("topic", "keyword", "category"):
                entry = {"type": prefix, "name": name, "value": round(p["pref_value"], 3), "signals": p["signal_count"]}
                if p["pref_value"] > 0.02:
                    summary["boosted"].append(entry)
                elif p["pref_value"] < -0.02:
                    summary["penalized"].append(entry)
        summary["boosted"] = summary["boosted"][:10]
        summary["penalized"] = summary["penalized"][:10]

        return JSONResponse({
            "status": "ok",
            "count": inserted,
            "total_preferences": len(prefs_detail),
            "summary": summary,
        })
    except Exception:
        log.exception("Seed preferences save failed")
        return JSONResponse(
            {"status": "error", "error": "Failed to save seed preferences — check server logs"},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(run_id: int, domain: str):
    """Generate a markdown report and save to data/weeks/."""
    from src.db import get_run
    run = get_run(run_id)
    if not run:
        return

    papers = get_top_papers(domain, run_id=run_id, limit=20)
    if not papers:
        return

    config = SCORING_CONFIGS[domain]
    axis_labels = config["axis_labels"]
    date_start = run["date_start"]
    date_end = run["date_end"]

    if domain == "aiml":
        title = f"AI/ML Research Weekly: {date_start} – {date_end}"
    else:
        title = f"Security Research Weekly: {date_start} – {date_end}"

    lines = [f"# {title}\n\n"]
    lines.append(f"> **{run.get('paper_count', len(papers))}** papers analyzed and scored.\n\n")

    # Top 5
    top5 = papers[:5]
    honorable = papers[5:20]

    lines.append("## Top Papers\n\n")
    for i, p in enumerate(top5, 1):
        authors = p.get("authors", [])
        if isinstance(authors, str):
            authors_str = authors
        elif len(authors) > 3:
            authors_str = ", ".join(authors[:3]) + " et al."
        else:
            authors_str = ", ".join(authors)

        lines.append(f"### {i}. {p['title']}\n\n")
        lines.append(f"**Authors:** {authors_str}\n")
        arxiv_id = p.get("arxiv_id", "")
        lines.append(f"**arXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n")
        if p.get("code_url"):
            lines.append(f"**Code:** [{p['code_url']}]({p['code_url']})\n")
        lines.append("\n")

        if p.get("summary"):
            lines.append(f"> {p['summary']}\n\n")

        lines.append("| Metric | Score | |\n|--------|-------|-|\n")
        for j, label in enumerate(axis_labels):
            val = p.get(f"score_axis_{j+1}", 0) or 0
            bar = score_bar(val)
            lines.append(f"| {label} | {val}/10 | `{bar}` |\n")
        comp = p.get("composite", 0) or 0
        lines.append(f"| **Composite** | **{comp}/10** | `{score_bar(comp)}` |\n\n")

        if p.get("reasoning"):
            lines.append(f"*{p['reasoning']}*\n\n")
        lines.append("---\n\n")

    # Honorable mentions
    if honorable:
        lines.append("## Honorable Mentions\n\n")
        lines.append("| # | Paper | Score | Summary |\n")
        lines.append("|---|-------|-------|---------|\n")
        for i, p in enumerate(honorable, 6):
            t = p["title"][:80].replace("|", "\\|")
            if len(p["title"]) > 80:
                t += "..."
            s = (p.get("summary") or "")[:120].replace("|", "\\|")
            if len(p.get("summary") or "") > 120:
                s += "..."
            aid = p.get("arxiv_id", "")
            lines.append(f"| {i} | [{t}](https://arxiv.org/abs/{aid}) | {p.get('composite', 0)} | {s} |\n")
        lines.append("\n")

    lines.append("---\n*Generated by Research Intelligence*\n")

    report = "".join(lines)

    weeks_dir = DATA_DIR / "weeks"
    weeks_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{date_start}-{domain}.md"
    (weeks_dir / filename).write_text(report)
    log.info("Report written to %s", weeks_dir / filename)
