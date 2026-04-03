"""AI/ML paper pipeline — ported from hfrev/fetch_papers.py.

Fetches papers from HuggingFace Daily Papers + arXiv, enriches with
HF ecosystem metadata, and writes to the database.
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone

import arxiv
import requests

from src.config import (
    ARXIV_LARGE_CATS,
    ARXIV_SMALL_CATS,
    EXCLUDE_RE,
    GITHUB_URL_RE,
    HF_API,
    HF_MAX_AGE_DAYS,
    INCLUDE_RE,
)
from src.db import create_run, finish_run, insert_papers

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace API
# ---------------------------------------------------------------------------


def fetch_hf_daily(date_str: str) -> list[dict]:
    """Fetch HF Daily Papers for a given date."""
    url = f"{HF_API}/daily_papers?date={date_str}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return []


def fetch_hf_trending(limit: int = 50) -> list[dict]:
    """Fetch HF trending papers."""
    url = f"{HF_API}/daily_papers?sort=trending&limit={limit}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return []


def arxiv_id_to_date(arxiv_id: str) -> datetime | None:
    """Extract approximate publication date from arXiv ID (YYMM.NNNNN)."""
    match = re.match(r"(\d{2})(\d{2})\.\d+", arxiv_id)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    if not (1 <= month <= 12):
        return None
    return datetime(year, month, 1, tzinfo=timezone.utc)


def normalize_hf_paper(hf_entry: dict) -> dict | None:
    """Convert an HF daily_papers entry to our normalized format.

    Returns None if the paper is too old.
    """
    paper = hf_entry.get("paper", hf_entry)
    arxiv_id = paper.get("id", "")

    authors_raw = paper.get("authors", [])
    authors = []
    for a in authors_raw:
        if isinstance(a, dict):
            name = a.get("name", a.get("user", {}).get("fullname", ""))
            if name:
                authors.append(name)
        elif isinstance(a, str):
            authors.append(a)

    github_repo = hf_entry.get("githubRepo") or paper.get("githubRepo") or ""

    pub_date = arxiv_id_to_date(arxiv_id)
    if pub_date and (datetime.now(timezone.utc) - pub_date).days > HF_MAX_AGE_DAYS:
        return None

    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title", "").replace("\n", " ").strip(),
        "authors": authors[:10],
        "abstract": paper.get("summary", paper.get("abstract", "")).replace("\n", " ").strip(),
        "published": paper.get("publishedAt", paper.get("published", "")),
        "categories": paper.get("categories", []),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        "comment": "",
        "source": "hf",
        "hf_upvotes": hf_entry.get("paper", {}).get("upvotes", hf_entry.get("upvotes", 0)),
        "github_repo": github_repo,
        "github_stars": None,
        "hf_models": [],
        "hf_datasets": [],
        "hf_spaces": [],
    }


# ---------------------------------------------------------------------------
# arXiv fetching
# ---------------------------------------------------------------------------


def fetch_arxiv_papers(
    start: datetime,
    end: datetime,
    max_results: int = 2000,
) -> list[dict]:
    """Fetch papers from all AI/ML categories in a single combined query.

    Large categories (cs.CV, cs.CL, cs.LG) get keyword filtering;
    small categories (eess.AS, cs.SD) are included unfiltered.
    """
    all_cats = ARXIV_LARGE_CATS + ARXIV_SMALL_CATS
    query_str = " OR ".join(f"cat:{c}" for c in all_cats)

    client = arxiv.Client(page_size=500, delay_seconds=5.0, num_retries=5)
    query = arxiv.Search(
        query=query_str,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    large_set = set(ARXIV_LARGE_CATS)
    small_set = set(ARXIV_SMALL_CATS)
    papers = []
    cat_counts: dict[str, int] = {}

    for result in client.results(query):
        pub = result.published.replace(tzinfo=timezone.utc)
        if pub < start:
            break
        if pub > end:
            continue

        cats = list(result.categories)
        # Papers in any small category pass through unfiltered (matches prior behavior)
        in_small = any(c in small_set for c in cats)
        in_large = any(c in large_set for c in cats)

        if in_large and not in_small:
            text = f"{result.title} {result.summary}"
            if not INCLUDE_RE.search(text):
                continue
            if EXCLUDE_RE.search(text):
                continue

        paper = _arxiv_result_to_dict(result)
        papers.append(paper)

        # Track counts per category for logging
        for c in cats:
            if c in all_cats:
                cat_counts[c] = cat_counts.get(c, 0) + 1

    for cat in all_cats:
        count = cat_counts.get(cat, 0)
        filtered = "(keyword-filtered)" if cat in large_set else ""
        log.info("  %s: %d papers %s", cat, count, filtered)

    return papers


def _arxiv_result_to_dict(result: arxiv.Result) -> dict:
    """Convert an arxiv.Result to our normalized format."""
    arxiv_id = result.entry_id.split("/abs/")[-1]
    base_id = re.sub(r"v\d+$", "", arxiv_id)

    github_urls = GITHUB_URL_RE.findall(f"{result.summary} {result.comment or ''}")
    github_repo = github_urls[0].rstrip(".") if github_urls else ""

    return {
        "arxiv_id": base_id,
        "title": result.title.replace("\n", " ").strip(),
        "authors": [a.name for a in result.authors[:10]],
        "abstract": result.summary.replace("\n", " ").strip(),
        "published": result.published.isoformat(),
        "categories": list(result.categories),
        "pdf_url": result.pdf_url,
        "arxiv_url": result.entry_id,
        "comment": (result.comment or "").replace("\n", " ").strip(),
        "source": "arxiv",
        "hf_upvotes": 0,
        "github_repo": github_repo,
        "github_stars": None,
        "hf_models": [],
        "hf_datasets": [],
        "hf_spaces": [],
    }


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_paper(paper: dict) -> dict:
    """Query HF API for linked models, datasets, and spaces."""
    arxiv_id = paper["arxiv_id"]
    if not arxiv_id:
        return paper

    base_id = re.sub(r"v\d+$", "", arxiv_id)

    for resource, key, limit in [
        ("models", "hf_models", 5),
        ("datasets", "hf_datasets", 3),
        ("spaces", "hf_spaces", 3),
    ]:
        url = f"{HF_API}/{resource}?filter=arxiv:{base_id}&limit={limit}&sort=likes"
        try:
            resp = requests.get(url, timeout=15)
            if resp.ok:
                items = resp.json()
                paper[key] = [
                    {"id": item.get("id", item.get("_id", "")), "likes": item.get("likes", 0)}
                    for item in items
                ]
        except (requests.RequestException, ValueError):
            pass

    time.sleep(0.2)
    return paper


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_papers(hf_papers: list[dict], arxiv_papers: list[dict]) -> list[dict]:
    """Deduplicate by arXiv ID. When both sources have a paper, merge."""
    by_id: dict[str, dict] = {}

    for p in arxiv_papers:
        aid = re.sub(r"v\d+$", "", p["arxiv_id"])
        if aid:
            by_id[aid] = p

    for p in hf_papers:
        aid = re.sub(r"v\d+$", "", p["arxiv_id"])
        if not aid:
            continue
        if aid in by_id:
            existing = by_id[aid]
            existing["source"] = "both"
            existing["hf_upvotes"] = max(existing.get("hf_upvotes", 0), p.get("hf_upvotes", 0))
            if p.get("github_repo") and not existing.get("github_repo"):
                existing["github_repo"] = p["github_repo"]
            if not existing.get("categories") and p.get("categories"):
                existing["categories"] = p["categories"]
        else:
            by_id[aid] = p

    return list(by_id.values())


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_aiml_pipeline(
    start: datetime | None = None,
    end: datetime | None = None,
    max_papers: int = 300,
    skip_enrich: bool = False,
) -> int:
    """Run the full AI/ML pipeline. Returns the run ID."""
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=7)

    # Ensure timezone-aware
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)

    run_id = create_run("aiml", start.date().isoformat(), end.date().isoformat())
    log.info("Run %d: %s to %s", run_id, start.date(), end.date())

    try:
        # Step 1: Fetch HF papers
        log.info("Fetching HuggingFace Daily Papers ...")
        hf_papers_raw = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily = fetch_hf_daily(date_str)
            hf_papers_raw.extend(daily)
            current += timedelta(days=1)

        trending = fetch_hf_trending(limit=50)
        hf_papers_raw.extend(trending)

        hf_papers = [p for p in (normalize_hf_paper(e) for e in hf_papers_raw) if p is not None]
        log.info("HF papers: %d", len(hf_papers))

        # Step 2: Fetch arXiv papers (single combined query)
        log.info("Fetching arXiv papers ...")
        arxiv_papers = fetch_arxiv_papers(start, end, max_results=max_papers)

        # Step 3: Merge
        all_papers = merge_papers(hf_papers, arxiv_papers)
        log.info("Merged: %d unique papers", len(all_papers))

        # Step 4: Enrich
        if not skip_enrich:
            log.info("Enriching with HF ecosystem links ...")
            for i, paper in enumerate(all_papers):
                all_papers[i] = enrich_paper(paper)
                if (i + 1) % 25 == 0:
                    log.info("  Enriched %d/%d ...", i + 1, len(all_papers))
            log.info("Enrichment complete")

        # Step 5: Insert into DB
        insert_papers(all_papers, run_id, "aiml")
        finish_run(run_id, len(all_papers))
        log.info("Done — %d papers inserted", len(all_papers))
        return run_id

    except Exception as e:
        finish_run(run_id, 0, status="failed")
        log.exception("Pipeline failed")
        raise
