"""Security paper pipeline — ported from arxivrev/arxivrev.py.

Fetches security papers from arXiv (cs.CR + adjacent categories),
finds code URLs, and writes to the database.
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone

import arxiv
import requests

from src.config import (
    ADJACENT_CATEGORIES,
    GITHUB_TOKEN,
    GITHUB_URL_RE,
    SECURITY_EXCLUDE_RE,
    SECURITY_KEYWORDS,
    SECURITY_LLM_RE,
)
from src.db import create_run, finish_run, insert_papers

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# arXiv fetching
# ---------------------------------------------------------------------------


def fetch_arxiv_papers(start: datetime, end: datetime, max_papers: int) -> list[dict]:
    """Fetch papers from arXiv: all cs.CR + security-filtered adjacent categories.

    Uses a single combined query to minimize arXiv API requests.
    cs.CR papers are included unconditionally; adjacent categories
    are filtered for security-relevant keywords client-side.
    """
    all_cats = ["cs.CR"] + list(ADJACENT_CATEGORIES)
    query_str = " OR ".join(f"cat:{c}" for c in all_cats)

    client = arxiv.Client(page_size=500, delay_seconds=5.0, num_retries=5)
    query = arxiv.Search(
        query=query_str,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    log.info("Fetching security papers (combined query: %s) ...", query_str)

    papers: dict[str, dict] = {}
    adj_set = set(ADJACENT_CATEGORIES)
    cat_counts: dict[str, int] = {}

    for result in client.results(query):
        pub = result.published.replace(tzinfo=timezone.utc)
        if pub < start:
            break
        if pub > end:
            continue

        cats = list(result.categories)
        is_primary_cr = "cs.CR" in cats

        if not is_primary_cr:
            # Adjacent category — must match security keywords
            text = f"{result.title} {result.summary}"
            if not SECURITY_KEYWORDS.search(text):
                continue

        paper = _result_to_dict(result)
        if paper["entry_id"] not in papers:
            papers[paper["entry_id"]] = paper
            # Track which category contributed this paper
            if is_primary_cr:
                cat_counts["cs.CR"] = cat_counts.get("cs.CR", 0) + 1
            else:
                for c in cats:
                    if c in adj_set:
                        cat_counts[c] = cat_counts.get(c, 0) + 1
                        break

    # Log per-category counts
    for cat in all_cats:
        count = cat_counts.get(cat, 0)
        suffix = "" if cat == "cs.CR" else " security-relevant"
        if count > 0 or cat == "cs.CR":
            log.info("  %s: %d%s papers", cat, count, suffix)

    # Pre-filter: remove excluded topics (blockchain, surveys, etc.)
    before = len(papers)
    papers = {
        eid: p for eid, p in papers.items()
        if not SECURITY_EXCLUDE_RE.search(f"{p['title']} {p['abstract']}")
    }
    excluded = before - len(papers)
    if excluded:
        log.info("Excluded %d papers (blockchain/survey/off-topic)", excluded)

    # Tag LLM-adjacent papers so the scoring prompt can apply hard caps
    for p in papers.values():
        text = f"{p['title']} {p['abstract']}"
        p["llm_adjacent"] = bool(SECURITY_LLM_RE.search(text))

    llm_count = sum(1 for p in papers.values() if p["llm_adjacent"])
    if llm_count:
        log.info("Tagged %d papers as LLM-adjacent", llm_count)

    all_papers = list(papers.values())
    log.info("Total unique papers: %d", len(all_papers))
    return all_papers


def _result_to_dict(result: arxiv.Result) -> dict:
    """Convert an arxiv.Result to a plain dict."""
    arxiv_id = result.entry_id.split("/abs/")[-1]
    base_id = re.sub(r"v\d+$", "", arxiv_id)

    return {
        "arxiv_id": base_id,
        "entry_id": result.entry_id,
        "title": result.title.replace("\n", " ").strip(),
        "authors": [a.name for a in result.authors[:10]],
        "abstract": result.summary.replace("\n", " ").strip(),
        "published": result.published.isoformat(),
        "categories": list(result.categories),
        "pdf_url": result.pdf_url,
        "arxiv_url": result.entry_id,
        "comment": (result.comment or "").replace("\n", " ").strip(),
        "source": "arxiv",
        "github_repo": "",
        "github_stars": None,
        "hf_upvotes": 0,
        "hf_models": [],
        "hf_datasets": [],
        "hf_spaces": [],
    }


# ---------------------------------------------------------------------------
# Code URL finding
# ---------------------------------------------------------------------------


def extract_github_urls(paper: dict) -> list[str]:
    """Extract GitHub URLs from abstract and comments."""
    text = f"{paper['abstract']} {paper.get('comment', '')}"
    return list(set(GITHUB_URL_RE.findall(text)))


def search_github_for_paper(title: str, token: str | None) -> str | None:
    """Search GitHub for a repo matching the paper title."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    if token:
        try:
            resp = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=10)
            if resp.ok:
                remaining = resp.json().get("resources", {}).get("search", {}).get("remaining", 0)
                if remaining < 5:
                    return None
        except requests.RequestException:
            pass

    clean = re.sub(r"[^\w\s]", " ", title)
    words = clean.split()[:8]
    query = " ".join(words)

    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "updated", "per_page": 3},
            headers=headers,
            timeout=10,
        )
        if not resp.ok:
            return None
        items = resp.json().get("items", [])
        if items:
            return items[0]["html_url"]
    except requests.RequestException:
        pass
    return None


def find_code_urls(papers: list[dict]) -> dict[str, str | None]:
    """Find code/repo URLs for each paper."""
    token = GITHUB_TOKEN or None
    code_urls: dict[str, str | None] = {}

    for paper in papers:
        urls = extract_github_urls(paper)
        if urls:
            code_urls[paper["entry_id"]] = urls[0]
            continue

        url = search_github_for_paper(paper["title"], token)
        code_urls[paper["entry_id"]] = url
        if not token:
            time.sleep(2)

    return code_urls


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_security_pipeline(
    start: datetime | None = None,
    end: datetime | None = None,
    max_papers: int = 300,
) -> int:
    """Run the full security pipeline. Returns the run ID."""
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=7)

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)

    run_id = create_run("security", start.date().isoformat(), end.date().isoformat())
    log.info("Run %d: %s to %s", run_id, start.date(), end.date())

    try:
        # Step 1: Fetch papers
        papers = fetch_arxiv_papers(start, end, max_papers)

        if not papers:
            log.info("No papers found")
            finish_run(run_id, 0)
            return run_id

        # Step 2: Find code URLs
        log.info("Searching for code repositories ...")
        code_urls = find_code_urls(papers)
        with_code = sum(1 for v in code_urls.values() if v)
        log.info("Found code for %d/%d papers", with_code, len(papers))

        # Attach code URLs to papers as github_repo
        for paper in papers:
            url = code_urls.get(paper["entry_id"])
            if url:
                paper["github_repo"] = url

        # Step 3: Insert into DB
        insert_papers(papers, run_id, "security")
        finish_run(run_id, len(papers))
        log.info("Done — %d papers inserted", len(papers))
        return run_id

    except Exception as e:
        finish_run(run_id, 0, status="failed")
        log.exception("Pipeline failed")
        raise
