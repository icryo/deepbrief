"""GitHub projects pipeline — discover trending repos via OSSInsight.io API.

Two strategies:
1. Trending repos — weekly trending filtered by AI/ML and security keywords
2. Collection rankings — curated collections ranked by star growth
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from src.config import (
    GITHUB_AIML_KEYWORDS,
    GITHUB_SECURITY_KEYWORDS,
    OSSINSIGHT_API,
    OSSINSIGHT_COLLECTIONS,
    OSSINSIGHT_TRENDING_LANGUAGES,
)
from src.db import create_run, finish_run, insert_github_projects

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers["Accept"] = "application/json"


def _safe_int(val, default=0) -> int:
    """Parse an int from a value that may be empty string or None."""
    if not val and val != 0:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0) -> float:
    if not val and val != 0:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _api_get(path: str, params: dict | None = None) -> list[dict]:
    """Make an OSSInsight API request and return the rows."""
    url = f"{OSSINSIGHT_API}{path}"
    try:
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return data.get("rows", [])
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("OSSInsight API error for %s: %s", path, e)
        return []


def _classify_domain(repo_name: str, description: str, collection_names: str = "") -> str | None:
    """Classify a repo into aiml, security, or None based on keywords."""
    text = f"{repo_name} {description} {collection_names}"
    if GITHUB_SECURITY_KEYWORDS.search(text):
        return "security"
    if GITHUB_AIML_KEYWORDS.search(text):
        return "aiml"
    return None


def fetch_trending_repos() -> list[dict]:
    """Fetch trending repos across configured languages for the past week."""
    seen: set[str] = set()
    projects: list[dict] = []

    # Also fetch "All" to catch cross-language breakouts
    languages = ["All"] + OSSINSIGHT_TRENDING_LANGUAGES

    for lang in languages:
        lang_param = lang if lang != "C++" else "C%2B%2B"
        rows = _api_get("/trends/repos", {"language": lang_param, "period": "past_week"})
        log.info("Trending %s: %d repos", lang, len(rows))

        for row in rows:
            repo_name = row.get("repo_name", "")
            if not repo_name or repo_name in seen:
                continue
            seen.add(repo_name)

            description = row.get("description", "") or ""
            collection_names = row.get("collection_names", "") or ""
            domain = _classify_domain(repo_name, description, collection_names)

            if domain is None:
                continue

            projects.append({
                "repo_id": _safe_int(row.get("repo_id")),
                "repo_name": repo_name,
                "description": description,
                "language": row.get("primary_language", "") or "",
                "stars": _safe_int(row.get("stars")),
                "forks": _safe_int(row.get("forks")),
                "pull_requests": _safe_int(row.get("pull_requests")),
                "total_score": _safe_float(row.get("total_score")),
                "collection_names": collection_names,
                "topics": [],
                "url": f"https://github.com/{repo_name}",
                "domain": domain,
            })

        time.sleep(0.5)

    return projects


def fetch_collection_rankings() -> list[dict]:
    """Fetch top repos from curated AI/ML and security collections."""
    seen: set[str] = set()
    projects: list[dict] = []

    for cid, (cname, domain) in OSSINSIGHT_COLLECTIONS.items():
        rows = _api_get(f"/collections/{cid}/ranking_by_stars", {"period": "past_28_days"})
        log.info("Collection '%s' (%d): %d repos", cname, cid, len(rows))

        for row in rows:
            repo_name = row.get("repo_name", "")
            if not repo_name or repo_name in seen:
                continue
            seen.add(repo_name)

            growth = _safe_int(row.get("current_period_growth"))
            if growth <= 0:
                continue

            projects.append({
                "repo_id": _safe_int(row.get("repo_id")),
                "repo_name": repo_name,
                "description": "",
                "language": "",
                "stars": growth,
                "forks": 0,
                "pull_requests": 0,
                "total_score": _safe_float(growth),
                "collection_names": cname,
                "topics": [],
                "url": f"https://github.com/{repo_name}",
                "domain": domain,
            })

        time.sleep(0.5)

    return projects


def run_github_pipeline() -> int:
    """Run the full GitHub projects pipeline. Returns run_id."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=7)).date().isoformat()
    end = now.date().isoformat()

    run_id = create_run("github", start, end)
    log.info("GitHub pipeline started — run %d (%s to %s)", run_id, start, end)

    try:
        # Strategy 1: Trending repos
        trending = fetch_trending_repos()
        log.info("Trending repos (filtered): %d", len(trending))

        # Strategy 2: Collection rankings
        collections = fetch_collection_rankings()
        log.info("Collection repos: %d", len(collections))

        # Merge — trending takes priority (has richer data)
        seen = {p["repo_name"] for p in trending}
        merged = list(trending)
        for p in collections:
            if p["repo_name"] not in seen:
                seen.add(p["repo_name"])
                merged.append(p)

        log.info("Total unique projects: %d", len(merged))

        if merged:
            insert_github_projects(merged, run_id)

        finish_run(run_id, len(merged))
        log.info("GitHub pipeline complete — %d projects stored", len(merged))
        return run_id

    except Exception:
        finish_run(run_id, 0, status="failed")
        log.exception("GitHub pipeline failed")
        raise
