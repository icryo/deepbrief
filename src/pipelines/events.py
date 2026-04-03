"""Events pipeline — conferences, releases, and news.

Three sub-collectors:
1. Conferences: curated list + aideadlin.es scrape
2. Releases: HF trending models/spaces
3. News: RSS feeds from key AI/security blogs
"""

import logging
import time
from datetime import datetime, timezone

import feedparser
import requests

from src.config import CONFERENCES, HF_API, RSS_FEEDS
from src.db import insert_events

log = logging.getLogger(__name__)


def run_events_pipeline() -> int:
    """Run all event sub-collectors. Returns total events collected."""
    log.info("Starting events pipeline ...")
    all_events = []

    # 1. Conference deadlines
    conf_events = fetch_conference_deadlines()
    all_events.extend(conf_events)
    log.info("Conferences: %d", len(conf_events))

    # 2. HF trending releases
    release_events = fetch_hf_releases()
    all_events.extend(release_events)
    log.info("Releases: %d", len(release_events))

    # 3. RSS news
    news_events = fetch_rss_news()
    all_events.extend(news_events)
    log.info("News: %d", len(news_events))

    if all_events:
        insert_events(all_events)

    log.info("Done — %d total events", len(all_events))
    return len(all_events)


# ---------------------------------------------------------------------------
# Conferences
# ---------------------------------------------------------------------------


def fetch_conference_deadlines() -> list[dict]:
    """Return curated conference list as events + try aideadlin.es."""
    events = []

    # Static curated list
    for conf in CONFERENCES:
        deadline = conf.get("deadline", "")
        conf_date = conf.get("date", "")
        desc = conf.get("description", "")
        if deadline and conf_date:
            desc = f"{desc} Deadline: {deadline}. Conference: {conf_date}."
        elif deadline:
            desc = f"{desc} Deadline: {deadline}."
        elif conf_date:
            desc = f"{desc} Conference: {conf_date}."
        events.append({
            "category": "conference",
            "title": conf["name"],
            "description": desc,
            "url": conf["url"],
            "event_date": deadline or conf_date or "",
            "source": "curated",
        })

    # Try aideadlin.es for dynamic deadlines
    try:
        resp = requests.get("https://aideadlin.es/ai-deadlines.json", timeout=15)
        if resp.ok:
            deadlines = resp.json()
            for d in deadlines:
                if d.get("deadline", "TBA") == "TBA":
                    continue
                events.append({
                    "category": "conference",
                    "title": d.get("title", d.get("name", "")),
                    "description": d.get("full_name", ""),
                    "url": d.get("link", ""),
                    "event_date": d.get("deadline", ""),
                    "source": "aideadlin.es",
                })
    except (requests.RequestException, ValueError) as e:
        log.warning("aideadlin.es fetch failed: %s", e)

    return events


# ---------------------------------------------------------------------------
# HF/GitHub releases
# ---------------------------------------------------------------------------


def fetch_hf_releases() -> list[dict]:
    """Fetch trending models and spaces from HuggingFace."""
    events = []

    # Trending models
    try:
        resp = requests.get(
            f"{HF_API}/models",
            params={"sort": "trending", "limit": 15},
            timeout=15,
        )
        if resp.ok:
            for model in resp.json():
                events.append({
                    "category": "release",
                    "title": model.get("id", ""),
                    "description": f"Trending model — {model.get('likes', 0)} likes, "
                                   f"{model.get('downloads', 0)} downloads",
                    "url": f"https://huggingface.co/{model.get('id', '')}",
                    "event_date": model.get("lastModified", ""),
                    "source": "huggingface",
                    "relevance_score": None,
                })
    except (requests.RequestException, ValueError):
        pass

    time.sleep(0.5)

    # Trending spaces
    try:
        resp = requests.get(
            f"{HF_API}/spaces",
            params={"sort": "trending", "limit": 10},
            timeout=15,
        )
        if resp.ok:
            for space in resp.json():
                events.append({
                    "category": "release",
                    "title": f"Space: {space.get('id', '')}",
                    "description": f"Trending space — {space.get('likes', 0)} likes",
                    "url": f"https://huggingface.co/spaces/{space.get('id', '')}",
                    "event_date": space.get("lastModified", ""),
                    "source": "huggingface",
                    "relevance_score": None,
                })
    except (requests.RequestException, ValueError):
        pass

    return events


# ---------------------------------------------------------------------------
# RSS news
# ---------------------------------------------------------------------------


def fetch_rss_news() -> list[dict]:
    """Fetch recent entries from configured RSS feeds."""
    events = []

    for feed_config in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_config["url"])
            for entry in feed.entries[:5]:
                published = ""
                if hasattr(entry, "published"):
                    published = entry.published
                elif hasattr(entry, "updated"):
                    published = entry.updated

                events.append({
                    "category": "news",
                    "title": entry.get("title", ""),
                    "description": _clean_html(entry.get("summary", ""))[:300],
                    "url": entry.get("link", ""),
                    "event_date": published,
                    "source": feed_config["name"],
                    "relevance_score": None,
                })
        except Exception as e:
            log.warning("RSS fetch failed for %s: %s", feed_config['name'], e)
        time.sleep(0.3)

    return events


def _clean_html(text: str) -> str:
    """Strip HTML tags from text."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.replace("\n", " ").strip()
