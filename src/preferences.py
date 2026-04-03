"""Preference engine — learns from user signals to personalize paper rankings.

Adds a preference_boost (max +3.0 / min -2.0) on top of stored composite scores.
Never re-scores papers. Papers with composite >= 8 are never penalized.
"""

import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone

from src.db import (
    get_all_signals_with_papers,
    load_preferences,
    save_preferences,
    get_paper_signals_batch,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal weights
# ---------------------------------------------------------------------------

SIGNAL_WEIGHTS = {
    "save": 3.0,
    "upvote": 2.0,
    "view": 0.5,
    "downvote": -2.0,
    "dismiss": -1.5,
}

HALF_LIFE_DAYS = 60.0

# Dimension weights for combining into final boost
DIMENSION_WEIGHTS = {
    "topic": 0.35,
    "axis": 0.25,
    "keyword": 0.15,
    "category": 0.15,
    "author": 0.10,
}

# Scaling factors for tanh normalization (tuned per dimension)
SCALING_FACTORS = {
    "topic": 5.0,
    "axis": 4.0,
    "keyword": 8.0,
    "category": 5.0,
    "author": 6.0,
}

# Stopwords for keyword extraction from titles
_STOPWORDS = frozenset(
    "a an the and or but in on of for to with from by at is are was were "
    "be been being have has had do does did will would shall should may might "
    "can could this that these those it its we our their".split()
)

_WORD_RE = re.compile(r"[a-z]{3,}", re.IGNORECASE)


def _extract_keywords(title: str) -> list[str]:
    """Extract meaningful keywords from a paper title."""
    words = _WORD_RE.findall(title.lower())
    return [w for w in words if w not in _STOPWORDS]


def _time_decay(created_at: str) -> float:
    """Compute time decay factor: 2^(-age_days / half_life)."""
    try:
        signal_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.5
    now = datetime.now(timezone.utc)
    age_days = max(0, (now - signal_dt).total_seconds() / 86400)
    return math.pow(2, -age_days / HALF_LIFE_DAYS)


# ---------------------------------------------------------------------------
# Preference computation
# ---------------------------------------------------------------------------

def compute_preferences() -> dict[str, float]:
    """Compute user preference profile from all signals.

    Returns the preference dict (also saved to DB).
    """
    signals = get_all_signals_with_papers()
    if not signals:
        save_preferences({})
        return {}

    # Accumulate raw scores per preference key
    raw: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    # For axis preferences: track domain means
    axis_sums: dict[str, list[float]] = defaultdict(list)

    for sig in signals:
        base_weight = SIGNAL_WEIGHTS.get(sig["action"], 0)
        decay = _time_decay(sig["created_at"])
        weight = base_weight * decay

        # Topics
        topics = sig.get("topics") or []
        if topics:
            per_topic = weight / len(topics)
            for t in topics:
                key = f"topic:{t}"
                raw[key] += per_topic
                counts[key] += 1

        # Categories
        categories = sig.get("categories") or []
        if categories:
            per_cat = weight / len(categories)
            for c in categories:
                key = f"category:{c}"
                raw[key] += per_cat
                counts[key] += 1

        # Keywords from title
        keywords = _extract_keywords(sig.get("title", ""))
        if keywords:
            per_kw = weight / len(keywords)
            for kw in keywords:
                key = f"keyword:{kw}"
                raw[key] += per_kw
                counts[key] += 1

        # Authors (first 3 only)
        authors = sig.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        for author in authors[:3]:
            name = author if isinstance(author, str) else str(author)
            key = f"author:{name}"
            raw[key] += weight * 0.5  # reduced weight for authors
            counts[key] += 1

        # Axis preferences (track which axes are high on liked papers)
        domain = sig.get("domain", "")
        for i in range(1, 4):
            axis_val = sig.get(f"score_axis_{i}")
            if axis_val is not None:
                axis_sums[f"{domain}:axis{i}"].append(axis_val)

    # Compute axis preferences relative to domain mean
    for sig in signals:
        base_weight = SIGNAL_WEIGHTS.get(sig["action"], 0)
        if base_weight <= 0:
            continue  # Only positive signals inform axis preferences
        decay = _time_decay(sig["created_at"])
        weight = base_weight * decay
        domain = sig.get("domain", "")

        for i in range(1, 4):
            axis_val = sig.get(f"score_axis_{i}")
            mean_key = f"{domain}:axis{i}"
            if axis_val is not None and axis_sums.get(mean_key):
                mean = sum(axis_sums[mean_key]) / len(axis_sums[mean_key])
                deviation = axis_val - mean
                key = f"axis_pref:{domain}:axis{i}"
                raw[key] += deviation * weight * 0.1
                counts[key] += 1

    # Normalize via tanh
    prefs: dict[str, tuple[float, int]] = {}
    for key, value in raw.items():
        prefix = key.split(":")[0]
        scale = SCALING_FACTORS.get(prefix, 5.0)
        normalized = math.tanh(value / scale)
        # Clamp to [-1, 1]
        normalized = max(-1.0, min(1.0, normalized))
        prefs[key] = (round(normalized, 4), counts[key])

    save_preferences(prefs)
    return {k: v for k, (v, _) in prefs.items()}


# ---------------------------------------------------------------------------
# Paper boost computation
# ---------------------------------------------------------------------------

def compute_paper_boost(paper: dict, preferences: dict[str, float]) -> tuple[float, list[str]]:
    """Compute preference boost for a single paper.

    Returns (boost_value, list_of_reasons).
    Boost is clamped to [-2.0, +3.0].
    Papers with composite >= 8 are never penalized (boost >= 0).
    """
    if not preferences:
        return 0.0, []

    scores: dict[str, float] = {}
    reasons: list[str] = []

    # Topic match
    topics = paper.get("topics") or []
    if topics:
        topic_scores = []
        for t in topics:
            key = f"topic:{t}"
            if key in preferences:
                topic_scores.append((t, preferences[key]))
        if topic_scores:
            scores["topic"] = sum(v for _, v in topic_scores) / len(topic_scores)
            for name, val in sorted(topic_scores, key=lambda x: abs(x[1]), reverse=True)[:2]:
                if abs(val) > 0.05:
                    reasons.append(f"Topic: {name} {val:+.2f}")

    # Category match
    categories = paper.get("categories") or []
    if categories:
        cat_scores = []
        for c in categories:
            key = f"category:{c}"
            if key in preferences:
                cat_scores.append((c, preferences[key]))
        if cat_scores:
            scores["category"] = sum(v for _, v in cat_scores) / len(cat_scores)
            for name, val in sorted(cat_scores, key=lambda x: abs(x[1]), reverse=True)[:1]:
                if abs(val) > 0.05:
                    reasons.append(f"Category: {name} {val:+.2f}")

    # Keyword match
    keywords = _extract_keywords(paper.get("title", ""))
    if keywords:
        kw_scores = []
        for kw in keywords:
            key = f"keyword:{kw}"
            if key in preferences:
                kw_scores.append((kw, preferences[key]))
        if kw_scores:
            scores["keyword"] = sum(v for _, v in kw_scores) / len(kw_scores)
            for name, val in sorted(kw_scores, key=lambda x: abs(x[1]), reverse=True)[:1]:
                if abs(val) > 0.1:
                    reasons.append(f"Keyword: {name} {val:+.2f}")

    # Axis alignment
    domain = paper.get("domain", "")
    axis_scores = []
    for i in range(1, 4):
        key = f"axis_pref:{domain}:axis{i}"
        if key in preferences:
            axis_val = paper.get(f"score_axis_{i}")
            if axis_val is not None:
                # Higher axis value * positive preference = boost
                axis_scores.append(preferences[key] * (axis_val / 10.0))
    if axis_scores:
        scores["axis"] = sum(axis_scores) / len(axis_scores)

    # Author match
    authors = paper.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    author_scores = []
    for author in authors[:5]:
        name = author if isinstance(author, str) else str(author)
        key = f"author:{name}"
        if key in preferences:
            author_scores.append((name.split()[-1] if " " in name else name, preferences[key]))
    if author_scores:
        scores["author"] = max(v for _, v in author_scores)  # Best author match
        for name, val in sorted(author_scores, key=lambda x: abs(x[1]), reverse=True)[:1]:
            if abs(val) > 0.1:
                reasons.append(f"Author: {name} {val:+.2f}")

    # Weighted combine
    if not scores:
        return 0.0, []

    boost = 0.0
    total_weight = 0.0
    for dim, dim_score in scores.items():
        w = DIMENSION_WEIGHTS.get(dim, 0.1)
        boost += dim_score * w
        total_weight += w

    if total_weight > 0:
        boost = boost / total_weight  # Normalize by actual weight used

    # Scale to boost range: preferences are [-1, 1], we want [-2, 3]
    boost = boost * 3.0

    # Clamp
    boost = max(-2.0, min(3.0, boost))

    # Safety net: high-scoring papers never penalized
    composite = paper.get("composite") or 0
    if composite >= 8 and boost < 0:
        boost = 0.0

    return round(boost, 2), reasons


def is_discovery(paper: dict, boost: float) -> bool:
    """Paper is 'discovery' if composite >= 6 AND boost <= 0."""
    composite = paper.get("composite") or 0
    return composite >= 6 and boost <= 0


def enrich_papers_with_preferences(
    papers: list[dict],
    preferences: dict[str, float] | None = None,
    sort_adjusted: bool = False,
) -> list[dict]:
    """Add preference fields to each paper dict.

    Adds: adjusted_score, preference_boost, boost_reasons, is_discovery, user_signal.
    """
    if preferences is None:
        preferences = load_preferences()

    # Batch fetch user signals
    paper_ids = [p["id"] for p in papers if "id" in p]
    signals_map = get_paper_signals_batch(paper_ids) if paper_ids else {}

    has_prefs = bool(preferences)

    for p in papers:
        pid = p.get("id")
        composite = p.get("composite") or 0

        if has_prefs:
            boost, reasons = compute_paper_boost(p, preferences)
        else:
            boost, reasons = 0.0, []

        p["preference_boost"] = boost
        p["adjusted_score"] = round(composite + boost, 2)
        p["boost_reasons"] = reasons
        p["is_discovery"] = is_discovery(p, boost) if has_prefs else False
        p["user_signal"] = signals_map.get(pid)

    if sort_adjusted and has_prefs:
        papers.sort(key=lambda p: p.get("adjusted_score", 0), reverse=True)

    return papers
