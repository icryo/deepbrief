"""Finboard integration — add papers to the reading list.

Inserts directly into finboard's SQLite database as a fish in the
Backlog column with group_name='reading'.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

FINBOARD_DB = Path(os.environ.get(
    "FINBOARD_DB", "/home/a/finboard/finboard-data/aquarium.db"
))


def _score_to_fish(composite: float) -> tuple[int, str, str]:
    """Map paper composite score to fish (size, species, color)."""
    size = max(1, min(13, round(float(composite) * 1.3)))

    if composite >= 7:
        color = "Green"
    elif composite >= 4:
        color = "Yellow"
    else:
        color = "Blue"

    if size <= 2:
        species = "Guppy"
    elif size <= 4:
        species = "Tetra"
    elif size <= 6:
        species = "Angelfish"
    elif size <= 9:
        species = "Clownfish"
    elif size <= 12:
        species = "Shark"
    else:
        species = "Whale"

    return size, species, color


def add_paper_to_finboard(paper: dict) -> str:
    """Add a paper to finboard's reading list.

    Returns:
        "added" if successfully inserted,
        "exists" if paper URL already in finboard,
        "unavailable" if finboard DB not found.
    """
    if not FINBOARD_DB.exists():
        return "unavailable"

    arxiv_url = paper.get("arxiv_url", "")
    if not arxiv_url:
        arxiv_url = f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}"

    conn = sqlite3.connect(str(FINBOARD_DB))
    try:
        # Check by title prefix since schema may lack url column
        title = paper.get("title", "Untitled")
        short_title = title[:50]
        existing = conn.execute(
            "SELECT 1 FROM fish WHERE title=? AND group_name='reading'",
            (short_title,),
        ).fetchone()
        if existing:
            return "exists"

        composite = float(paper.get("composite", 5) or 5)
        size, species, color = _score_to_fish(composite)

        # Build description with paper + code links
        desc_parts = []
        if paper.get("summary"):
            desc_parts.append(paper["summary"])
        desc_parts.append("")
        desc_parts.append(f"Paper: {arxiv_url}")
        if paper.get("pdf_url"):
            desc_parts.append(f"PDF: {paper['pdf_url']}")
        if paper.get("code_url"):
            desc_parts.append(f"Code: {paper['code_url']}")
        elif paper.get("github_repo"):
            desc_parts.append(f"Code: {paper['github_repo']}")
        desc_parts.append(f"Score: {composite}/10")
        description = "\n".join(desc_parts)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fish_id = str(uuid.uuid4())

        conn.execute(
            """INSERT INTO fish
               (id, title, description, species, size, color, column_name,
                hunger, times_fed, created_at, last_fed, last_updated,
                tags, parent_id, group_name, swim_behavior)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fish_id,
                short_title,
                description,
                species,
                size,
                color,
                "Backlog",
                100,  # hunger (full)
                0,    # times_fed
                now,
                now,
                now,
                "[]",   # tags
                None,   # parent_id
                "reading",
                "Schooling",
            ),
        )
        conn.commit()
        return "added"
    finally:
        conn.close()


def is_in_finboard(arxiv_url: str, title: str = "") -> bool:
    """Check if a paper is already in finboard's reading list."""
    if not FINBOARD_DB.exists():
        return False
    conn = sqlite3.connect(str(FINBOARD_DB))
    try:
        # Check by title (truncated to 50 chars, matching insertion)
        if title:
            row = conn.execute(
                "SELECT 1 FROM fish WHERE title=? AND group_name='reading'",
                (title[:50],),
            ).fetchone()
            if row:
                return True
        # Also check description contains the URL
        if arxiv_url:
            row = conn.execute(
                "SELECT 1 FROM fish WHERE description LIKE ? AND group_name='reading'",
                (f"%{arxiv_url}%",),
            ).fetchone()
            return row is not None
        return False
    finally:
        conn.close()
