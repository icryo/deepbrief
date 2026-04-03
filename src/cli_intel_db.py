"""Database layer — CLI Intelligence schema, connection, and query helpers."""

import json
import logging
from datetime import datetime, timezone

from src.db import get_conn, create_run, finish_run

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CLI_INTEL_SCHEMA = """\
CREATE TABLE IF NOT EXISTS cli_repos (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    repo_name TEXT NOT NULL,
    stars INTEGER DEFAULT 0,
    forks INTEGER DEFAULT 0,
    open_prs INTEGER DEFAULT 0,
    commits_30d INTEGER DEFAULT 0,
    contributors INTEGER DEFAULT 0,
    latest_release TEXT,
    release_date TEXT,
    description TEXT,
    language TEXT,
    tree_hash TEXT,
    snapshot_json TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(repo_name, run_id)
);

CREATE INDEX IF NOT EXISTS idx_cli_repos_run ON cli_repos(run_id);
CREATE INDEX IF NOT EXISTS idx_cli_repos_name ON cli_repos(repo_name);

CREATE TABLE IF NOT EXISTS cli_findings (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    repo_name TEXT NOT NULL,
    finding_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT,
    pr_number INTEGER,
    status TEXT,
    labels TEXT DEFAULT '[]',
    threat_level TEXT,
    score_threat REAL,
    score_novelty REAL,
    score_momentum REAL,
    composite REAL,
    summary TEXT,
    reasoning TEXT,
    first_seen TEXT,
    last_seen TEXT,
    is_new INTEGER DEFAULT 1,
    UNIQUE(repo_name, finding_type, title, run_id)
);

CREATE INDEX IF NOT EXISTS idx_cli_findings_run ON cli_findings(run_id);
CREATE INDEX IF NOT EXISTS idx_cli_findings_repo ON cli_findings(repo_name);
CREATE INDEX IF NOT EXISTS idx_cli_findings_type ON cli_findings(finding_type);
CREATE INDEX IF NOT EXISTS idx_cli_findings_composite ON cli_findings(composite DESC);
"""


# ---------------------------------------------------------------------------
# Sort options
# ---------------------------------------------------------------------------

CLI_SORT_OPTIONS = {
    "score": "composite DESC",
    "threat": "score_threat DESC",
    "novelty": "score_novelty DESC",
    "momentum": "score_momentum DESC",
    "repo": "repo_name ASC",
    "title": "title ASC",
    "type": "finding_type ASC",
}


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_cli_intel_db():
    """Create CLI intelligence tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(CLI_INTEL_SCHEMA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_json(val):
    """JSON-encode lists/dicts for storage."""
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return val


def _deserialize_finding(row) -> dict:
    """Convert a sqlite3.Row to a dict, parsing JSON fields."""
    d = dict(row)
    for key in ("labels",):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d[key] = []
    return d


def _get_latest_cli_run_id(conn) -> int | None:
    """Return the latest run ID that has cli_repos, or None."""
    row = conn.execute(
        "SELECT run_id FROM cli_repos ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


# ---------------------------------------------------------------------------
# Repo helpers
# ---------------------------------------------------------------------------

def insert_cli_repo(repo: dict, run_id: int):
    """Insert a repo snapshot into the DB."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO cli_repos
               (run_id, repo_name, stars, forks, open_prs, commits_30d,
                contributors, latest_release, release_date, description,
                language, tree_hash, snapshot_json, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                repo.get("repo_name", ""),
                repo.get("stars", 0),
                repo.get("forks", 0),
                repo.get("open_prs", 0),
                repo.get("commits_30d", 0),
                repo.get("contributors", 0),
                repo.get("latest_release"),
                repo.get("release_date"),
                repo.get("description", ""),
                repo.get("language", ""),
                repo.get("tree_hash"),
                _serialize_json(repo.get("snapshot_json")),
                now,
            ),
        )


def get_cli_repos(run_id: int | None = None) -> list[dict]:
    """Get repos for a run (latest if None)."""
    with get_conn() as conn:
        if not run_id:
            run_id = _get_latest_cli_run_id(conn)
            if not run_id:
                return []
        rows = conn.execute(
            "SELECT * FROM cli_repos WHERE run_id=? ORDER BY stars DESC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_previous_snapshot(repo_name: str) -> dict | None:
    """Get previous run's snapshot for diff comparison."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM cli_repos WHERE repo_name=? "
            "ORDER BY run_id DESC LIMIT 1 OFFSET 1",
            (repo_name,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Finding helpers
# ---------------------------------------------------------------------------

def insert_cli_findings(findings: list[dict], run_id: int):
    """Bulk-insert findings into the DB."""
    with get_conn() as conn:
        for f in findings:
            conn.execute(
                """INSERT OR IGNORE INTO cli_findings
                   (run_id, repo_name, finding_type, title, description,
                    url, pr_number, status, labels, threat_level,
                    score_threat, score_novelty, score_momentum, composite,
                    summary, reasoning, first_seen, last_seen, is_new)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    f.get("repo_name", ""),
                    f.get("finding_type", ""),
                    f.get("title", ""),
                    f.get("description", ""),
                    f.get("url", ""),
                    f.get("pr_number"),
                    f.get("status"),
                    _serialize_json(f.get("labels", [])),
                    f.get("threat_level"),
                    f.get("score_threat"),
                    f.get("score_novelty"),
                    f.get("score_momentum"),
                    f.get("composite"),
                    f.get("summary", ""),
                    f.get("reasoning", ""),
                    f.get("first_seen"),
                    f.get("last_seen"),
                    f.get("is_new", 1),
                ),
            )


def update_finding_scores(finding_id: int, scores: dict):
    """Update a finding's scores after Claude scoring."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE cli_findings SET
               score_threat=?, score_novelty=?, score_momentum=?,
               composite=?, threat_level=?, summary=?, reasoning=?
               WHERE id=?""",
            (
                scores.get("score_threat"),
                scores.get("score_novelty"),
                scores.get("score_momentum"),
                scores.get("composite"),
                scores.get("threat_level"),
                scores.get("summary", ""),
                scores.get("reasoning", ""),
                finding_id,
            ),
        )


def get_unscored_findings(run_id: int) -> list[dict]:
    """Get findings from a run that haven't been scored yet."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cli_findings WHERE run_id=? AND composite IS NULL",
            (run_id,),
        ).fetchall()
        return [_deserialize_finding(row) for row in rows]


def get_top_findings(run_id: int | None = None, limit: int = 10) -> list[dict]:
    """Get top-scored findings, optionally from a specific run."""
    with get_conn() as conn:
        if not run_id:
            run_id = _get_latest_cli_run_id(conn)
            if not run_id:
                return []
        rows = conn.execute(
            "SELECT * FROM cli_findings "
            "WHERE run_id=? AND composite IS NOT NULL "
            "ORDER BY composite DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [_deserialize_finding(row) for row in rows]


def get_cli_findings_page(
    run_id: int | None = None,
    offset: int = 0,
    limit: int = 50,
    repo: str | None = None,
    finding_type: str | None = None,
    threat_level: str | None = None,
    sort: str | None = None,
    new_only: bool = False,
) -> tuple[list[dict], int]:
    """Paginated, filterable findings list. Returns (findings, total_count)."""
    with get_conn() as conn:
        if not run_id:
            run_id = _get_latest_cli_run_id(conn)
            if not run_id:
                return [], 0

        conditions = ["run_id=?"]
        params: list = [run_id]

        if repo:
            conditions.append("repo_name=?")
            params.append(repo)

        if finding_type:
            conditions.append("finding_type=?")
            params.append(finding_type)

        if threat_level:
            conditions.append("threat_level=?")
            params.append(threat_level)

        if new_only:
            conditions.append("is_new=1")

        where = " AND ".join(conditions)
        order = CLI_SORT_OPTIONS.get(sort, "composite DESC")

        total = conn.execute(
            f"SELECT COUNT(*) FROM cli_findings WHERE {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT * FROM cli_findings WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return [_deserialize_finding(row) for row in rows], total


# ---------------------------------------------------------------------------
# Count helpers
# ---------------------------------------------------------------------------

def count_findings(run_id: int | None = None) -> int:
    """Count total findings for a run (latest if None)."""
    with get_conn() as conn:
        if not run_id:
            run_id = _get_latest_cli_run_id(conn)
            if not run_id:
                return 0
        return conn.execute(
            "SELECT COUNT(*) FROM cli_findings WHERE run_id=?", (run_id,)
        ).fetchone()[0]


def count_new_findings(run_id: int | None = None) -> int:
    """Count new findings for a run (latest if None)."""
    with get_conn() as conn:
        if not run_id:
            run_id = _get_latest_cli_run_id(conn)
            if not run_id:
                return 0
        return conn.execute(
            "SELECT COUNT(*) FROM cli_findings WHERE run_id=? AND is_new=1",
            (run_id,),
        ).fetchone()[0]


# ---------------------------------------------------------------------------
# Distinct repos in a run
# ---------------------------------------------------------------------------

def get_finding_repos(run_id: int) -> list[str]:
    """Get distinct repo names that have findings in a run."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT repo_name FROM cli_findings "
            "WHERE run_id=? ORDER BY repo_name",
            (run_id,),
        ).fetchall()
        return [row["repo_name"] for row in rows]
