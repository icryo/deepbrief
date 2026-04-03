"""Database layer — SQLite schema, connection, and query helpers."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

def get_db_path() -> Path:
    from src.config import DB_PATH
    return DB_PATH


@contextmanager
def get_conn():
    """Yield a SQLite connection with WAL mode and foreign keys."""
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("Database transaction failed")
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                    pass  # Expected — column/index already exists
                else:
                    log.warning("Migration failed: %s — %s", sql.strip()[:60], e)
        # Rebuild FTS index from content table (idempotent, fast for a few thousand rows)
        conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")


SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    paper_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    domain TEXT NOT NULL,
    arxiv_id TEXT NOT NULL,
    entry_id TEXT,
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    published TEXT,
    categories TEXT,
    pdf_url TEXT,
    arxiv_url TEXT,
    comment TEXT,
    source TEXT,
    github_repo TEXT,
    github_stars INTEGER,
    hf_upvotes INTEGER DEFAULT 0,
    hf_models TEXT,
    hf_datasets TEXT,
    hf_spaces TEXT,
    score_axis_1 REAL,
    score_axis_2 REAL,
    score_axis_3 REAL,
    composite REAL,
    summary TEXT,
    reasoning TEXT,
    code_url TEXT,
    UNIQUE(domain, arxiv_id, run_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT,
    event_date TEXT,
    source TEXT,
    relevance_score REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_connections (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    connected_arxiv_id TEXT,
    connected_s2_id TEXT,
    connected_title TEXT,
    connected_year INTEGER,
    connection_type TEXT NOT NULL,
    in_db_paper_id INTEGER,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_domain_composite
    ON papers(domain, composite DESC);
CREATE INDEX IF NOT EXISTS idx_papers_run ON papers(run_id);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category, event_date);
CREATE INDEX IF NOT EXISTS idx_connections_paper ON paper_connections(paper_id);
CREATE INDEX IF NOT EXISTS idx_connections_arxiv ON paper_connections(connected_arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);

CREATE TABLE IF NOT EXISTS github_projects (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    repo_id INTEGER NOT NULL,
    repo_name TEXT NOT NULL,
    description TEXT,
    language TEXT,
    stars INTEGER DEFAULT 0,
    forks INTEGER DEFAULT 0,
    pull_requests INTEGER DEFAULT 0,
    total_score REAL DEFAULT 0,
    collection_names TEXT,
    topics TEXT DEFAULT '[]',
    url TEXT NOT NULL,
    domain TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(repo_name, run_id)
);

CREATE INDEX IF NOT EXISTS idx_gh_run ON github_projects(run_id);
CREATE INDEX IF NOT EXISTS idx_gh_domain ON github_projects(domain, total_score DESC);
CREATE INDEX IF NOT EXISTS idx_gh_repo ON github_projects(repo_name);

CREATE TABLE IF NOT EXISTS user_signals (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    action TEXT NOT NULL CHECK(action IN ('save','view','upvote','downvote','dismiss')),
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_paper_action
    ON user_signals(paper_id, action) WHERE action != 'view';
CREATE INDEX IF NOT EXISTS idx_signals_created ON user_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_paper ON user_signals(paper_id);

CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY,
    pref_key TEXT NOT NULL UNIQUE,
    pref_value REAL NOT NULL DEFAULT 0.0,
    signal_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prefs_key ON user_preferences(pref_key);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, summary, topics,
    content='papers', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract, summary, topics)
    VALUES (new.id, new.title, new.abstract, new.summary, new.topics);
END;

CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, summary, topics)
    VALUES ('delete', old.id, old.title, old.abstract, old.summary, old.topics);
END;

CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, summary, topics)
    VALUES ('delete', old.id, old.title, old.abstract, old.summary, old.topics);
    INSERT INTO papers_fts(rowid, title, abstract, summary, topics)
    VALUES (new.id, new.title, new.abstract, new.summary, new.topics);
END;
"""

# Columns added after initial schema — idempotent via try/except
_MIGRATIONS = [
    "ALTER TABLE papers ADD COLUMN s2_tldr TEXT",
    "ALTER TABLE papers ADD COLUMN s2_paper_id TEXT",
    "ALTER TABLE papers ADD COLUMN topics TEXT DEFAULT '[]'",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique ON events(title, category)",
    # Prevent duplicate seed papers (NULL run_id) for the same arxiv_id+domain
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_seed_dedup ON papers(domain, arxiv_id) WHERE run_id IS NULL",
    # Video jobs table
    """CREATE TABLE IF NOT EXISTS video_jobs (
        id INTEGER PRIMARY KEY,
        paper_id INTEGER NOT NULL REFERENCES papers(id),
        job_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        stage TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        error TEXT,
        UNIQUE(paper_id, job_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_video_jobs_paper ON video_jobs(paper_id)",
]


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def create_run(domain: str, date_start: str, date_end: str) -> int:
    """Insert a new pipeline run, return its ID."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (domain, started_at, date_start, date_end, status) "
            "VALUES (?, ?, ?, ?, 'running')",
            (domain, now, date_start, date_end),
        )
        return cur.lastrowid


def finish_run(run_id: int, paper_count: int, status: str = "completed"):
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET finished_at=?, paper_count=?, status=? WHERE id=?",
            (now, paper_count, status, run_id),
        )


def get_latest_run(domain: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE domain=? ORDER BY id DESC LIMIT 1",
            (domain,),
        ).fetchone()
        return dict(row) if row else None


def get_run(run_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Paper helpers
# ---------------------------------------------------------------------------

def _serialize_json(val):
    """JSON-encode lists/dicts for storage."""
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    return val


def insert_papers(papers: list[dict], run_id: int, domain: str):
    """Bulk-insert papers into the DB."""
    with get_conn() as conn:
        for p in papers:
            conn.execute(
                """INSERT OR IGNORE INTO papers
                   (run_id, domain, arxiv_id, entry_id, title, authors, abstract,
                    published, categories, pdf_url, arxiv_url, comment, source,
                    github_repo, github_stars, hf_upvotes, hf_models, hf_datasets, hf_spaces)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, domain,
                    p.get("arxiv_id", ""),
                    p.get("entry_id", ""),
                    p.get("title", ""),
                    _serialize_json(p.get("authors", [])),
                    p.get("abstract", ""),
                    p.get("published", ""),
                    _serialize_json(p.get("categories", [])),
                    p.get("pdf_url", ""),
                    p.get("arxiv_url", ""),
                    p.get("comment", ""),
                    p.get("source", ""),
                    p.get("github_repo", ""),
                    p.get("github_stars"),
                    p.get("hf_upvotes", 0),
                    _serialize_json(p.get("hf_models", [])),
                    _serialize_json(p.get("hf_datasets", [])),
                    _serialize_json(p.get("hf_spaces", [])),
                ),
            )


def update_paper_scores(paper_id: int, scores: dict):
    """Update a paper's scores after Claude scoring."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE papers SET
               score_axis_1=?, score_axis_2=?, score_axis_3=?,
               composite=?, summary=?, reasoning=?, code_url=?
               WHERE id=?""",
            (
                scores.get("score_axis_1"),
                scores.get("score_axis_2"),
                scores.get("score_axis_3"),
                scores.get("composite"),
                scores.get("summary", ""),
                scores.get("reasoning", ""),
                scores.get("code_url"),
                paper_id,
            ),
        )


def get_unscored_papers(run_id: int) -> list[dict]:
    """Get papers from a run that haven't been scored yet."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE run_id=? AND composite IS NULL",
            (run_id,),
        ).fetchall()
        return [_deserialize_paper(row) for row in rows]


def get_top_papers(domain: str, run_id: int | None = None, limit: int = 20) -> list[dict]:
    """Get top-scored papers for a domain, optionally from a specific run."""
    with get_conn() as conn:
        if run_id:
            rows = conn.execute(
                "SELECT * FROM papers WHERE domain=? AND run_id=? AND composite IS NOT NULL "
                "ORDER BY composite DESC LIMIT ?",
                (domain, run_id, limit),
            ).fetchall()
        else:
            # Latest run
            latest = get_latest_run(domain)
            if not latest:
                return []
            rows = conn.execute(
                "SELECT * FROM papers WHERE domain=? AND run_id=? AND composite IS NOT NULL "
                "ORDER BY composite DESC LIMIT ?",
                (domain, latest["id"], limit),
            ).fetchall()
        return [_deserialize_paper(row) for row in rows]


def get_paper(paper_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        return _deserialize_paper(row) if row else None


SORT_OPTIONS = {
    "score": "composite DESC",
    "date": "published DESC",
    "axis1": "score_axis_1 DESC",
    "axis2": "score_axis_2 DESC",
    "axis3": "score_axis_3 DESC",
    "title": "title ASC",
}


def get_papers_page(domain: str, run_id: int | None = None,
                    offset: int = 0, limit: int = 50,
                    min_score: float | None = None,
                    has_code: bool | None = None,
                    search: str | None = None,
                    topic: str | None = None,
                    sort: str | None = None) -> tuple[list[dict], int]:
    """Paginated, filterable paper list. Returns (papers, total_count)."""
    with get_conn() as conn:
        if not run_id:
            latest = get_latest_run(domain)
            if not latest:
                return [], 0
            run_id = latest["id"]

        conditions = ["domain=?", "run_id=?", "composite IS NOT NULL"]
        params: list = [domain, run_id]

        if min_score is not None:
            conditions.append("composite >= ?")
            params.append(min_score)

        if has_code:
            conditions.append("(code_url IS NOT NULL AND code_url != '')")

        if search:
            conditions.append("(title LIKE ? OR abstract LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        if topic:
            conditions.append("topics LIKE ?")
            params.append(f'%"{topic}"%')

        where = " AND ".join(conditions)
        order = SORT_OPTIONS.get(sort, "composite DESC")

        total = conn.execute(
            f"SELECT COUNT(*) FROM papers WHERE {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT * FROM papers WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return [_deserialize_paper(row) for row in rows], total


def count_papers(domain: str, run_id: int | None = None, scored_only: bool = False) -> int:
    with get_conn() as conn:
        if not run_id:
            latest = get_latest_run(domain)
            if not latest:
                return 0
            run_id = latest["id"]
        sql = "SELECT COUNT(*) FROM papers WHERE domain=? AND run_id=?"
        if scored_only:
            sql += " AND composite IS NOT NULL"
        row = conn.execute(sql, (domain, run_id)).fetchone()
        return row[0] if row else 0


def _deserialize_paper(row) -> dict:
    """Convert a sqlite3.Row to a dict, parsing JSON fields."""
    d = dict(row)
    for key in ("authors", "categories", "hf_models", "hf_datasets", "hf_spaces", "topics"):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d[key] = []
    return d


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def insert_events(events: list[dict], run_id: int | None = None):
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for e in events:
            conn.execute(
                """INSERT OR IGNORE INTO events
                   (run_id, category, title, description, url, event_date,
                    source, relevance_score, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    e.get("category", ""),
                    e.get("title", ""),
                    e.get("description", ""),
                    e.get("url", ""),
                    e.get("event_date", ""),
                    e.get("source", ""),
                    e.get("relevance_score"),
                    now,
                ),
            )


def get_events(category: str | None = None, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        if category:
            rows = conn.execute(
                "SELECT * FROM events WHERE category=? ORDER BY event_date DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY fetched_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


def count_events() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

def get_all_runs(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Paper connections (Semantic Scholar)
# ---------------------------------------------------------------------------

def insert_connections(connections: list[dict]):
    """Bulk-insert paper connections."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for c in connections:
            conn.execute(
                """INSERT INTO paper_connections
                   (paper_id, connected_arxiv_id, connected_s2_id,
                    connected_title, connected_year, connection_type,
                    in_db_paper_id, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    c["paper_id"],
                    c.get("connected_arxiv_id", ""),
                    c.get("connected_s2_id", ""),
                    c.get("connected_title", ""),
                    c.get("connected_year"),
                    c["connection_type"],
                    c.get("in_db_paper_id"),
                    now,
                ),
            )


def get_paper_connections(paper_id: int) -> dict:
    """Get connected papers grouped by type."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_connections WHERE paper_id=? "
            "ORDER BY connection_type, connected_year DESC",
            (paper_id,),
        ).fetchall()

    result = {"references": [], "recommendations": []}
    for row in rows:
        d = dict(row)
        ctype = d["connection_type"]
        if ctype in result:
            result[ctype].append(d)
    return result


def clear_connections(paper_id: int):
    """Remove existing connections for a paper (before re-enrichment)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM paper_connections WHERE paper_id=?", (paper_id,))


def update_paper_s2(paper_id: int, s2_paper_id: str, s2_tldr: str):
    """Update S2 metadata on a paper."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE papers SET s2_paper_id=?, s2_tldr=? WHERE id=?",
            (s2_paper_id, s2_tldr, paper_id),
        )


def update_paper_topics(paper_id: int, topics: list[str]):
    """Update topic tags on a paper."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE papers SET topics=? WHERE id=?",
            (json.dumps(topics), paper_id),
        )


def get_arxiv_id_map(run_id: int) -> dict[str, int]:
    """Return {arxiv_id: paper_db_id} for all papers in a run."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, arxiv_id FROM papers WHERE run_id=?", (run_id,)
        ).fetchall()
        return {row["arxiv_id"]: row["id"] for row in rows}


def get_available_topics(domain: str, run_id: int) -> list[str]:
    """Get distinct topic tags used in a run."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT topics FROM papers "
            "WHERE domain=? AND run_id=? AND topics IS NOT NULL AND topics != '[]'",
            (domain, run_id),
        ).fetchall()

    all_topics: set[str] = set()
    for row in rows:
        try:
            all_topics.update(json.loads(row["topics"]))
        except (json.JSONDecodeError, TypeError):
            pass
    return sorted(all_topics)


# ---------------------------------------------------------------------------
# Full-text search (FTS5)
# ---------------------------------------------------------------------------

FTS_SORT_OPTIONS = {
    "rank": "fts_rank",
    "score": "p.composite DESC",
    "date": "p.published DESC",
}


def search_papers_fts(
    query: str,
    domain: str | None = None,
    sort: str = "rank",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Full-text search across all papers, deduped by arxiv_id.

    Returns (papers_with_snippets, total_count).
    """
    with get_conn() as conn:
        domain_filter = ""
        params: list = []
        if domain:
            domain_filter = "AND p.domain = ?"
            params.append(domain)

        # BM25 weights: title=10, abstract=1, summary=5, topics=2
        sql = f"""
            WITH deduped AS (
                SELECT MAX(id) AS id
                FROM papers
                WHERE composite IS NOT NULL
                GROUP BY arxiv_id, domain
            )
            SELECT p.*,
                   bm25(papers_fts, 10.0, 1.0, 5.0, 2.0) AS fts_rank,
                   snippet(papers_fts, 0, '<mark>', '</mark>', '...', 40) AS snip_title,
                   snippet(papers_fts, 1, '<mark>', '</mark>', '...', 40) AS snip_abstract,
                   snippet(papers_fts, 2, '<mark>', '</mark>', '...', 40) AS snip_summary
            FROM papers_fts
            JOIN deduped d ON papers_fts.rowid = d.id
            JOIN papers p ON p.id = d.id
            WHERE papers_fts MATCH ?
            {domain_filter}
            ORDER BY {FTS_SORT_OPTIONS.get(sort, "fts_rank")}
        """

        match_query = query
        params_full = [match_query] + params

        count_sql = f"""
            WITH deduped AS (
                SELECT MAX(id) AS id
                FROM papers
                WHERE composite IS NOT NULL
                GROUP BY arxiv_id, domain
            )
            SELECT COUNT(*)
            FROM papers_fts
            JOIN deduped d ON papers_fts.rowid = d.id
            JOIN papers p ON p.id = d.id
            WHERE papers_fts MATCH ?
            {domain_filter}
        """

        try:
            total = conn.execute(count_sql, params_full).fetchone()[0]
        except sqlite3.OperationalError:
            return [], 0

        try:
            rows = conn.execute(
                sql + " LIMIT ? OFFSET ?",
                params_full + [limit, offset],
            ).fetchall()
        except sqlite3.OperationalError:
            return [], 0

        results = []
        for row in rows:
            d = _deserialize_paper(row)
            d["snip_title"] = row["snip_title"]
            d["snip_abstract"] = row["snip_abstract"]
            d["snip_summary"] = row["snip_summary"]
            results.append(d)

        return results, total


# ---------------------------------------------------------------------------
# GitHub project helpers
# ---------------------------------------------------------------------------

def insert_github_projects(projects: list[dict], run_id: int):
    """Bulk-insert GitHub projects into the DB."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for p in projects:
            conn.execute(
                """INSERT OR IGNORE INTO github_projects
                   (run_id, repo_id, repo_name, description, language,
                    stars, forks, pull_requests, total_score,
                    collection_names, topics, url, domain, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    p.get("repo_id", 0),
                    p.get("repo_name", ""),
                    p.get("description", ""),
                    p.get("language", ""),
                    p.get("stars", 0),
                    p.get("forks", 0),
                    p.get("pull_requests", 0),
                    p.get("total_score", 0),
                    p.get("collection_names", ""),
                    _serialize_json(p.get("topics", [])),
                    p.get("url", ""),
                    p.get("domain", ""),
                    now,
                ),
            )


GH_SORT_OPTIONS = {
    "score": "total_score DESC",
    "stars": "stars DESC",
    "forks": "forks DESC",
    "name": "repo_name ASC",
}


def get_github_projects_page(
    run_id: int | None = None,
    offset: int = 0,
    limit: int = 50,
    search: str | None = None,
    language: str | None = None,
    domain: str | None = None,
    sort: str | None = None,
) -> tuple[list[dict], int]:
    """Paginated, filterable GitHub project list."""
    with get_conn() as conn:
        if not run_id:
            latest = get_latest_run("github")
            if not latest:
                return [], 0
            run_id = latest["id"]

        conditions = ["run_id=?"]
        params: list = [run_id]

        if search:
            conditions.append("(repo_name LIKE ? OR description LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        if language:
            conditions.append("language=?")
            params.append(language)

        if domain:
            conditions.append("domain=?")
            params.append(domain)

        where = " AND ".join(conditions)
        order = GH_SORT_OPTIONS.get(sort, "total_score DESC")

        total = conn.execute(
            f"SELECT COUNT(*) FROM github_projects WHERE {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT * FROM github_projects WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return [_deserialize_gh_project(row) for row in rows], total


def get_top_github_projects(run_id: int | None = None, limit: int = 10) -> list[dict]:
    """Get top GitHub projects by score."""
    with get_conn() as conn:
        if not run_id:
            latest = get_latest_run("github")
            if not latest:
                return []
            run_id = latest["id"]
        rows = conn.execute(
            "SELECT * FROM github_projects WHERE run_id=? ORDER BY total_score DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [_deserialize_gh_project(row) for row in rows]


def count_github_projects(run_id: int | None = None) -> int:
    with get_conn() as conn:
        if not run_id:
            latest = get_latest_run("github")
            if not latest:
                return 0
            run_id = latest["id"]
        return conn.execute(
            "SELECT COUNT(*) FROM github_projects WHERE run_id=?", (run_id,)
        ).fetchone()[0]


def get_github_languages(run_id: int) -> list[str]:
    """Get distinct languages in a GitHub run."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT language FROM github_projects "
            "WHERE run_id=? AND language IS NOT NULL AND language != '' "
            "ORDER BY language",
            (run_id,),
        ).fetchall()
        return [row["language"] for row in rows]


def _deserialize_gh_project(row) -> dict:
    d = dict(row)
    for key in ("topics",):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d[key] = []
    return d


# ---------------------------------------------------------------------------
# User signal helpers (preference learning)
# ---------------------------------------------------------------------------

def insert_signal(paper_id: int, action: str, metadata: dict | None = None) -> bool:
    """Record a user signal. Returns True if inserted, False if duplicate.

    Views are deduped by 5-minute window. Other actions use UNIQUE constraint.
    """
    now = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata or {})
    with get_conn() as conn:
        if action == "view":
            # Dedup views within 5-minute window
            recent = conn.execute(
                "SELECT 1 FROM user_signals "
                "WHERE paper_id=? AND action='view' "
                "AND created_at > datetime(?, '-5 minutes')",
                (paper_id, now),
            ).fetchone()
            if recent:
                return False
            conn.execute(
                "INSERT INTO user_signals (paper_id, action, created_at, metadata) "
                "VALUES (?, ?, ?, ?)",
                (paper_id, action, now, meta_json),
            )
            return True
        else:
            try:
                conn.execute(
                    "INSERT INTO user_signals (paper_id, action, created_at, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    (paper_id, action, now, meta_json),
                )
                return True
            except sqlite3.IntegrityError:
                return False


def delete_signal(paper_id: int, action: str) -> bool:
    """Remove a signal (for toggling off). Returns True if deleted."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM user_signals WHERE paper_id=? AND action=?",
            (paper_id, action),
        )
        return cur.rowcount > 0


def get_paper_signal(paper_id: int) -> str | None:
    """Return the user's latest non-view signal for a paper, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT action FROM user_signals "
            "WHERE paper_id=? AND action != 'view' "
            "ORDER BY created_at DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
        return row["action"] if row else None


def get_paper_signals_batch(paper_ids: list[int]) -> dict[int, str]:
    """Batch fetch latest non-view signal per paper. Returns {paper_id: action}."""
    if not paper_ids:
        return {}
    with get_conn() as conn:
        placeholders = ",".join("?" for _ in paper_ids)
        rows = conn.execute(
            f"SELECT paper_id, action FROM user_signals "
            f"WHERE paper_id IN ({placeholders}) AND action != 'view' "
            f"ORDER BY created_at DESC",
            paper_ids,
        ).fetchall()
    result: dict[int, str] = {}
    for row in rows:
        pid = row["paper_id"]
        if pid not in result:
            result[pid] = row["action"]
    return result


def get_all_signals_with_papers() -> list[dict]:
    """Join signals with paper data for preference computation."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT s.id as signal_id, s.paper_id, s.action, s.created_at,
                      p.title, p.categories, p.topics, p.authors, p.domain,
                      p.score_axis_1, p.score_axis_2, p.score_axis_3, p.composite
               FROM user_signals s
               JOIN papers p ON s.paper_id = p.id
               ORDER BY s.created_at DESC"""
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        for key in ("categories", "topics", "authors"):
            val = d.get(key)
            if isinstance(val, str):
                try:
                    d[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d[key] = []
        results.append(d)
    return results


def get_signal_counts() -> dict[str, int]:
    """Summary stats: count per action type."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT action, COUNT(*) as cnt FROM user_signals GROUP BY action"
        ).fetchall()
    return {row["action"]: row["cnt"] for row in rows}


def save_preferences(prefs: dict[str, tuple[float, int]]):
    """Bulk write preferences. prefs = {key: (value, signal_count)}."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute("DELETE FROM user_preferences")
        for key, (value, count) in prefs.items():
            conn.execute(
                "INSERT INTO user_preferences (pref_key, pref_value, signal_count, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (key, value, count, now),
            )


def load_preferences() -> dict[str, float]:
    """Load preference profile. Returns {pref_key: pref_value}."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT pref_key, pref_value FROM user_preferences"
        ).fetchall()
    return {row["pref_key"]: row["pref_value"] for row in rows}


def get_preferences_detail() -> list[dict]:
    """Load full preference details for the preferences page."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM user_preferences ORDER BY ABS(pref_value) DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_preferences_updated_at() -> str | None:
    """Return when preferences were last computed."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT updated_at FROM user_preferences ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row["updated_at"] if row else None


def clear_preferences():
    """Reset all preferences and signals."""
    with get_conn() as conn:
        conn.execute("DELETE FROM user_preferences")
        conn.execute("DELETE FROM user_signals")


def upsert_seed_papers(papers: list[dict]) -> dict[str, int]:
    """Ensure seed papers exist in DB, return {arxiv_id: paper_db_id}.

    For each paper: if arxiv_id already exists, use the existing row's id.
    Otherwise INSERT a stub row with run_id=NULL and source='seed'.
    """
    result: dict[str, int] = {}
    with get_conn() as conn:
        for p in papers:
            arxiv_id = p.get("arxiv_id", "").strip()
            if not arxiv_id:
                continue

            # Check if paper already exists (from any run)
            row = conn.execute(
                "SELECT id FROM papers WHERE arxiv_id=? LIMIT 1",
                (arxiv_id,),
            ).fetchone()

            if row:
                result[arxiv_id] = row["id"]
            else:
                domain = p.get("domain", "aiml")
                conn.execute(
                    """INSERT OR IGNORE INTO papers
                       (run_id, domain, arxiv_id, entry_id, title, authors,
                        abstract, published, categories, pdf_url, arxiv_url,
                        comment, source)
                       VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        domain,
                        arxiv_id,
                        p.get("entry_id", ""),
                        p.get("title", ""),
                        _serialize_json(p.get("authors", [])),
                        p.get("abstract", ""),
                        p.get("published", ""),
                        _serialize_json(p.get("categories", [])),
                        p.get("pdf_url", ""),
                        p.get("arxiv_url", f"https://arxiv.org/abs/{arxiv_id}"),
                        p.get("comment", ""),
                        "seed",
                    ),
                )
                inserted = conn.execute(
                    "SELECT id FROM papers WHERE arxiv_id=? AND run_id IS NULL LIMIT 1",
                    (arxiv_id,),
                ).fetchone()
                if inserted:
                    result[arxiv_id] = inserted["id"]

    return result


# ---------------------------------------------------------------------------
# Video job helpers
# ---------------------------------------------------------------------------

def create_video_job(paper_id: int, job_id: str) -> int:
    """Insert a new video job, return its DB id."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO video_jobs (paper_id, job_id, status, created_at) VALUES (?, ?, 'queued', ?)",
            (paper_id, job_id, now),
        )
        return cur.lastrowid


def update_video_job(job_id: str, status: str, stage: str | None = None, error: str | None = None):
    """Update a video job's status."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        completed = now if status in ("done", "failed") else None
        conn.execute(
            "UPDATE video_jobs SET status=?, stage=?, error=?, completed_at=COALESCE(?, completed_at) WHERE job_id=?",
            (status, stage, error, completed, job_id),
        )


def get_video_job(paper_id: int) -> dict | None:
    """Get the latest video job for a paper."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM video_jobs WHERE paper_id=? ORDER BY id DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
        return dict(row) if row else None
