# Research Intelligence System

Unified research triage system covering AI/ML and security papers with a web dashboard.

## Architecture

- **Web dashboard**: FastAPI + Jinja2 + HTMX on port 8888
- **Database**: SQLite at `data/researcher.db`
- **Pipelines**: `src/pipelines/aiml.py` (HF + arXiv), `src/pipelines/security.py` (arXiv cs.CR)
- **Scoring**: `src/scoring.py` — Claude API batch scoring with domain-specific prompts
- **Scheduler**: APScheduler runs weekly (Sunday 22:00 UTC)

## Working with the Database

```bash
# Open the SQLite database
sqlite3 data/researcher.db

# Top papers this week (AI/ML)
SELECT title, composite, summary FROM papers
WHERE domain='aiml' AND composite IS NOT NULL
ORDER BY composite DESC LIMIT 10;

# Top papers this week (Security)
SELECT title, composite, summary FROM papers
WHERE domain='security' AND composite IS NOT NULL
ORDER BY composite DESC LIMIT 10;

# Papers with code
SELECT title, composite, code_url FROM papers
WHERE code_url IS NOT NULL AND code_url != ''
ORDER BY composite DESC LIMIT 20;

# Score distribution
SELECT domain, ROUND(composite, 0) as score, COUNT(*) as count
FROM papers WHERE composite IS NOT NULL
GROUP BY domain, score ORDER BY domain, score DESC;

# Recent runs
SELECT * FROM runs ORDER BY id DESC LIMIT 10;

# Events summary
SELECT category, COUNT(*) FROM events GROUP BY category;
```

## Running Pipelines Manually

```bash
# From the project directory
python -c "from src.pipelines.aiml import run_aiml_pipeline; run_aiml_pipeline()"
python -c "from src.pipelines.security import run_security_pipeline; run_security_pipeline()"
python -c "from src.pipelines.events import run_events_pipeline; run_events_pipeline()"

# Score a completed run
python -c "from src.scoring import score_run; score_run(RUN_ID, 'aiml')"
```

## Key Files

| File | Purpose |
|------|---------|
| `src/db.py` | SQLite schema + query helpers |
| `src/config.py` | All constants, scoring prompts, weights |
| `src/pipelines/aiml.py` | AI/ML paper fetching (HF + arXiv) |
| `src/pipelines/security.py` | Security paper fetching (arXiv cs.CR) |
| `src/scoring.py` | Unified Claude API scorer |
| `src/pipelines/events.py` | Conferences, releases, RSS news |
| `src/web/app.py` | FastAPI routes + report generation |
| `src/scheduler.py` | APScheduler weekly trigger |

## Scoring Criteria

### AI/ML (from hfrev)
- **Code & Weights** (30%) — open weights on HF, code on GitHub
- **Novelty** (35%) — paradigm shifts over incremental
- **Practical Applicability** (35%) — usable by practitioners soon

Prefer: video/image generation, world models, TTS, open-weight models, efficiency breakthroughs
Avoid: medical imaging, climate, surveys, sentiment analysis, closed-model papers

### Security (from arxivrev)
- **Has Code/PoC** (25%) — working tools, repos, artifacts
- **Novel Attack Surface** (40%) — first-of-kind research
- **Real-World Impact** (35%) — affects production systems

Prefer: "someone broke something real in a new way"
Avoid: LLM safety/alignment, ML-for-secure-code, theoretical privacy

## Docker

```bash
# Build and run
docker compose up --build

# Access dashboard
open http://localhost:9090

# Trigger pipelines via API
curl -X POST http://localhost:9090/run/aiml
curl -X POST http://localhost:9090/run/security
curl -X POST http://localhost:9090/run/events
```

## Allowed Tools

When working with this project in Claude Code:
- **Bash**: python, sqlite3, curl, docker commands
- **WebSearch/WebFetch**: arXiv, GitHub, HuggingFace for paper details
- **Read/Edit**: all project files and data/
