"""CLI competitive intelligence pipeline — track competitor CLI tools via GitHub API.

Uses the `gh` CLI to fetch repo metadata, PRs, commits, releases, and file trees
for a set of tracked CLI tool repositories. Generates findings (PRs, releases,
architecture patterns) and flags new activity since the last run.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import anthropic
import requests

import src.config as config
from src.config import _cfg, GITHUB_TOKEN
from src.db import create_run, finish_run, get_latest_run

log = logging.getLogger(__name__)

DEFAULT_REPOS = [
    "google-gemini/gemini-cli",
    "openclaw/openclaw",
    "openai/codex",
    "block/goose",
    "Aider-AI/aider",
    "aws/amazon-q-developer-cli",
]

# Architecture detection patterns: directory prefix -> capability label
_ARCH_PATTERNS = {
    "tools/": "tools",
    "src/tools/": "tools",
    "hooks/": "hooks",
    "src/hooks/": "hooks",
    "mcp/": "mcp",
    "sandbox/": "sandboxing",
    "security/": "sandboxing",
    "agents/": "multi-agent",
    "sdk/": "sdk",
    "voice/": "voice",
}


def _get_tracked_repos() -> list[str]:
    """Read tracked repos from config, falling back to defaults."""
    cli_cfg = _cfg.get("cli_intel", {})
    repos = cli_cfg.get("repos", DEFAULT_REPOS)
    return repos


def _gh_headers() -> dict:
    """Build GitHub API headers with optional auth token."""
    headers = {"Accept": "application/vnd.github+json"}
    token = GITHUB_TOKEN
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _gh_api(endpoint: str) -> dict | list | None:
    """Call GitHub REST API and return parsed JSON, or None on failure."""
    url = f"https://api.github.com{endpoint}" if endpoint.startswith("/") else endpoint
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=30)
        if resp.status_code != 200:
            log.warning("GitHub API %s returned %d: %s", endpoint, resp.status_code, resp.text[:200])
            return None
        return resp.json()
    except requests.RequestException as e:
        log.warning("GitHub API %s error: %s", endpoint, e)
        return None


def _gh_api_items(endpoint: str) -> list[dict]:
    """Call GitHub API expecting a JSON array. Returns [] on failure."""
    data = _gh_api(endpoint)
    if isinstance(data, list):
        return data
    return []


def _fetch_repo_metadata(repo: str) -> dict | None:
    """Fetch basic repo metadata (stars, forks, description, language)."""
    data = _gh_api(f"/repos/{repo}")
    if not isinstance(data, dict):
        return None
    return {
        "full_name": data.get("full_name", repo),
        "description": data.get("description", "") or "",
        "language": data.get("language", "") or "",
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "default_branch": data.get("default_branch", "main"),
        "updated_at": data.get("updated_at", ""),
    }


def _fetch_open_prs(repo: str) -> list[dict]:
    """Fetch open + recently merged PRs sorted by recently updated."""
    # Open PRs
    items = _gh_api_items(
        f"/repos/{repo}/pulls?state=open&sort=updated&direction=desc&per_page=30"
    )
    # Recently closed/merged PRs (where the real intel is)
    closed = _gh_api_items(
        f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc&per_page=50"
    )
    # Only keep merged PRs from closed
    for pr in closed:
        if pr.get("merged_at"):
            items.append(pr)
    prs = []
    for pr in items:
        body = (pr.get("body") or "")[:1500]  # PR description, truncated
        prs.append({
            "number": pr.get("number"),
            "title": pr.get("title", ""),
            "body": body,
            "user": (pr.get("user") or {}).get("login", ""),
            "state": "merged" if pr.get("merged_at") else pr.get("state", "open"),
            "merged": bool(pr.get("merged_at")),
            "draft": pr.get("draft", False),
            "created_at": pr.get("created_at", ""),
            "updated_at": pr.get("updated_at", ""),
            "url": pr.get("html_url", ""),
            "labels": [lb.get("name", "") for lb in (pr.get("labels") or [])],
        })
    return prs


def _fetch_recent_commits(repo: str, since_days: int = 30) -> list[dict]:
    """Fetch commits from the last N days (limit 100)."""
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    items = _gh_api_items(
        f"/repos/{repo}/commits?since={since}&per_page=100"
    )
    commits = []
    for c in items:
        commit_data = c.get("commit", {})
        commits.append({
            "sha": c.get("sha", "")[:12],
            "message": (commit_data.get("message") or "").split("\n")[0],
            "author": (commit_data.get("author") or {}).get("name", ""),
            "date": (commit_data.get("author") or {}).get("date", ""),
            "url": c.get("html_url", ""),
        })
    return commits


def _fetch_releases(repo: str, limit: int = 5) -> list[dict]:
    """Fetch the latest releases."""
    items = _gh_api_items(
        f"/repos/{repo}/releases?per_page={limit}"
    )
    releases = []
    for r in items:
        releases.append({
            "tag": r.get("tag_name", ""),
            "name": r.get("name", "") or r.get("tag_name", ""),
            "published_at": r.get("published_at", ""),
            "prerelease": r.get("prerelease", False),
            "url": r.get("html_url", ""),
            "body": (r.get("body") or "")[:500],
        })
    return releases


def _fetch_file_tree(repo: str, default_branch: str = "main") -> list[str]:
    """Fetch the recursive file tree via git/trees endpoint."""
    data = _gh_api(f"/repos/{repo}/git/trees/{default_branch}?recursive=1")
    if not isinstance(data, dict):
        return []
    tree = data.get("tree", [])
    return [item.get("path", "") for item in tree if item.get("type") in ("blob", "tree")]


def _detect_architecture(paths: list[str]) -> list[dict]:
    """Detect architectural patterns from a file tree."""
    capabilities: dict[str, int] = {}

    for path in paths:
        normalized = path.lower() + ("/" if not path.endswith("/") else "")
        for prefix, label in _ARCH_PATTERNS.items():
            if f"/{prefix}" in normalized or normalized.startswith(prefix):
                capabilities[label] = capabilities.get(label, 0) + 1

    findings = []
    for capability, count in sorted(capabilities.items(), key=lambda x: -x[1]):
        findings.append({
            "capability": capability,
            "file_count": count,
            "detail": f"Detected {count} files/dirs related to {capability}",
        })

    return findings


def _generate_findings(
    repo: str,
    prs: list[dict],
    releases: list[dict],
    arch_patterns: list[dict],
) -> list[dict]:
    """Convert raw data into normalized findings."""
    findings = []

    # PR findings
    for pr in prs:
        status = "merged" if pr.get("merged") else ("draft" if pr.get("draft") else "open")
        detail_obj = {"number": pr["number"], "labels": pr.get("labels", [])}
        if pr.get("body"):
            detail_obj["body"] = pr["body"]
        findings.append({
            "repo": repo,
            "type": "pr",
            "status": status,
            "title": pr.get("title", ""),
            "url": pr.get("url", ""),
            "author": pr.get("user", ""),
            "date": pr.get("updated_at", ""),
            "detail": json.dumps(detail_obj),
        })

    # Release findings
    for rel in releases:
        findings.append({
            "repo": repo,
            "type": "release",
            "status": "shipped",
            "title": f"{rel.get('name', '')} ({rel.get('tag', '')})",
            "url": rel.get("url", ""),
            "author": "",
            "date": rel.get("published_at", ""),
            "detail": rel.get("body", ""),
        })

    # Architecture findings
    for arch in arch_patterns:
        findings.append({
            "repo": repo,
            "type": "architecture",
            "status": "detected",
            "title": arch["capability"],
            "url": f"https://github.com/{repo}",
            "author": "",
            "date": datetime.now(timezone.utc).isoformat(),
            "detail": arch["detail"],
        })

    return findings


def _load_previous_snapshot(domain: str) -> set[str]:
    """Load finding keys from the previous run for is_new comparison.

    Returns a set of 'repo:type:title' keys from the previous run's findings.
    Falls back to empty set if no previous run or DB tables don't exist yet.
    """
    try:
        from src.cli_intel_db import get_cli_findings_page
        prev_run = get_latest_run(domain)
        if not prev_run:
            return set()
        prev_findings, _ = get_cli_findings_page(run_id=prev_run["id"], limit=5000)
        return {f"{f['repo_name']}:{f['finding_type']}:{f['title']}" for f in prev_findings}
    except Exception as e:
        log.debug("No previous snapshot available: %s", e)
        return set()


def _flag_new_findings(findings: list[dict], previous_keys: set[str]) -> list[dict]:
    """Mark each finding with is_new based on whether it appeared in the last run."""
    for f in findings:
        key = f"{f['repo']}:{f['type']}:{f['title']}"
        f["is_new"] = key not in previous_keys
    return findings


CLI_INTEL_SCORING_PROMPT = """\
You are a senior competitive intelligence analyst for Claude Code (Anthropic's AI CLI coding tool).
You are analyzing findings from competitor CLI tools: Gemini CLI (Google), OpenClaw, Codex (OpenAI), \
Goose (Block), Aider, Amazon Q Developer CLI.

Your job is to find genuinely novel PARADIGM SHIFTS — not routine engineering work. \
The bar is extremely high. A typical batch of 30 findings should produce AT MOST 1-3 non-noise results.

=== WHAT SCORES HIGH (7-10) — paradigm shifts only ===
- New architectural paradigms: pluggable context engines, new sandboxing approaches (kernel-level, \
  not just Docker), novel edit strategies (multi-tier fuzzy matching with LLM self-correction)
- New protocol standards: Agent-to-Agent protocols, new interop formats
- First-of-kind capabilities: native search grounding in model calls, DAG-based context compression, \
  cross-session vector memory, deterministic multi-agent workflow engines
- Capabilities that SURPASS what Claude Code can do today

=== WHAT IS NOISE (score 0) — be ruthless ===
- Adding SDKs, publishing packages — this is standard engineering, not novel
- "Safety features" without specifics — vague PR titles are noise
- Adding model support (e.g., "Add Copilot models") — routine integration work
- UI refreshes, layout changes, button tweaks, cosmetic improvements
- Bug fixes, docs, typos, CI/CD, dependency bumps, refactoring
- Generic "multi-worker" or "parallel execution" PRs without architectural novelty
- Features that every CLI tool already has (file reading, shell execution, web search)
- PRs that SOUND important from the title but are standard engineering work
- Releases that just bundle incremental improvements
- Architecture pattern detections (tools/, hooks/, mcp/ directories) — existence of directories is not a finding

=== CALIBRATION ===
Think about what would make a senior engineering leader say "we need to respond to this." \
An SDK being published does NOT trigger that reaction. A competitor implementing kernel-level \
sandboxing that's faster and more secure than our approach DOES.

Score 8-10: "This changes the competitive landscape" (expect 0-1 per batch)
Score 5-7: "Worth monitoring, may need a response" (expect 1-2 per batch)
Score 1-4: low — minor but real signal
Score 0: noise — the vast majority of findings

=== SCORING AXES (0-10) ===
1. **threat** — Would this make users switch FROM Claude Code? (0 = no, 10 = immediately)
2. **novelty** — Is this a genuinely new paradigm or approach no one else has? (0 = standard, 10 = first-of-kind)
3. **momentum** — Does this indicate a strategic direction that compounds? (0 = one-off, 10 = platform play)

=== OUTPUT ===
JSON array. Each object: id, threat, novelty, momentum, threat_level, summary, reasoning
threat_level: "high" (7+), "medium" (4-6), "low" (1-3), "noise" (0)
For noise: all scores 0, summary and reasoning empty strings.
For non-noise: summary should explain the SPECIFIC technical innovation, not just restate the PR title.\
"""


def _score_cli_findings(run_id: int) -> int:
    """Score CLI intel findings using Claude. Returns count scored."""
    if not config.ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — skipping CLI intel scoring")
        return 0

    from src.cli_intel_db import get_unscored_findings, update_finding_scores

    findings = get_unscored_findings(run_id)
    if not findings:
        log.info("No unscored CLI intel findings")
        return 0

    model = config.RESCORE_MODEL or config.SCORING_MODEL
    log.info("Scoring %d CLI intel findings with %s ...", len(findings), model)

    client = anthropic.Anthropic(timeout=120.0)
    batch_size = 30
    scored = 0
    t0 = time.monotonic()

    for i in range(0, len(findings), batch_size):
        batch = findings[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(findings) + batch_size - 1) // batch_size
        log.info("CLI intel batch %d/%d (%d findings) ...", batch_num, total_batches, len(batch))

        # Build content
        lines = []
        for f in batch:
            lines.append("---")
            lines.append(f"id: {f['id']}")
            lines.append(f"repo: {f['repo_name']}")
            lines.append(f"type: {f['finding_type']}")
            lines.append(f"title: {f['title']}")
            if f.get("description"):
                lines.append(f"description: {f['description'][:1000]}")
            if f.get("url"):
                lines.append(f"url: {f['url']}")
            if f.get("status"):
                lines.append(f"status: {f['status']}")
            if f.get("labels"):
                labels = f["labels"] if isinstance(f["labels"], list) else []
                if labels:
                    lines.append(f"labels: {', '.join(str(l) for l in labels)}")
            lines.append(f"is_new: {bool(f.get('is_new'))}")
            lines.append("")
        user_content = "\n".join(lines)

        # Call Claude
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=CLI_INTEL_SCORING_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                text = response.content[0].text
                json_match = re.search(r"\[.*\]", text, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    break
                log.warning("CLI intel scoring: no JSON array (attempt %d)", attempt + 1)
                scores = []
            except (anthropic.APIError, json.JSONDecodeError) as e:
                log.error("CLI intel scoring error (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                scores = []
        else:
            log.error("Skipping CLI intel batch after 3 failures")
            continue

        # Apply scores
        score_map = {s.get("id"): s for s in scores if isinstance(s, dict)}
        for f in batch:
            s = score_map.get(f["id"])
            if not s:
                continue

            threat = float(s.get("threat", 0))
            novelty = float(s.get("novelty", 0))
            momentum = float(s.get("momentum", 0))
            # Weighted composite: threat 40%, novelty 35%, momentum 25%
            composite = round(threat * 0.4 + novelty * 0.35 + momentum * 0.25, 2)

            update_finding_scores(f["id"], {
                "score_threat": threat,
                "score_novelty": novelty,
                "score_momentum": momentum,
                "composite": composite,
                "threat_level": s.get("threat_level", "noise"),
                "summary": s.get("summary", ""),
                "reasoning": s.get("reasoning", ""),
            })
            scored += 1

    elapsed = time.monotonic() - t0
    log.info("Scored %d/%d CLI intel findings in %.0fs", scored, len(findings), elapsed)
    return scored


def run_cli_intel_pipeline() -> int:
    """Run the full CLI competitive intelligence pipeline. Returns run_id."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=30)).date().isoformat()
    end = now.date().isoformat()

    run_id = create_run("cli_intel", start, end)
    log.info("CLI intel pipeline started — run %d (%s to %s)", run_id, start, end)

    try:
        repos = _get_tracked_repos()
        log.info("Tracking %d repos: %s", len(repos), ", ".join(repos))

        previous_keys = _load_previous_snapshot("cli_intel")
        log.info("Previous snapshot has %d finding keys", len(previous_keys))

        all_snapshots = []
        all_findings = []

        for repo in repos:
            log.info("Fetching data for %s", repo)

            # Repo metadata
            metadata = _fetch_repo_metadata(repo)
            if metadata is None:
                log.warning("Skipping %s — could not fetch metadata", repo)
                continue

            default_branch = metadata.get("default_branch", "main")

            # Fetch all data sources
            prs = _fetch_open_prs(repo)
            log.info("  %s: %d open PRs", repo, len(prs))

            commits = _fetch_recent_commits(repo)
            log.info("  %s: %d recent commits", repo, len(commits))

            releases = _fetch_releases(repo)
            log.info("  %s: %d releases", repo, len(releases))

            file_tree = _fetch_file_tree(repo, default_branch)
            log.info("  %s: %d files in tree", repo, len(file_tree))

            arch_patterns = _detect_architecture(file_tree)
            log.info("  %s: %d architecture patterns", repo, len(arch_patterns))

            # Build snapshot
            snapshot = {
                "repo": repo,
                "metadata": metadata,
                "prs": prs,
                "commits": commits,
                "releases": releases,
                "arch_patterns": arch_patterns,
                "file_count": len(file_tree),
            }
            all_snapshots.append(snapshot)

            # Generate findings
            findings = _generate_findings(repo, prs, releases, arch_patterns)
            all_findings.extend(findings)

        # Flag new vs. previously-seen findings
        all_findings = _flag_new_findings(all_findings, previous_keys)
        new_count = sum(1 for f in all_findings if f.get("is_new"))
        log.info("Total findings: %d (%d new)", len(all_findings), new_count)

        # Store via DB helpers
        from src.cli_intel_db import insert_cli_repo, insert_cli_findings

        for snap in all_snapshots:
            meta = snap["metadata"]
            latest_rel = snap["releases"][0] if snap["releases"] else {}
            insert_cli_repo({
                "repo_name": snap["repo"],
                "stars": meta.get("stars", 0),
                "forks": meta.get("forks", 0),
                "open_prs": len(snap["prs"]),
                "commits_30d": len(snap["commits"]),
                "description": meta.get("description", ""),
                "language": meta.get("language", ""),
                "latest_release": latest_rel.get("tag"),
                "release_date": latest_rel.get("published_at"),
                "snapshot_json": snap,
            }, run_id)

        # Normalize finding field names for DB layer
        db_findings = []
        for f in all_findings:
            detail = f.get("detail", "")
            pr_number = None
            labels = []
            pr_body = ""
            if f["type"] == "pr" and detail:
                try:
                    d = json.loads(detail)
                    pr_number = d.get("number")
                    labels = d.get("labels", [])
                    pr_body = d.get("body", "")
                except (json.JSONDecodeError, TypeError):
                    pass
            db_findings.append({
                "repo_name": f["repo"],
                "finding_type": f["type"],
                "title": f["title"],
                "description": pr_body if f["type"] == "pr" else detail,
                "url": f.get("url", ""),
                "pr_number": pr_number,
                "status": f.get("status"),
                "labels": labels,
                "is_new": 1 if f.get("is_new") else 0,
                "first_seen": f.get("date", ""),
                "last_seen": f.get("date", ""),
            })

        if db_findings:
            insert_cli_findings(db_findings, run_id)

        # Score findings with Claude
        scored = _score_cli_findings(run_id)
        log.info("Scored %d findings", scored)

        finish_run(run_id, len(all_findings))
        log.info("CLI intel pipeline complete — %d snapshots, %d findings (%d scored)",
                 len(all_snapshots), len(all_findings), scored)
        return run_id

    except Exception:
        finish_run(run_id, 0, status="failed")
        log.exception("CLI intel pipeline failed")
        raise
