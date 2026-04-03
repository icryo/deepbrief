"""Unified Claude API scoring for both AI/ML and security domains.

Supports both direct Anthropic API and AWS Bedrock.
Set USE_BEDROCK=true (or configure in config.yaml) to use Bedrock.
"""

import json
import logging
import os
import re
import time

log = logging.getLogger(__name__)

import src.config as config
from src.config import SECURITY_LLM_RE
from src.db import get_unscored_papers, update_paper_scores


def _get_client():
    """Return an Anthropic or Bedrock client based on configuration."""
    use_bedrock = os.environ.get("USE_BEDROCK", "").lower() in ("1", "true", "yes")
    if not use_bedrock:
        # Check config.yaml too
        try:
            from src.config import _cfg
            use_bedrock = _cfg.get("scoring", {}).get("use_bedrock", False)
        except Exception:
            pass

    if use_bedrock:
        import anthropic
        return anthropic.AnthropicBedrock(timeout=120.0)
    else:
        import anthropic
        return anthropic.Anthropic(timeout=120.0)


def _has_credentials() -> bool:
    """Check if we have valid credentials for scoring."""
    use_bedrock = os.environ.get("USE_BEDROCK", "").lower() in ("1", "true", "yes")
    if not use_bedrock:
        try:
            from src.config import _cfg
            use_bedrock = _cfg.get("scoring", {}).get("use_bedrock", False)
        except Exception:
            pass

    if use_bedrock:
        # Bedrock uses AWS credentials (env vars, profile, or instance role)
        return bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
                     or os.environ.get("AWS_DEFAULT_REGION"))
    else:
        return bool(config.ANTHROPIC_API_KEY)


def score_run(run_id: int, domain: str) -> int:
    """Score all unscored papers in a run. Returns count of scored papers."""
    if not _has_credentials():
        log.warning("No API credentials set — skipping scoring")
        return 0

    scoring_model = config.SCORING_MODEL
    batch_size = config.BATCH_SIZE

    scoring_config = config.SCORING_CONFIGS[domain]
    papers = get_unscored_papers(run_id)

    if not papers:
        log.info("No unscored papers for run %d", run_id)
        return 0

    log.info("Scoring %d %s papers with %s ...", len(papers), domain, scoring_model)

    client = _get_client()
    max_chars = config.MAX_ABSTRACT_CHARS_AIML if domain == "aiml" else config.MAX_ABSTRACT_CHARS_SECURITY
    scored_count = 0
    t0 = time.monotonic()

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(papers) + batch_size - 1) // batch_size
        log.info("Batch %d/%d (%d papers) ...", batch_num, total_batches, len(batch))

        user_content = _build_batch_content(batch, domain, max_chars)
        scores = _call_claude(client, scoring_config["prompt"], user_content, model=scoring_model)
        if not scores:
            continue

        scored_count += _apply_scores(batch, scores, domain, scoring_config)

    elapsed = time.monotonic() - t0
    log.info("Scored %d/%d papers with %s in %.0fs", scored_count, len(papers), scoring_model, elapsed)
    return scored_count


def _build_batch_content(papers: list[dict], domain: str, max_chars: int) -> str:
    """Build the user content string for a batch of papers."""
    lines = []
    for p in papers:
        abstract = (p.get("abstract") or "")[:max_chars]
        id_field = p.get("entry_id") or p.get("arxiv_url") or p.get("arxiv_id", "")

        lines.append("---")

        if domain == "security":
            lines.append(f"entry_id: {id_field}")
        else:
            lines.append(f"arxiv_id: {p.get('arxiv_id', '')}")

        authors_list = p.get("authors", [])
        if isinstance(authors_list, str):
            authors_str = authors_list
        else:
            authors_str = ", ".join(authors_list[:5])

        cats = p.get("categories", [])
        if isinstance(cats, str):
            cats_str = cats
        else:
            cats_str = ", ".join(cats)

        lines.append(f"title: {p.get('title', '')}")
        lines.append(f"authors: {authors_str}")
        lines.append(f"categories: {cats_str}")

        code_url = p.get("github_repo") or p.get("code_url") or "none found"
        lines.append(f"code_url_found: {code_url}")

        if domain == "security":
            if "llm_adjacent" not in p:
                text = f"{p.get('title', '')} {p.get('abstract', '')}"
                p["llm_adjacent"] = bool(SECURITY_LLM_RE.search(text))
            lines.append(f"llm_adjacent: {str(p['llm_adjacent']).lower()}")

        if domain == "aiml":
            lines.append(f"hf_upvotes: {p.get('hf_upvotes', 0)}")
            hf_models = p.get("hf_models", [])
            if hf_models:
                model_ids = [m["id"] if isinstance(m, dict) else str(m) for m in hf_models[:3]]
                lines.append(f"hf_models: {', '.join(model_ids)}")
            hf_spaces = p.get("hf_spaces", [])
            if hf_spaces:
                space_ids = [s["id"] if isinstance(s, dict) else str(s) for s in hf_spaces[:3]]
                lines.append(f"hf_spaces: {', '.join(space_ids)}")
            lines.append(f"source: {p.get('source', 'unknown')}")

        lines.append(f"abstract: {abstract}")
        lines.append(f"comment: {p.get('comment', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def _call_claude(client, system_prompt: str, user_content: str, *, model: str) -> list[dict]:
    """Call Claude API (direct or Bedrock) and extract JSON response."""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            log.warning("No JSON array in response (attempt %d)", attempt + 1)
        except (json.JSONDecodeError, Exception) as e:
            log.error("Scoring API error (attempt %d): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                log.error("Skipping batch after 3 failures")
    return []


def _apply_scores(papers: list[dict], scores: list[dict], domain: str, config: dict) -> int:
    """Apply scores from Claude response to papers in DB. Returns count applied."""
    axes = config["axes"]
    weights = config["weights"]
    weight_values = list(weights.values())

    if domain == "security":
        score_map = {s.get("entry_id", ""): s for s in scores}
    else:
        score_map = {s.get("arxiv_id", ""): s for s in scores}

    applied = 0
    for paper in papers:
        if domain == "security":
            key = paper.get("entry_id") or paper.get("arxiv_url") or ""
        else:
            key = paper.get("arxiv_id", "")

        score = score_map.get(key)
        if not score:
            continue

        axis_scores = [score.get(ax, 0) for ax in axes]
        composite = sum(s * w for s, w in zip(axis_scores, weight_values))

        update_paper_scores(paper["id"], {
            "score_axis_1": axis_scores[0] if len(axis_scores) > 0 else None,
            "score_axis_2": axis_scores[1] if len(axis_scores) > 1 else None,
            "score_axis_3": axis_scores[2] if len(axis_scores) > 2 else None,
            "composite": round(composite, 2),
            "summary": score.get("summary", ""),
            "reasoning": score.get("reasoning", ""),
            "code_url": score.get("code_url"),
        })
        applied += 1

    return applied


def rescore_top(run_id: int, domain: str, n: int = 0) -> int:
    """Re-score the top N papers from a run using the stronger rescore model."""
    rescore_model = config.RESCORE_MODEL
    scoring_model = config.SCORING_MODEL

    n = n or config.RESCORE_TOP_N
    if n <= 0:
        return 0
    if not _has_credentials():
        log.warning("No API credentials set — skipping re-scoring")
        return 0
    if rescore_model == scoring_model:
        log.info("Rescore model same as scoring model — skipping re-score")
        return 0

    from src.db import get_top_papers

    scoring_config = config.SCORING_CONFIGS[domain]
    papers = get_top_papers(domain, run_id=run_id, limit=n)
    if not papers:
        log.info("No papers to re-score for run %d", run_id)
        return 0

    log.info("Re-scoring top %d %s papers with %s ...", len(papers), domain, rescore_model)

    client = _get_client()
    max_chars = config.MAX_ABSTRACT_CHARS_AIML if domain == "aiml" else config.MAX_ABSTRACT_CHARS_SECURITY
    t0 = time.monotonic()

    user_content = _build_batch_content(papers, domain, max_chars)
    scores = _call_claude(client, scoring_config["prompt"], user_content, model=rescore_model)

    if not scores:
        log.warning("Re-scoring returned no results")
        return 0

    rescored = _apply_scores(papers, scores, domain, scoring_config)
    elapsed = time.monotonic() - t0
    log.info("Re-scored %d/%d papers with %s in %.0fs", rescored, len(papers), rescore_model, elapsed)
    return rescored
