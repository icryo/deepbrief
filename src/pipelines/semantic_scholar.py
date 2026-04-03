"""Semantic Scholar enrichment — connected papers, TL;DR, and topic extraction.

Uses the free S2 Academic Graph API. No API key required but rate-limited
to a shared pool. With a key (x-api-key header), 1 req/sec guaranteed.

Enrichment strategy:
1. Batch lookup all papers → TL;DR + S2 paper ID  (1 API call per 500 papers)
2. Top N papers by score → references + recommendations  (2 calls each)
3. Topic extraction from title/abstract  (local, no API)
"""

import logging
import re
import time

import requests

log = logging.getLogger(__name__)

from src.db import (
    clear_connections,
    get_arxiv_id_map,
    get_conn,
    get_top_papers,
    insert_connections,
    update_paper_s2,
    update_paper_topics,
)

S2_GRAPH = "https://api.semanticscholar.org/graph/v1"
S2_RECO = "https://api.semanticscholar.org/recommendations/v1"
S2_HEADERS: dict[str, str] = {}  # Add {"x-api-key": "..."} if you have one

# How many top papers get full connection enrichment
TOP_N_CONNECTIONS = 30
# Rate limit pause between requests (seconds)
RATE_LIMIT = 1.1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def enrich_run(run_id: int, domain: str):
    """Enrich all scored papers in a run with S2 data + topics."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, arxiv_id, title, abstract, composite FROM papers "
            "WHERE run_id=? AND composite IS NOT NULL "
            "ORDER BY composite DESC",
            (run_id,),
        ).fetchall()
        papers = [dict(r) for r in rows]

    if not papers:
        log.info("No scored papers in run %d, skipping", run_id)
        return

    arxiv_map = get_arxiv_id_map(run_id)
    log.info("Enriching %d papers from run %d (%s)...", len(papers), run_id, domain)

    # Step 1: Batch TL;DR + S2 ID
    _batch_tldr(papers)

    # Step 2: Connected papers for top N
    top_papers = papers[:TOP_N_CONNECTIONS]
    for i, p in enumerate(top_papers):
        try:
            _fetch_connections(p, arxiv_map)
        except Exception as e:
            log.warning("Error fetching connections for %s: %s", p['arxiv_id'], e)
        if (i + 1) % 10 == 0:
            log.info("Connections: %d/%d", i + 1, len(top_papers))

    # Step 3: Topic extraction (local, instant)
    for p in papers:
        topics = extract_topics(p["title"], p.get("abstract", ""), domain)
        if topics:
            update_paper_topics(p["id"], topics)

    log.info("Done enriching run %d", run_id)


# ---------------------------------------------------------------------------
# Step 1: Batch TL;DR
# ---------------------------------------------------------------------------


def _batch_tldr(papers: list[dict]):
    """Batch fetch TL;DR and S2 paper IDs."""
    chunk_size = 500
    for start in range(0, len(papers), chunk_size):
        chunk = papers[start : start + chunk_size]
        ids = [f"arXiv:{p['arxiv_id']}" for p in chunk]

        try:
            resp = requests.post(
                f"{S2_GRAPH}/paper/batch",
                params={"fields": "externalIds,tldr"},
                json={"ids": ids},
                headers=S2_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            log.warning("Batch TL;DR failed: %s", e)
            time.sleep(RATE_LIMIT)
            continue

        for paper, s2_data in zip(chunk, results):
            if s2_data is None:
                continue
            s2_id = s2_data.get("paperId", "")
            tldr_obj = s2_data.get("tldr")
            tldr_text = tldr_obj.get("text", "") if tldr_obj else ""
            update_paper_s2(paper["id"], s2_id, tldr_text)
            paper["s2_paper_id"] = s2_id

        found = sum(1 for r in results if r is not None)
        log.info("Batch TL;DR: %d/%d papers found in S2", found, len(chunk))
        time.sleep(RATE_LIMIT)


# ---------------------------------------------------------------------------
# Step 2: Connected papers (references + recommendations)
# ---------------------------------------------------------------------------


def _fetch_connections(paper: dict, arxiv_map: dict[str, int]):
    """Fetch references and recommendations for a single paper."""
    arxiv_id = paper["arxiv_id"]
    paper_id = paper["id"]

    # Clear old connections before re-fetching
    clear_connections(paper_id)

    connections: list[dict] = []

    # References
    time.sleep(RATE_LIMIT)
    try:
        resp = requests.get(
            f"{S2_GRAPH}/paper/arXiv:{arxiv_id}/references",
            params={"fields": "title,year,externalIds", "limit": 30},
            headers=S2_HEADERS,
            timeout=15,
        )
        if resp.ok:
            for item in resp.json().get("data", []):
                cited = item.get("citedPaper")
                if not cited or not cited.get("title"):
                    continue
                ext = cited.get("externalIds") or {}
                c_arxiv = ext.get("ArXiv", "")
                connections.append({
                    "paper_id": paper_id,
                    "connected_arxiv_id": c_arxiv,
                    "connected_s2_id": cited.get("paperId", ""),
                    "connected_title": cited.get("title", ""),
                    "connected_year": cited.get("year"),
                    "connection_type": "reference",
                    "in_db_paper_id": arxiv_map.get(c_arxiv),
                })
    except requests.RequestException as e:
        log.warning("References failed for %s: %s", arxiv_id, e)

    # Recommendations
    time.sleep(RATE_LIMIT)
    try:
        resp = requests.get(
            f"{S2_RECO}/papers/forpaper/arXiv:{arxiv_id}",
            params={"fields": "title,year,externalIds", "limit": 15},
            headers=S2_HEADERS,
            timeout=15,
        )
        if resp.ok:
            for rec in resp.json().get("recommendedPapers", []):
                if not rec or not rec.get("title"):
                    continue
                ext = rec.get("externalIds") or {}
                c_arxiv = ext.get("ArXiv", "")
                connections.append({
                    "paper_id": paper_id,
                    "connected_arxiv_id": c_arxiv,
                    "connected_s2_id": rec.get("paperId", ""),
                    "connected_title": rec.get("title", ""),
                    "connected_year": rec.get("year"),
                    "connection_type": "recommendation",
                    "in_db_paper_id": arxiv_map.get(c_arxiv),
                })
    except requests.RequestException as e:
        log.warning("Recommendations failed for %s: %s", arxiv_id, e)

    if connections:
        insert_connections(connections)


# ---------------------------------------------------------------------------
# Step 3: Topic extraction (local, no API)
# ---------------------------------------------------------------------------

AIML_TOPICS = {
    "Video Generation": re.compile(
        r"video.generat|text.to.video|video.diffusion|video.synth|video.edit", re.I),
    "Image Generation": re.compile(
        r"image.generat|text.to.image|(?:stable|latent).diffusion|image.synth|image.edit", re.I),
    "Language Models": re.compile(
        r"language.model|(?:large|foundation).model|\bllm\b|\bgpt\b|instruction.tun|fine.tun", re.I),
    "Code": re.compile(
        r"code.generat|code.complet|program.synth|vibe.cod|software.engineer", re.I),
    "Multimodal": re.compile(
        r"multimodal|vision.language|\bvlm\b|visual.question|image.text", re.I),
    "Efficiency": re.compile(
        r"quantiz|distillat|pruning|efficient|scaling.law|compress|accelerat", re.I),
    "Agents": re.compile(
        r"\bagent\b|tool.use|function.call|planning|agentic", re.I),
    "Speech / Audio": re.compile(
        r"text.to.speech|\btts\b|speech|audio.generat|voice|music.generat", re.I),
    "3D / Vision": re.compile(
        r"\b3d\b|nerf|gaussian.splat|point.cloud|depth.estim|object.detect|segmentat", re.I),
    "Retrieval / RAG": re.compile(
        r"retriev|\brag\b|knowledge.(?:base|graph)|in.context.learn|embedding", re.I),
    "Robotics": re.compile(
        r"robot|embodied|manipulat|locomotion|navigation", re.I),
    "Reasoning": re.compile(
        r"reasoning|chain.of.thought|mathemat|logic|theorem", re.I),
    "Training": re.compile(
        r"reinforcement.learn|\brlhf\b|\bdpo\b|preference|reward.model|alignment", re.I),
    "Architecture": re.compile(
        r"attention.mechanism|state.space|\bmamba\b|mixture.of.expert|\bmoe\b|transformer", re.I),
    "Benchmark": re.compile(
        r"benchmark|evaluat|leaderboard|dataset|scaling.law", re.I),
    "World Models": re.compile(
        r"world.model|environment.model|predictive.model|dynamics.model", re.I),
    "Optimization": re.compile(
        r"optimi[zs]|gradient|convergence|learning.rate|loss.function|multi.objective|adversarial.train", re.I),
    "RL": re.compile(
        r"reinforcement.learn|\brl\b|reward|policy.gradient|q.learning|bandit", re.I),
}

SECURITY_TOPICS = {
    "Web Security": re.compile(
        r"web.(?:secur|app|vuln)|xss|injection|csrf|waf|\bbrowser.secur", re.I),
    "Network": re.compile(
        r"network.secur|intrusion|\bids\b|firewall|traffic|\bdns\b|\bbgp\b|\bddos\b|fingerprint|scanning|packet", re.I),
    "Malware": re.compile(
        r"malware|ransomware|trojan|botnet|rootkit|worm|backdoor", re.I),
    "Vulnerabilities": re.compile(
        r"vulnerab|\bcve\b|exploit|fuzzing|fuzz|buffer.overflow|zero.day|attack.surface|security.bench", re.I),
    "Cryptography": re.compile(
        r"cryptograph|encryption|decrypt|protocol|\btls\b|\bssl\b|cipher", re.I),
    "Hardware": re.compile(
        r"side.channel|timing.attack|spectre|meltdown|hardware|firmware|microarch|fault.inject|emfi|embedded.secur", re.I),
    "Reverse Engineering": re.compile(
        r"reverse.engineer|binary|decompil|obfuscat|disassembl", re.I),
    "Mobile": re.compile(
        r"\bandroid\b|\bios.secur|mobile.secur", re.I),
    "Cloud": re.compile(
        r"cloud.secur|container.secur|docker|kubernetes|serverless|devsecops", re.I),
    "Authentication": re.compile(
        r"authentica|identity|credential|phishing|password|oauth|passkey|webauthn", re.I),
    "Privacy": re.compile(
        r"privacy|anonymi|differential.privacy|data.leak|tracking|membership.inference", re.I),
    "LLM Security": re.compile(
        r"(?:llm|language.model).*(secur|attack|jailbreak|safety|risk|unsafe|inject|adversar)|prompt.inject|red.team|rubric.attack|preference.drift", re.I),
    "Forensics": re.compile(
        r"forensic|incident.response|audit|log.analy|carver|tamper|evidence", re.I),
    "Blockchain": re.compile(
        r"blockchain|smart.contract|solana|ethereum|memecoin|mev|defi|token|cryptocurrency", re.I),
    "Supply Chain": re.compile(
        r"supply.chain|dependency|package.secur|software.comp|sbom", re.I),
}


def extract_topics(title: str, abstract: str, domain: str) -> list[str]:
    """Extract up to 3 topic tags from title and abstract."""
    patterns = AIML_TOPICS if domain == "aiml" else SECURITY_TOPICS
    abstract_head = (abstract or "")[:500]

    scored: dict[str, int] = {}
    for topic, pattern in patterns.items():
        score = 0
        if pattern.search(title):
            score += 3  # Title match is strong signal
        if pattern.search(abstract_head):
            score += 1
        if score > 0:
            scored[topic] = score

    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [t for t, _ in ranked[:3]]
