"""Configuration loader — reads from config.yaml, falls back to defaults."""

import logging
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging (always available, before config loads)
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    stream=sys.stdout,
)

# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config file path
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "config.yaml"))
FIRST_RUN = not CONFIG_PATH.exists()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
USE_BEDROCK = os.environ.get("USE_BEDROCK", "").lower() in ("1", "true", "yes")


def validate_env():
    """Check required environment variables at startup. Warn on missing."""
    if USE_BEDROCK or _cfg.get("scoring", {}).get("use_bedrock", False):
        log.info("Using AWS Bedrock for scoring")
    elif not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set and Bedrock not enabled — scoring will be disabled")
    if not GITHUB_TOKEN:
        log.info("GITHUB_TOKEN not set — GitHub API calls will be rate-limited")


# ---------------------------------------------------------------------------
# Load config.yaml (or defaults)
# ---------------------------------------------------------------------------

def _load_yaml() -> dict:
    """Load config.yaml if present, otherwise return empty dict."""
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH) as f:
                data = yaml.safe_load(f) or {}
            log.info("Loaded config from %s", CONFIG_PATH)
            return data
        except Exception as e:
            log.error("Failed to load %s: %s — using defaults", CONFIG_PATH, e)
    return {}


_cfg = _load_yaml()

# ---------------------------------------------------------------------------
# Claude API / Scoring models
# ---------------------------------------------------------------------------

_scoring_cfg = _cfg.get("scoring", {})
SCORING_MODEL = _scoring_cfg.get("model", _cfg.get("claude_model", "claude-haiku-4-5-20251001"))
RESCORE_MODEL = _scoring_cfg.get("rescore_model", "claude-sonnet-4-5-20250929")
RESCORE_TOP_N = _scoring_cfg.get("rescore_top_n", 15)
BATCH_SIZE = _scoring_cfg.get("batch_size", _cfg.get("batch_size", 20))

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = Path(_cfg.get("database", {}).get("path", os.environ.get("DB_PATH", "data/researcher.db")))

# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

WEB_HOST = _cfg.get("web", {}).get("host", "0.0.0.0")
WEB_PORT = _cfg.get("web", {}).get("port", 8888)

# ---------------------------------------------------------------------------
# Paper2Video integration
# ---------------------------------------------------------------------------

PAPER2VIDEO_URL = os.environ.get("PAPER2VIDEO_URL") or _cfg.get("paper2video", {}).get("url", "http://localhost:8001")

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

SCHEDULE_CRON = _cfg.get("schedule", {}).get("cron", "0 22 * * 0")

# ---------------------------------------------------------------------------
# Domains from config
# ---------------------------------------------------------------------------

_domains_cfg = _cfg.get("domains", {})

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HF_API = "https://huggingface.co/api"
GITHUB_URL_RE = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
MAX_ABSTRACT_CHARS_AIML = 2000
MAX_ABSTRACT_CHARS_SECURITY = 1500
HF_MAX_AGE_DAYS = 90

# ---------------------------------------------------------------------------
# AI/ML pipeline constants
# ---------------------------------------------------------------------------

_aiml_cfg = _domains_cfg.get("aiml", {})

ARXIV_LARGE_CATS = _aiml_cfg.get("arxiv_categories", ["cs.CV", "cs.CL", "cs.LG"])
ARXIV_SMALL_CATS = ["eess.AS", "cs.SD"]

_aiml_include = _aiml_cfg.get("include_patterns", [])
_aiml_exclude = _aiml_cfg.get("exclude_patterns", [])

_DEFAULT_INCLUDE = (
    r"video.generat|world.model|image.generat|diffusion|text.to.image|text.to.video|"
    r"code.generat|foundation.model|open.weight|large.language|language.model|"
    r"text.to.speech|tts|speech.synth|voice.clon|audio.generat|"
    r"transformer|attention.mechanism|state.space|mamba|mixture.of.expert|\bmoe\b|"
    r"scaling.law|architecture|quantiz|distillat|pruning|"
    r"multimodal|vision.language|\bvlm\b|agent|reasoning|"
    r"reinforcement.learn|rlhf|dpo|preference.optim|"
    r"retrieval.augment|\brag\b|in.context.learn|"
    r"image.edit|video.edit|3d.generat|nerf|gaussian.splat|"
    r"robot|embodied|simulat|"
    r"benchmark|evaluat|leaderboard|"
    r"open.source|reproducib|"
    r"instruction.tun|fine.tun|align|"
    r"long.context|context.window|"
    r"token|vocab|embedding|"
    r"training.efficien|parallel|distributed.train|"
    r"synthetic.data|data.curat"
)

_DEFAULT_EXCLUDE = (
    r"medical.imag|clinical|radiology|pathology|histolog|"
    r"climate.model|weather.predict|meteorolog|"
    r"survey.of|comprehensive.survey|"
    r"sentiment.analysis|named.entity|"
    r"drug.discover|protein.fold|molecular.dock|"
    r"software.engineering.practice|code.smell|technical.debt|"
    r"autonomous.driv|traffic.signal|"
    r"remote.sens|satellite.imag|crop.yield|"
    r"stock.predict|financial.forecast|"
    r"electronic.health|patient.record|"
    r"seismic|geophys|oceanograph|"
    r"educational.data|student.perform|"
    r"blockchain|smart.contract|\bdefi\b|decentralized.finance|cryptocurrency|"
    r"jailbreak|guardrail|red.teaming|llm.safety|"
    r"safe.alignment|safety.tuning|harmful.content|toxicity"
)

INCLUDE_RE = re.compile(
    "|".join(_aiml_include) if _aiml_include else _DEFAULT_INCLUDE,
    re.IGNORECASE,
)

EXCLUDE_RE = re.compile(
    "|".join(_aiml_exclude) if _aiml_exclude else _DEFAULT_EXCLUDE,
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Security pipeline constants
# ---------------------------------------------------------------------------

_sec_cfg = _domains_cfg.get("security", {})

SECURITY_KEYWORDS = re.compile(
    r"\b(?:attack|vulnerability|exploit|fuzzing|fuzz|malware|"
    r"intrusion|ransomware|phishing|adversarial|"
    r"defense|defence|secure|security|privacy|"
    r"cryptograph|authentication|authorization|"
    r"injection|xss|csrf|cve\-\d|penetration.test|"
    r"threat|anomaly.detect|ids\b|ips\b|firewall|"
    r"reverse.engineer|obfuscat|sandbox|"
    r"side.channel|buffer.overflow|zero.day|"
    r"botnet|rootkit|trojan|worm)\b",
    re.IGNORECASE,
)

ADJACENT_CATEGORIES = ["cs.AI", "cs.SE", "cs.NI", "cs.DC", "cs.OS", "cs.LG"]

SECURITY_EXCLUDE_RE = re.compile(
    r"blockchain|smart.contract|\bdefi\b|decentralized.finance|"
    r"memecoin|meme.coin|cryptocurrency.trading|\bnft\b|"
    r"comprehensive.survey|systematization.of.knowledge|"
    r"differential.privacy.(?:mechanism|framework)|"
    r"stock.predict|financial.forecast|crop.yield|"
    r"sentiment.analysis|educational.data",
    re.IGNORECASE,
)

SECURITY_LLM_RE = re.compile(
    r"jailbreak|guardrail|red.teaming|"
    r"llm.safety|safe.alignment|safety.tuning|"
    r"harmful.(?:content|output)|toxicity|content.moderation|"
    r"prompt.injection|"
    r"reward.model.(?:for|safety|alignment)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dynamic scoring prompt builder
# ---------------------------------------------------------------------------

def _build_scoring_prompt(domain: str, axes: list[dict], preferences: dict) -> str:
    """Build a Claude scoring prompt from config axes + preferences."""
    boost = preferences.get("boost_topics", [])
    penalize = preferences.get("penalize_topics", [])

    if domain == "aiml":
        return _build_aiml_prompt(axes, boost, penalize)
    elif domain == "security":
        return _build_security_prompt(axes, boost, penalize)
    return ""


def _build_aiml_prompt(axes: list[dict], boost: list[str], penalize: list[str]) -> str:
    """Generate AI/ML scoring prompt from axes config."""
    axis_fields = []
    axis_section = []
    for i, ax in enumerate(axes, 1):
        name = ax.get("name", f"axis_{i}")
        desc = ax.get("description", "")
        field = name.lower().replace(" ", "_").replace("&", "and").replace("/", "_").replace("-", "_")
        axis_fields.append(field)
        axis_section.append(f"{i}. **{field}** — {name}: {desc}")

    boost_line = ", ".join(boost) if boost else (
        "New architectures, open-weight models, breakthrough methods, "
        "papers with code AND weights, efficiency improvements"
    )
    penalize_line = ", ".join(penalize) if penalize else (
        "Surveys, incremental SOTA, closed-model papers, "
        "medical/climate/remote sensing applications"
    )

    return f"""\
You are an AI/ML research analyst. Score each paper on three axes (1-10):

{chr(10).join(axis_section)}

Scoring preferences:
- Score UP: {boost_line}
- Score DOWN: {penalize_line}

Use HF ecosystem signals: hf_upvotes > 50 means community interest; hf_models present = weights available;
hf_spaces = demo exists; github_repo = code available; source "both" = higher visibility.

Also provide:
- **summary**: 2-3 sentence practitioner-focused summary.
- **reasoning**: 1-2 sentences explaining scoring.
- **code_url**: Extract GitHub/GitLab URL from abstract/comments if present, else null.

Respond with a JSON array of objects, one per paper, each with fields:
arxiv_id, {", ".join(axis_fields)}, summary, reasoning, code_url
"""


def _build_security_prompt(axes: list[dict], boost: list[str], penalize: list[str]) -> str:
    """Generate security scoring prompt from axes config."""
    axis_fields = []
    axes_section = []
    for i, ax in enumerate(axes, 1):
        name = ax.get("name", f"axis_{i}")
        desc = ax.get("description", "")
        field = name.lower().replace(" ", "_").replace("&", "and").replace("/", "_").replace("-", "_")
        axis_fields.append(field)
        axes_section.append(f"{i}. **{field}** (1-10) — {name}: {desc}")

    return f"""\
You are a security research analyst. Score each paper on three axes (1-10).

=== HARD RULES (apply BEFORE scoring) ===

1. If the paper is primarily about LLM safety, alignment, jailbreaking, guardrails,
   red-teaming LLMs, or making AI models safer: cap ALL three axes at 3 max.
   Check the "llm_adjacent" field — if true, this rule almost certainly applies.

2. If the paper is a survey, SoK, or literature review: cap {axis_fields[1] if len(axis_fields) > 1 else 'axis_2'} at 2 max.

3. If the paper is about blockchain, DeFi, cryptocurrency, smart contracts: cap ALL three axes at 2 max.

4. If the paper is about theoretical differential privacy or federated learning
   without concrete security attacks: cap ALL three axes at 3 max.

=== SCORING AXES ===

{chr(10).join(axes_section)}

=== OUTPUT ===

For each paper also provide:
- **summary**: 2-3 sentence practitioner-focused summary.
- **reasoning**: 1-2 sentences explaining your scoring.
- **code_url**: Extract GitHub/GitLab URL from abstract/comments if present, else null.

Respond with a JSON array of objects, one per paper, each with fields:
entry_id, {", ".join(axis_fields)}, summary, reasoning, code_url
"""


# ---------------------------------------------------------------------------
# Scoring configs per domain
# ---------------------------------------------------------------------------

def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weight values so they sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _build_scoring_configs() -> dict:
    """Build SCORING_CONFIGS from config.yaml or defaults."""
    configs = {}

    # AI/ML config
    aiml_axes_cfg = _aiml_cfg.get("scoring_axes", [
        {"name": "Code & Weights", "weight": 0.30, "description": "Open weights on HF, code on GitHub"},
        {"name": "Novelty", "weight": 0.35, "description": "Paradigm shifts over incremental"},
        {"name": "Practical Applicability", "weight": 0.35, "description": "Usable by practitioners soon"},
    ])
    aiml_prefs = _aiml_cfg.get("preferences", {})
    aiml_weight_keys = ["code_weights", "novelty", "practical"]
    aiml_weights = {}
    for i, ax in enumerate(aiml_axes_cfg):
        key = aiml_weight_keys[i] if i < len(aiml_weight_keys) else f"axis_{i+1}"
        aiml_weights[key] = ax.get("weight", 1.0 / len(aiml_axes_cfg))
    aiml_weights = _normalize_weights(aiml_weights)

    # Generate axis field names dynamically (same transform as prompt builder)
    aiml_axis_fields = [
        ax.get("name", f"axis_{i+1}").lower().replace(" ", "_").replace("&", "and").replace("/", "_").replace("-", "_")
        for i, ax in enumerate(aiml_axes_cfg)
    ]

    configs["aiml"] = {
        "weights": aiml_weights,
        "axes": aiml_axis_fields,
        "axis_labels": [ax.get("name", f"Axis {i+1}") for i, ax in enumerate(aiml_axes_cfg)],
        "prompt": _build_scoring_prompt("aiml", aiml_axes_cfg, aiml_prefs),
    }

    # Security config
    sec_axes_cfg = _sec_cfg.get("scoring_axes", [
        {"name": "Has Code/PoC", "weight": 0.25, "description": "Working tools, repos, artifacts"},
        {"name": "Novel Attack Surface", "weight": 0.40, "description": "First-of-kind research"},
        {"name": "Real-World Impact", "weight": 0.35, "description": "Affects production systems"},
    ])
    sec_prefs = _sec_cfg.get("preferences", {})
    sec_weight_keys = ["code", "novelty", "impact"]
    sec_weights = {}
    for i, ax in enumerate(sec_axes_cfg):
        key = sec_weight_keys[i] if i < len(sec_weight_keys) else f"axis_{i+1}"
        sec_weights[key] = ax.get("weight", 1.0 / len(sec_axes_cfg))
    sec_weights = _normalize_weights(sec_weights)

    sec_axis_fields = [
        ax.get("name", f"axis_{i+1}").lower().replace(" ", "_").replace("&", "and").replace("/", "_").replace("-", "_")
        for i, ax in enumerate(sec_axes_cfg)
    ]

    configs["security"] = {
        "weights": sec_weights,
        "axes": sec_axis_fields,
        "axis_labels": [ax.get("name", f"Axis {i+1}") for i, ax in enumerate(sec_axes_cfg)],
        "prompt": _build_scoring_prompt("security", sec_axes_cfg, sec_prefs),
    }

    return configs


SCORING_CONFIGS = _build_scoring_configs()

# ---------------------------------------------------------------------------
# Events config
# ---------------------------------------------------------------------------

RSS_FEEDS = _cfg.get("rss_feeds", [
    {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml", "category": "news"},
    {"name": "Anthropic Blog", "url": "https://www.anthropic.com/rss.xml", "category": "news"},
    {"name": "Google DeepMind", "url": "https://deepmind.google/blog/rss.xml", "category": "news"},
    {"name": "Meta AI", "url": "https://ai.meta.com/blog/rss/", "category": "news"},
    {"name": "HuggingFace Blog", "url": "https://huggingface.co/blog/feed.xml", "category": "news"},
    {"name": "Krebs on Security", "url": "https://krebsonsecurity.com/feed/", "category": "news"},
    {"name": "The Record", "url": "https://therecord.media/feed", "category": "news"},
    {"name": "Microsoft Security", "url": "https://www.microsoft.com/en-us/security/blog/feed/", "category": "news"},
])

CONFERENCES = _cfg.get("conferences", [
    # AI/ML conferences
    {"name": "NeurIPS 2026", "url": "https://neurips.cc/", "domain": "aiml",
     "deadline": "2026-05-16", "date": "2026-12-07",
     "description": "Conference on Neural Information Processing Systems. Top venue for ML."},
    {"name": "ICML 2026", "url": "https://icml.cc/", "domain": "aiml",
     "deadline": "2026-01-23", "date": "2026-07-19",
     "description": "International Conference on Machine Learning. Premier ML venue."},
    {"name": "ICLR 2026", "url": "https://iclr.cc/", "domain": "aiml",
     "deadline": "2025-10-01", "date": "2026-04-24",
     "description": "International Conference on Learning Representations."},
    {"name": "CVPR 2026", "url": "https://cvpr.thecvf.com/", "domain": "aiml",
     "deadline": "2025-11-14", "date": "2026-06-15",
     "description": "IEEE/CVF Conference on Computer Vision and Pattern Recognition."},
    {"name": "ACL 2026", "url": "https://www.aclweb.org/", "domain": "aiml",
     "deadline": "2026-02-20", "date": "2026-08-02",
     "description": "Annual Meeting of the Association for Computational Linguistics."},
    {"name": "EMNLP 2026", "url": "https://emnlp.org/", "domain": "aiml",
     "deadline": "2026-06-01", "date": "2026-12-08",
     "description": "Conference on Empirical Methods in Natural Language Processing."},
    {"name": "AAAI 2027", "url": "https://aaai.org/conference/aaai/", "domain": "aiml",
     "deadline": "2026-08-15", "date": "2027-02-22",
     "description": "AAAI Conference on Artificial Intelligence."},
    {"name": "COLM 2026", "url": "https://colmweb.org/", "domain": "aiml",
     "deadline": "2026-03-28", "date": "2026-10-06",
     "description": "Conference on Language Modeling. Focused on LLMs."},
    # Security conferences
    {"name": "IEEE S&P 2026", "url": "https://www.ieee-security.org/TC/SP/", "domain": "security",
     "deadline": "2026-06-05", "date": "2026-05-18",
     "description": "IEEE Symposium on Security and Privacy (Oakland). Premier security venue."},
    {"name": "USENIX Security 2026", "url": "https://www.usenix.org/conference/usenixsecurity/", "domain": "security",
     "deadline": "2026-02-04", "date": "2026-08-12",
     "description": "USENIX Security Symposium. Top systems security venue."},
    {"name": "CCS 2026", "url": "https://www.sigsac.org/ccs/", "domain": "security",
     "deadline": "2026-05-01", "date": "2026-11-09",
     "description": "ACM Conference on Computer and Communications Security."},
    {"name": "NDSS 2027", "url": "https://www.ndss-symposium.org/", "domain": "security",
     "deadline": "2026-04-17", "date": "2027-02-23",
     "description": "Network and Distributed System Security Symposium."},
    {"name": "Black Hat USA 2026", "url": "https://www.blackhat.com/", "domain": "security",
     "deadline": "2026-04-01", "date": "2026-08-04",
     "description": "Black Hat USA. Industry security conference with briefings and training."},
    {"name": "DEF CON 34", "url": "https://defcon.org/", "domain": "security",
     "deadline": "2026-05-01", "date": "2026-08-06",
     "description": "DEF CON hacker conference. Villages, CTF, talks."},
])

# ---------------------------------------------------------------------------
# GitHub projects (OSSInsight) config
# ---------------------------------------------------------------------------

OSSINSIGHT_API = "https://api.ossinsight.io/v1"

_github_cfg = _cfg.get("github", {})

OSSINSIGHT_COLLECTIONS = {}
for _coll in _github_cfg.get("collections", []):
    if isinstance(_coll, dict):
        OSSINSIGHT_COLLECTIONS[_coll["id"]] = (_coll["name"], _coll.get("domain", "aiml"))
    elif isinstance(_coll, int):
        OSSINSIGHT_COLLECTIONS[_coll] = (str(_coll), "aiml")

if not OSSINSIGHT_COLLECTIONS:
    OSSINSIGHT_COLLECTIONS = {
        # AI/ML
        10010: ("Artificial Intelligence", "aiml"),
        10076: ("LLM Tools", "aiml"),
        10098: ("AI Agent Frameworks", "aiml"),
        10087: ("LLM DevTools", "aiml"),
        10079: ("Stable Diffusion Ecosystem", "aiml"),
        10075: ("ChatGPT Alternatives", "aiml"),
        10094: ("Vector Database", "aiml"),
        10095: ("GraphRAG", "aiml"),
        10099: ("MCP Client", "aiml"),
        10058: ("MLOps Tools", "aiml"),
        # Security
        10051: ("Security Tool", "security"),
        10082: ("Web Scanner", "security"),
    }

# Languages to scan in trending endpoint
OSSINSIGHT_TRENDING_LANGUAGES = ["Python", "Rust", "Go", "TypeScript", "C++"]

# Keywords for filtering trending repos into AI/ML vs Security domains
GITHUB_AIML_KEYWORDS = re.compile(
    r"machine.learn|deep.learn|neural.net|transformer|llm|large.language|"
    r"diffusion|generat.ai|gpt|bert|llama|vision.model|multimodal|"
    r"reinforcement.learn|computer.vision|nlp|natural.language|"
    r"text.to|speech.to|image.generat|video.generat|"
    r"fine.tun|training|inference|quantiz|embedding|vector|"
    r"rag|retrieval.augment|agent|langchain|"
    r"hugging.?face|pytorch|tensorflow|jax|"
    r"stable.diffusion|comfyui|ollama|vllm|"
    r"tokeniz|dataset|benchmark|model.serv|mlops",
    re.IGNORECASE,
)

GITHUB_SECURITY_KEYWORDS = re.compile(
    r"security|pentest|penetration.test|vulnerability|exploit|"
    r"fuzzing|fuzz|malware|scanner|scanning|"
    r"intrusion|ransomware|phishing|"
    r"reverse.engineer|decompil|disassembl|"
    r"ctf|capture.the.flag|"
    r"firewall|ids\b|ips\b|siem|"
    r"password|credential|auth|"
    r"xss|csrf|injection|"
    r"osint|reconnaissance|recon|"
    r"forensic|incident.response|"
    r"encryption|cryptograph|"
    r"burp|nuclei|nmap|metasploit|wireshark",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_enabled_domains() -> list[str]:
    """Return list of enabled domain keys."""
    if not _domains_cfg:
        return ["aiml", "security"]
    return [k for k, v in _domains_cfg.items() if v.get("enabled", True)]


def is_pipeline_enabled(pipeline: str) -> bool:
    """Check if a pipeline is enabled.

    For 'aiml'/'security': checks domain enabled flag.
    For 'github'/'events': checks feature enabled flag.
    """
    if pipeline in ("aiml", "security"):
        if not _domains_cfg:
            return True
        return _domains_cfg.get(pipeline, {}).get("enabled", True)
    if pipeline in ("github", "events"):
        return _cfg.get(pipeline, {}).get("enabled", True)
    return False


def get_domain_label(domain: str) -> str:
    """Return human-readable label for a domain."""
    if _domains_cfg and domain in _domains_cfg:
        return _domains_cfg[domain].get("label", domain.upper())
    return {"aiml": "AI/ML", "security": "Security"}.get(domain, domain.upper())


def save_config(data: dict):
    """Write config data to config.yaml."""
    import yaml
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    log.info("Config saved to %s", CONFIG_PATH)
    global _cfg, FIRST_RUN, SCORING_CONFIGS, SCORING_MODEL, RESCORE_MODEL, RESCORE_TOP_N, BATCH_SIZE
    _cfg = data
    FIRST_RUN = False
    # Reload scoring model settings
    _sc = data.get("scoring", {})
    SCORING_MODEL = _sc.get("model", data.get("claude_model", "claude-haiku-4-5-20251001"))
    RESCORE_MODEL = _sc.get("rescore_model", "claude-sonnet-4-5-20250929")
    RESCORE_TOP_N = _sc.get("rescore_top_n", 15)
    BATCH_SIZE = _sc.get("batch_size", data.get("batch_size", 20))
    SCORING_CONFIGS.update(_build_scoring_configs())
