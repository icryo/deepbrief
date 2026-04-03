# DeepBrief

Research intelligence dashboard with automated paper-to-video generation.

- **Dashboard** — Scores and surfaces AI/ML and security papers from arXiv and HuggingFace
- **Paper2Video** — Generates narrated video summaries from any paper with one click
- **Preferences** — Learns from your signals (upvote, save, dismiss) to personalize rankings

## Setup

```bash
cp .env.example .env
# Edit .env with your AWS credentials

docker compose up -d --build
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | AWS credentials for Bedrock + Polly |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS credentials |
| `AWS_DEFAULT_REGION` | No | Defaults to `us-east-1` |
| `USE_BEDROCK` | No | `true` (default) to use Bedrock for scoring |
| `ANTHROPIC_API_KEY` | No | Alternative to Bedrock |
| `GITHUB_TOKEN` | No | Higher GitHub API rate limits |
| `WEB_PORT` | No | Dashboard port inside container (default `8888`) |
| `WEB_PORT_HOST` | No | Dashboard port on host (default `9090`) |
| `P2V_PORT` | No | Paper2Video API port (default `8001`) |

## Paper2Video CLI (standalone)

The pipeline can also be run directly without the dashboard:

```bash
cd paper2video
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# AWS: Bedrock + Polly
python pipeline_editorial.py \
    --pdf_path paper.pdf \
    --bedrock --tts_engine polly --voice Ruth

# Local: Anthropic API + Orpheus TTS
python pipeline_editorial.py \
    --pdf_path paper.pdf \
    --model_name claude-sonnet --voice tara
```

## Project structure

```
├── src/                       # Dashboard
│   ├── web/app.py             # FastAPI routes + HTMX
│   ├── scoring.py             # Claude scoring (Bedrock or direct)
│   ├── pipelines/             # Data pipelines (aiml, security, github, events)
│   ├── db.py                  # SQLite schema + queries
│   └── config.py              # Configuration
├── paper2video/               # Video generation pipeline
│   ├── pipeline_editorial.py  # 5-stage pipeline CLI
│   ├── api.py                 # FastAPI job server
│   ├── tts_gen.py             # Orpheus or Polly TTS
│   └── ...
├── data/                      # SQLite DB + config (gitignored)
├── docker-compose.yml
└── Dockerfile
```
