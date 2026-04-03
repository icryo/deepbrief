"""
Stage 2: Generate editorial narration script with visual queries.

Takes source document text + available image captions, and produces a
~3 minute narration script highlighting what makes the paper interesting.
Each chunk maps to a specific available image.
"""

import json
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage


SCRIPT_SYSTEM_PROMPT = """You are a science communicator who makes research papers compelling and accessible. You create narration scripts for short (~3 minute) video summaries.

Your tone is clear, direct, and engaging — like a knowledgeable friend explaining why a paper is worth reading. Not a textbook. Not hype. Substance delivered accessibly.

Your output must be valid JSON — an array of chunk objects."""

SCRIPT_USER_PROMPT = """You are writing the narration for a ~3 minute video summary of a research paper. The goal is to help a viewer quickly understand what makes this paper interesting and why its contribution matters.

## Available images

These are the actual figures and images extracted from the paper. You MUST reference these when writing visual_query — do not invent figures that don't exist. Each image has an ID and a caption describing what it shows.

{image_inventory}

## Structure

Follow this arc (approximate timings for ~3 minutes total):

1. **Hook** (15-20s, 1-2 chunks): What problem does this solve? Why should I care? Lead with the impact, not the title.
2. **Gap** (15-25s, 1-2 chunks): What existed before and what was missing? Set up the tension.
3. **Key insight** (30-45s, 2-4 chunks): What is the novel idea? Explain the core contribution in plain language.
4. **How it works** (40-60s, 3-5 chunks): Walk through the method using the paper's own figures. Be specific but accessible.
5. **Results** (20-30s, 2-3 chunks): Does it actually work? Highlight the most impressive result. Reference result figures/tables.
6. **Why it matters** (15-20s, 1-2 chunks): What does this enable? Where could this go?

Aim for 12-20 chunks total. Each chunk = one image on screen.

## Output format

Output strictly as a JSON array:
```json
[
  {{
    "chunk_id": 1,
    "narration": "What if you could turn any research paper into a presentation video automatically?",
    "visual_query": "image_3",
    "visual_query_description": "architecture diagram showing the full pipeline from paper input to video output",
    "keywords": ["pipeline", "architecture", "video generation"]
  }}
]
```

Rules:
- `visual_query` MUST be an image ID from the inventory above (e.g., "image_0", "image_5")
- `visual_query_description` is a fallback description in case the exact image doesn't match well
- Each chunk's narration should be 15-50 words — tight, spoken-word pacing
- Don't waste chunks on boilerplate ("In this paper, the authors..."). Get to the point.
- Use the paper's actual figures to drive the narrative — if the paper has a great results table, build a chunk around it
- It's OK to reuse an image across 2 consecutive chunks if the narration is still discussing that figure
- But don't use the same image more than 2 times total

## Document text
---
{document_text}
---"""


def generate_script(document_text, model_config, image_captions=None, max_text_length=50000):
    """
    Generate an editorial narration script from document text.

    Args:
        document_text: Full text of the source document.
        model_config: CAMEL model config dict for the LLM.
        image_captions: List of dicts with "id" and "caption" for available images.
            If None, falls back to description-only visual queries.
        max_text_length: Truncate document text if longer than this.

    Returns:
        (chunks, usage) where chunks is a list of dicts with
        chunk_id, narration, visual_query, visual_query_description, keywords.
    """
    if len(document_text) > max_text_length:
        document_text = document_text[:max_text_length] + "\n\n[Document truncated...]"

    # Build image inventory string
    if image_captions:
        inventory_lines = []
        for img in image_captions:
            inventory_lines.append(f"- **{img['id']}**: {img['caption']}")
        image_inventory = "\n".join(inventory_lines)
    else:
        image_inventory = "(No image inventory available — write visual_query_description only)"

    model = ModelFactory.create(
        model_platform=model_config["model_platform"],
        model_type=model_config["model_type"],
        model_config_dict=model_config.get("model_config"),
        url=model_config.get("url", None),
    )
    agent = ChatAgent(model=model, system_message=SCRIPT_SYSTEM_PROMPT)

    prompt = SCRIPT_USER_PROMPT.format(
        document_text=document_text,
        image_inventory=image_inventory,
    )
    message = BaseMessage.make_user_message(role_name="user", content=prompt, meta_dict={})
    response = agent.step(message)

    raw = response.msg.content.strip()
    chunks = _parse_script_json(raw)

    print(f"[ScriptGen] Generated {len(chunks)} narration chunks")
    return chunks, response.info.get("usage", {})


def _parse_script_json(raw_text):
    """Parse JSON from LLM response, handling markdown code fences."""
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    chunks = json.loads(text)

    for i, chunk in enumerate(chunks):
        if "chunk_id" not in chunk:
            chunk["chunk_id"] = i + 1
        if "narration" not in chunk:
            raise ValueError(f"Chunk {i} missing 'narration' field")
        if "visual_query" not in chunk:
            chunk["visual_query"] = ""
        if "visual_query_description" not in chunk:
            chunk["visual_query_description"] = chunk.get("visual_query", "")
        if "keywords" not in chunk:
            chunk["keywords"] = []

    return chunks


def save_script(chunks, save_path):
    """Save script chunks to JSON file."""
    with open(save_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"[ScriptGen] Saved script to {save_path}")


def load_script(load_path):
    """Load script chunks from JSON file."""
    with open(load_path, "r") as f:
        return json.load(f)
