"""
Stage 1: Extract images from PDF documents and caption them with a VLM.

Extracts embedded raster images and renders full pages for vector figures.
Each image gets a VLM-generated caption for downstream semantic matching.
Embedded images are preferred; page renders are fallback only.
"""

import os
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage


def extract_images_from_pdf(pdf_path, output_dir, min_size=100, page_dpi=200, composite_threshold=5):
    """
    Extract images from a PDF. Three strategies per page:
    1. If a page has few (< composite_threshold) large embedded images → extract them
    2. If a page has many small embedded images (composite figure) → render the full page
    3. If a page has no embedded images → render the full page as fallback

    Returns:
        List of dicts: [{"id": str, "path": str, "source": str, "page": int}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted = []
    image_counter = 0

    # First pass: analyze each page's image composition
    page_img_dir = os.path.join(output_dir, "pages")
    os.makedirs(page_img_dir, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)

        # Deduplicate by xref
        seen_xrefs = set()
        page_images = []
        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base_image = doc.extract_image(xref)
                if base_image is None:
                    continue
                pil_img = Image.open(BytesIO(base_image["image"]))
                if pil_img.width >= min_size and pil_img.height >= min_size:
                    page_images.append((xref, base_image, pil_img))
            except Exception as e:
                print(f"[Warning] Failed to read xref={xref} on page {page_idx + 1}: {e}")

        # Decide strategy for this page
        if len(page_images) == 0:
            # No images — render full page as fallback (skip text-only pages with little content)
            text = page.get_text("text").strip()
            if len(text) < 100:
                continue  # Near-empty page, skip entirely
            strategy = "page_render"
        elif len(page_images) >= composite_threshold:
            # Many images — likely a composite figure grid. Use page render.
            strategy = "page_render"
            print(f"  Page {page_idx + 1}: {len(page_images)} images → composite, using page render")
        else:
            # Few large images — extract individually
            strategy = "extract"

        if strategy == "extract":
            for xref, base_image, pil_img in page_images:
                filename = f"image_{image_counter}.png"
                save_path = os.path.join(output_dir, filename)
                if pil_img.mode == "RGBA":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(save_path, "PNG")
                extracted.append({
                    "id": f"image_{image_counter}",
                    "path": save_path,
                    "source": "embedded",
                    "page": page_idx + 1,
                })
                image_counter += 1
        else:
            # Render full page
            mat = fitz.Matrix(page_dpi / 72, page_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            filename = f"page_{page_idx + 1}.png"
            save_path = os.path.join(page_img_dir, filename)
            pix.save(save_path)
            extracted.append({
                "id": f"image_{image_counter}",
                "path": save_path,
                "source": "page",
                "page": page_idx + 1,
            })
            image_counter += 1

    doc.close()

    n_embedded = sum(1 for r in extracted if r["source"] == "embedded")
    n_pages = sum(1 for r in extracted if r["source"] == "page")
    print(f"[ImageExtractor] {n_embedded} embedded images + {n_pages} page fallbacks = {len(extracted)} total")
    return extracted


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF, page by page."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        pages_text.append({"page": page_idx + 1, "text": text})
    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages_text)
    return full_text, pages_text


CAPTION_PROMPT = (
    "You are captioning an image extracted from a scientific or technical document. "
    "Your caption will be used for semantic search — someone will query with a concept "
    "and your caption must contain the right terms to match.\n\n"
    "Analyze this image thoroughly and provide:\n\n"
    "1. FIGURE TYPE: Exactly what kind of figure this is (e.g., bar chart, line plot, "
    "architecture diagram, flow chart, table, photograph, scatter plot, heatmap, "
    "confusion matrix, network diagram, pseudocode, equation, screenshot, etc.)\n\n"
    "2. CONTENT: Describe what the figure shows. For charts/plots: what are the axes, "
    "what trends are visible, what are the key data points? For diagrams: what are the "
    "components and how do they connect? For tables: what are the columns and key findings?\n\n"
    "3. TEXT IN IMAGE: Transcribe ALL visible text — titles, axis labels, legend entries, "
    "annotations, column headers, node labels, arrow labels. This is critical.\n\n"
    "4. KEY CONCEPTS: What scientific/technical concepts does this figure illustrate? "
    "Use the specific domain terminology.\n\n"
    "Write 3-6 dense sentences. Prioritize specificity and domain terms over generic descriptions. "
    "Someone searching for 'attention mechanism performance comparison' should find a figure "
    "that shows exactly that."
)


def _caption_with_anthropic(image_records, model_name="claude-sonnet-4-20250514", use_bedrock=False):
    """Caption images using the Anthropic SDK directly (bypasses CAMEL's broken image handling)."""
    import anthropic
    import base64

    if use_bedrock:
        client = anthropic.AnthropicBedrock()
    else:
        client = anthropic.Anthropic()
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for record in image_records:
        try:
            with open(record["path"], "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            # Detect media type
            path_lower = record["path"].lower()
            if path_lower.endswith(".png"):
                media_type = "image/png"
            elif path_lower.endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            else:
                media_type = "image/png"

            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system="You are an expert at describing scientific and technical figures for retrieval.",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }],
            )
            record["caption"] = response.content[0].text.strip()
            total_usage["input_tokens"] += response.usage.input_tokens
            total_usage["output_tokens"] += response.usage.output_tokens

            print(f"  [Caption] {record['id']}: {record['caption'][:80]}...")
        except Exception as e:
            print(f"  [Warning] Failed to caption {record['path']}: {e}")
            record["caption"] = ""

    return image_records, total_usage


def _caption_with_camel(image_records, model_config):
    """Caption images using the CAMEL framework (for non-Anthropic models)."""
    from camel.models import ModelFactory
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage

    model = ModelFactory.create(
        model_platform=model_config["model_platform"],
        model_type=model_config["model_type"],
        model_config_dict=model_config.get("model_config"),
        url=model_config.get("url", None),
    )
    agent = ChatAgent(
        model=model,
        system_message="You are an expert at describing scientific and technical figures for retrieval.",
    )

    total_usage = {}
    for record in image_records:
        try:
            img = Image.open(record["path"])
            if img.mode == "RGBA":
                img = img.convert("RGB")

            message = BaseMessage.make_user_message(
                role_name="user",
                content=CAPTION_PROMPT,
                image_list=[img],
                meta_dict={},
            )
            response = agent.step(message)
            record["caption"] = response.msg.content.strip()

            usage = response.info.get("usage", {})
            for k, v in usage.items():
                total_usage[k] = total_usage.get(k, 0) + (v if isinstance(v, (int, float)) else 0)

            print(f"  [Caption] {record['id']}: {record['caption'][:80]}...")
        except Exception as e:
            print(f"  [Warning] Failed to caption {record['path']}: {e}")
            record["caption"] = ""

    return image_records, total_usage


def caption_images(image_records, model_config):
    """
    Generate captions for extracted images using a VLM.

    Uses the Anthropic SDK directly for Claude models (CAMEL's image handling
    is broken for Anthropic), falls back to CAMEL for other providers.

    Returns:
        Updated image_records with "caption" field added, and total token usage.
    """
    from camel.types import ModelPlatformType
    platform = model_config.get("model_platform")
    if platform in (ModelPlatformType.ANTHROPIC, ModelPlatformType.AWS_BEDROCK):
        model_name = str(model_config["model_type"])
        if hasattr(model_config["model_type"], "value"):
            model_name = model_config["model_type"].value
        use_bedrock = platform == ModelPlatformType.AWS_BEDROCK
        label = "Bedrock" if use_bedrock else "Anthropic"
        print(f"[Caption] Using {label} SDK directly (model={model_name})")
        return _caption_with_anthropic(image_records, model_name=model_name, use_bedrock=use_bedrock)
    else:
        return _caption_with_camel(image_records, model_config)
