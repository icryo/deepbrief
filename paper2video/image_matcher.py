"""
Stage 3: Match narration chunks to extracted images.

Two matching modes:
- Direct: LLM already assigned image IDs in the script (preferred path)
- Semantic fallback: CLIP + BM25 for chunks without a valid image ID

Includes duplicate suppression to avoid showing the same image repeatedly.
"""

import os
import math
import torch
import numpy as np
from PIL import Image
from collections import Counter


def match_chunks_to_images(
    script_chunks,
    image_records,
    clip_model_name="openai/clip-vit-large-patch14",
    clip_weight=0.6,
    caption_weight=0.4,
    max_consecutive=2,
    device=None,
):
    """
    Match each narration chunk to the best image.

    First tries direct assignment (LLM wrote an image ID in visual_query).
    Falls back to CLIP + BM25 semantic matching for unresolved chunks.
    Applies duplicate suppression to avoid the same image N times in a row.

    Args:
        script_chunks: List of dicts with "visual_query" and "visual_query_description".
        image_records: List of dicts with "id", "path", "caption", "source".
        clip_model_name: HuggingFace model for semantic fallback.
        clip_weight: Weight for CLIP vs BM25 in semantic fallback.
        caption_weight: Weight for BM25 caption matching.
        max_consecutive: Max times the same image can appear in a row.
        device: torch device.

    Returns:
        List of image record indices, one per chunk.
    """
    # Build ID → index lookup
    id_to_idx = {r["id"]: i for i, r in enumerate(image_records)}

    # Phase 1: Resolve direct assignments from LLM
    matches = [None] * len(script_chunks)
    needs_semantic = []

    for i, chunk in enumerate(script_chunks):
        vq = chunk.get("visual_query", "")
        if vq in id_to_idx:
            matches[i] = id_to_idx[vq]
        else:
            needs_semantic.append(i)

    resolved = sum(1 for m in matches if m is not None)
    print(f"[ImageMatcher] {resolved}/{len(script_chunks)} chunks resolved by direct ID assignment")

    # Phase 2: Semantic fallback for unresolved chunks
    if needs_semantic:
        print(f"[ImageMatcher] Running semantic matching for {len(needs_semantic)} remaining chunks...")
        semantic_matches = _semantic_match(
            [script_chunks[i] for i in needs_semantic],
            image_records,
            clip_model_name, clip_weight, caption_weight, device,
        )
        for chunk_pos, img_idx in zip(needs_semantic, semantic_matches):
            matches[chunk_pos] = img_idx

    # Phase 3: Duplicate suppression
    matches = _suppress_consecutive_duplicates(matches, image_records, max_consecutive)

    # Log final assignments
    for i, idx in enumerate(matches):
        rec = image_records[idx]
        print(f"  Chunk {i + 1}: → {rec['id']} ({rec['source']}, p{rec['page']})")

    return [[m] for m in matches]  # Wrap in lists for compatibility with assembly


def _semantic_match(chunks, image_records, clip_model_name, clip_weight, caption_weight, device):
    """CLIP + BM25 semantic matching. Returns list of image indices."""
    from transformers import CLIPProcessor, CLIPModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ImageMatcher] Loading CLIP model...")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    # Embed images
    image_embeddings = []
    for record in image_records:
        try:
            img = Image.open(record["path"]).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            image_embeddings.append(emb.cpu().numpy()[0])
        except Exception as e:
            print(f"  [Warning] Failed to embed {record['path']}: {e}")
            image_embeddings.append(np.zeros(model.config.projection_dim))
    image_embeddings = np.array(image_embeddings)

    # Embed text queries
    text_embeddings = []
    for chunk in chunks:
        query = chunk.get("visual_query_description", chunk.get("narration", ""))
        inputs = processor(text=[query], return_tensors="pt", truncation=True, max_length=77).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(emb.cpu().numpy()[0])
    text_embeddings = np.array(text_embeddings)

    # CLIP scores
    clip_scores = text_embeddings @ image_embeddings.T
    clip_scores_norm = _normalize_rows(clip_scores)

    # BM25 over captions
    captions = [r.get("caption", "") for r in image_records]
    caption_tokens_list = [_tokenize(c) for c in captions]

    bm25_scores = np.zeros_like(clip_scores)
    for i, chunk in enumerate(chunks):
        query = chunk.get("visual_query_description", "") + " " + " ".join(chunk.get("keywords", []))
        bm25_scores[i] = _bm25_score(_tokenize(query), caption_tokens_list)
    bm25_scores_norm = _normalize_rows(bm25_scores)

    # Penalize page renders — prefer embedded figures
    source_penalty = np.array([0.0 if r["source"] == "embedded" else -0.15 for r in image_records])
    combined = clip_weight * clip_scores_norm + caption_weight * bm25_scores_norm + source_penalty

    results = [int(np.argmax(combined[i])) for i in range(len(chunks))]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def _suppress_consecutive_duplicates(matches, image_records, max_consecutive):
    """
    If the same image appears more than max_consecutive times in a row,
    replace extras with the next-best option from the same page or
    a nearby page's image.
    """
    if len(matches) <= 1:
        return matches

    result = list(matches)
    for i in range(len(result)):
        # Count how many times this image has appeared consecutively
        run_start = i
        while run_start > 0 and result[run_start - 1] == result[i]:
            run_start -= 1
        run_length = i - run_start + 1

        if run_length > max_consecutive:
            # Find an alternative — prefer images from nearby pages
            current_page = image_records[result[i]]["page"]
            candidates = []
            for j, rec in enumerate(image_records):
                if j == result[i]:
                    continue
                page_dist = abs(rec["page"] - current_page)
                source_bonus = 0 if rec["source"] == "embedded" else 5
                candidates.append((page_dist + source_bonus, j))
            candidates.sort()

            # Pick the closest alternative that isn't the current image
            if candidates:
                result[i] = candidates[0][1]

    return result


def _normalize_rows(scores):
    """Normalize each row of a score matrix to [0, 1]."""
    mins = scores.min(axis=1, keepdims=True)
    maxs = scores.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (scores - mins) / ranges


def _tokenize(text):
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _bm25_score(query_tokens, doc_tokens_list, k1=1.5, b=0.75):
    """BM25 scoring over tokenized documents."""
    n_docs = len(doc_tokens_list)
    avg_dl = np.mean([len(d) for d in doc_tokens_list]) if doc_tokens_list else 1.0

    df = Counter()
    for doc_tokens in doc_tokens_list:
        for token in set(doc_tokens):
            df[token] += 1

    scores = np.zeros(n_docs)
    for i, doc_tokens in enumerate(doc_tokens_list):
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)
        for token in query_tokens:
            if token not in tf:
                continue
            f = tf[token]
            idf = math.log((n_docs - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
            tf_score = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
            scores[i] += idf * tf_score

    return scores
