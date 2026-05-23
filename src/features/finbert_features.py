"""
finbert_features.py  (v2.1)

FinBERT-based semantic similarity between consecutive filings.
"""
import os
import gc
import hashlib
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_device = None

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 510
CHUNK_STRIDE = MAX_TOKENS
BATCH_SIZE = 8
CACHE_FLUSH_EVERY = 5


def _detect_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _device = _detect_device()
        logger.info(f"Loading {MODEL_NAME} on device={_device}...")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        model.to(_device)
        _model = model

        if _device == "mps":
            with torch.no_grad():
                dummy = _tokenizer("warmup", return_tensors="pt", truncation=True, max_length=16).to(_device)
                _ = model(**dummy)

        logger.info("FinBERT loaded.")
        return _model, _tokenizer, _device

    except Exception as e:
        logger.warning(f"Could not load FinBERT: {e}. finbert_cosine_sim will be NaN.")
        return None, None, None


def _sha1_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _chunk_text_by_tokens(tokenizer, text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    enc = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(enc) <= max_tokens:
        return [text]
    chunks = []
    for start in range(0, len(enc), CHUNK_STRIDE):
        chunk_ids = enc[start: start + max_tokens]
        if len(chunk_ids) < 16:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


def _embed_batch(model, tokenizer, device, chunks: List[str]) -> np.ndarray:
    import torch
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_TOKENS + 2, return_tensors="pt",
            ).to(device)
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_emb = summed / counts
            all_embeds.append(mean_emb.cpu().numpy())
    return np.vstack(all_embeds)


def _embed_document(model, tokenizer, device, text: str) -> Optional[np.ndarray]:
    chunks = _chunk_text_by_tokens(tokenizer, text)
    if not chunks:
        return None
    try:
        chunk_embeds = _embed_batch(model, tokenizer, device, chunks)
        return chunk_embeds.mean(axis=0).astype(np.float32)
    except Exception as e:
        logger.warning(f"Embedding failed for a document: {e}")
        return None


def _load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception as e:
        logger.warning(f"Cache load failed ({cache_path}): {e}. Starting fresh.")
        return {}


def _save_cache(cache_path: str, cache: dict) -> None:
    """Save cache. np.savez_compressed auto-adds .npz so we save directly."""
    if not cache:
        return
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # cache_path ends in .npz; np.savez_compressed will save to that exact path
    # because the suffix already matches
    base = cache_path[:-4] if cache_path.endswith(".npz") else cache_path
    np.savez_compressed(base, **cache)


def compute_finbert_similarity(
    df: pd.DataFrame,
    text_col: str,
    ticker: str,
    cache_dir: str = "data/raw/finbert_cache",
) -> pd.Series:
    model, tokenizer, device = _load_model()
    if model is None:
        return pd.Series([np.nan] * len(df), index=df.index)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_embeddings.npz")
    cache = _load_cache(cache_path)

    texts = df[text_col].fillna("").tolist()
    embeddings: List[Optional[np.ndarray]] = []
    new_count = 0

    for text in texts:
        if not text.strip():
            embeddings.append(None)
            continue

        key = _sha1_hash(text)
        if key in cache:
            embeddings.append(cache[key])
            continue

        emb = _embed_document(model, tokenizer, device, text)
        embeddings.append(emb)

        if emb is not None:
            cache[key] = emb
            new_count += 1
            if new_count % CACHE_FLUSH_EVERY == 0:
                _save_cache(cache_path, cache)

    if new_count > 0:
        _save_cache(cache_path, cache)

    sims = [np.nan]
    for i in range(1, len(embeddings)):
        e_prev, e_curr = embeddings[i - 1], embeddings[i]
        if e_prev is None or e_curr is None:
            sims.append(np.nan)
            continue
        denom = float(np.linalg.norm(e_prev) * np.linalg.norm(e_curr))
        if denom < 1e-10:
            sims.append(np.nan)
        else:
            sims.append(float(np.dot(e_prev, e_curr) / denom))

    if device == "mps":
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

    return pd.Series(sims, index=df.index)
