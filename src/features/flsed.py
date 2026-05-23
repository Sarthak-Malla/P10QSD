"""
flsed.py - Forward-Looking Sentence Embedding Drift

Captures sentence-level forward-looking signal in 10-Q Risk Factors:
  1. Identify sentences that are genuinely NEW (not in last 8 quarters for same company)
  2. Score each new sentence by: forward-looking density × specificity × FinBERT sentiment
  3. Detect cross-company novelty (sector-wide story vs company-specific signal)

Outputs 4 filing-level features:
  - n_new_sentences          : count of truly new sentences (above novelty threshold)
  - avg_finbert_sent_new     : mean FinBERT pos-neg score of new sentences
  - avg_fwd_specificity_new  : mean (forward_looking * specificity) of new sentences
  - peer_sentence_overlap    : fraction of new sentences also appearing in peer filings

Designed for incremental, resumable execution. All expensive computations cached.
"""
import os
import re
import gc
import json
import hashlib
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────
_nlp = None              # spaCy pipeline
_finbert_cls = None      # FinBERT classification model
_finbert_tok = None
_finbert_dev = None

# ── Config ───────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"  # has 3-way pos/neu/neg classification head
MIN_SENT_LEN = 6                     # ignore sentences shorter than this (in words)
MAX_SENT_LEN = 80                    # truncate very long sentences (rarely useful as units)
HISTORY_QUARTERS = 8                 # how many prior quarters to dedupe against
SENT_HASH_LEN = 16                   # truncate hex hash for memory
NEW_NOVELTY_THRESHOLD = 0.5          # min novelty score to count as "truly new"
BATCH_SIZE = 32                      # FinBERT classification batch
FLUSH_EVERY = 5                      # save cache every N tickers processed

# Forward-looking term patterns (compiled once)
_FORWARD_PATTERNS = re.compile(
    r"\b("
    r"may|might|could|would|should|will|shall|expect|expects|expected|"
    r"anticipate|anticipates|anticipated|believe|believes|believed|"
    r"plan|plans|planned|forecast|forecasts|forecasted|"
    r"intend|intends|intended|project|projects|projected|"
    r"estimate|estimates|estimated|likely|unlikely|"
    r"future|upcoming|prospective|potential|"
    r"materially|adverse|adversely|"
    r"continue|continues|continuing|going forward"
    r")\b",
    re.IGNORECASE,
)


# ─── spaCy loader ────────────────────────────────────────────────────────
def _load_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            # Use sentencizer for fast sentence splitting (parser is too slow)
            if "sentencizer" not in _nlp.pipe_names:
                _nlp.add_pipe("sentencizer", first=True)
            logger.info("Loaded spaCy en_core_web_sm")
        except OSError:
            # Model not installed — fallback to a minimal pipeline
            from spacy.lang.en import English
            _nlp = English()
            _nlp.add_pipe("sentencizer")
            logger.warning("en_core_web_sm not installed — using basic sentencizer (no NER). "
                           "Install with: python -m spacy download en_core_web_sm")
        return _nlp
    except Exception as e:
        logger.error(f"spaCy load failed: {e}")
        return None


# ─── FinBERT classification loader (DIFFERENT from finbert_features.py!) ──
def _load_finbert_classifier():
    """Load FinBERT for 3-way sentiment classification (pos/neu/neg)."""
    global _finbert_cls, _finbert_tok, _finbert_dev
    if _finbert_cls is not None:
        return _finbert_cls, _finbert_tok, _finbert_dev
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if torch.backends.mps.is_available():
            _finbert_dev = "mps"
        elif torch.cuda.is_available():
            _finbert_dev = "cuda"
        else:
            _finbert_dev = "cpu"
        logger.info(f"Loading FinBERT classifier on device={_finbert_dev}")
        _finbert_tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _finbert_cls = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        _finbert_cls.eval().to(_finbert_dev)
        # Warmup
        if _finbert_dev == "mps":
            with torch.no_grad():
                dummy = _finbert_tok("warmup", return_tensors="pt", truncation=True, max_length=16).to(_finbert_dev)
                _ = _finbert_cls(**dummy)
        return _finbert_cls, _finbert_tok, _finbert_dev
    except Exception as e:
        logger.error(f"FinBERT classifier load failed: {e}")
        return None, None, None


# ─── Sentence utilities ──────────────────────────────────────────────────
def _hash_sentence(text: str) -> str:
    """Normalize and hash a sentence for dedup. Lowercase, strip punctuation, collapse whitespace."""
    norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:SENT_HASH_LEN]


def _segment_sentences(text: str) -> List[str]:
    """Split text into sentences, filter by length."""
    if not isinstance(text, str) or len(text) < 20:
        return []
    nlp = _load_spacy()
    if nlp is None:
        return []
    try:
        doc = nlp(text[:1_000_000])  # spaCy max length safety
        sents = []
        for s in doc.sents:
            s_text = s.text.strip()
            words = s_text.split()
            if MIN_SENT_LEN <= len(words) <= 500:  # skip headers and giant tables
                sents.append(s_text[:1000])         # cap length per sentence
        return sents
    except Exception as e:
        logger.warning(f"Sentence segmentation failed: {e}")
        return []


def _forward_specificity_score(sentence: str, nlp_doc=None) -> Tuple[float, float]:
    """Compute (forward_density, specificity) for a sentence."""
    words = sentence.split()
    n = max(len(words), 1)

    # Forward-looking density
    matches = _FORWARD_PATTERNS.findall(sentence)
    fwd_density = min(len(matches) / n * 10, 1.0)  # cap at 1.0

    # Specificity from NER (if available)
    spec = 0.0
    if nlp_doc is not None and hasattr(nlp_doc, "ents"):
        try:
            ents = list(nlp_doc.ents)
            # Each named entity adds specificity, capped
            named_entity_types = {"GPE", "ORG", "MONEY", "PERCENT", "DATE", "PRODUCT", "PERSON", "NORP"}
            specific_ents = [e for e in ents if e.label_ in named_entity_types]
            spec = min(len(specific_ents) / 3, 1.0)
        except Exception:
            spec = 0.0
    else:
        # Fallback: regex-based specificity
        spec_terms = re.findall(r"\$[\d,.]+|\b\d+(?:\.\d+)?%|\b[A-Z][a-z]+(?:land|nia|stan|ina|key)\b", sentence)
        spec = min(len(spec_terms) / 3, 1.0)

    return float(fwd_density), float(spec)


# ─── FinBERT batch sentiment ─────────────────────────────────────────────
def _finbert_sentiments(sentences: List[str]) -> np.ndarray:
    """Return Nx3 array of (positive, negative, neutral) probabilities."""
    if not sentences:
        return np.zeros((0, 3), dtype=np.float32)
    model, tok, dev = _load_finbert_classifier()
    if model is None:
        return np.full((len(sentences), 3), np.nan, dtype=np.float32)

    import torch
    out = []
    # FinBERT label order is: [positive, negative, neutral]
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i: i + BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.vstack(out).astype(np.float32)


# ─── Per-ticker FLSED computation ────────────────────────────────────────
def compute_flsed_features(
    df: pd.DataFrame,
    text_col: str,
    ticker: str,
    cache_dir: str = "data/raw/flsed_cache",
    peer_sentence_pool: Optional[Dict[pd.Timestamp, set]] = None,
) -> pd.DataFrame:
    """
    For a single ticker (sorted by filed_at), compute 4 FLSED features per row.

    Returns DataFrame with columns:
       n_new_sentences, avg_finbert_sent_new, avg_fwd_specificity_new, peer_sentence_overlap
    indexed identically to df.
    """
    n = len(df)
    out = pd.DataFrame({
        "n_new_sentences":          np.zeros(n, dtype=np.int32),
        "avg_finbert_sent_new":     np.full(n, np.nan, dtype=np.float32),
        "avg_fwd_specificity_new":  np.full(n, np.nan, dtype=np.float32),
        "peer_sentence_overlap":    np.full(n, np.nan, dtype=np.float32),
    }, index=df.index)

    # Per-ticker sentence cache: {sent_hash: {"text": str, "finbert_sent": float, "fwd_spec": float}}
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_flsed.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    else:
        cache = {}

    # Rolling history of sentence hashes (most recent quarters first)
    sent_history: List[set] = []  # list of sets, one per past quarter
    nlp = _load_spacy()

    # Process filings in chronological order
    texts = df[text_col].fillna("").tolist()
    filed_dates = df["filed_at"].tolist() if "filed_at" in df.columns else [None] * n

    for i, text in enumerate(texts):
        if not text.strip():
            sent_history.insert(0, set())
            if len(sent_history) > HISTORY_QUARTERS:
                sent_history.pop()
            continue

        sentences = _segment_sentences(text)
        if not sentences:
            sent_history.insert(0, set())
            if len(sent_history) > HISTORY_QUARTERS:
                sent_history.pop()
            continue

        # Hash all sentences in this filing
        sent_hashes = [_hash_sentence(s) for s in sentences]

        # Identify "new" sentences (not in last HISTORY_QUARTERS quarters)
        prior_pool = set().union(*sent_history) if sent_history else set()
        new_idx = [j for j, h in enumerate(sent_hashes) if h not in prior_pool]
        new_sentences = [sentences[j] for j in new_idx]
        new_hashes = [sent_hashes[j] for j in new_idx]

        # Update history for next iteration BEFORE we filter by novelty
        sent_history.insert(0, set(sent_hashes))
        if len(sent_history) > HISTORY_QUARTERS:
            sent_history.pop()

        if not new_sentences:
            out.iloc[i, out.columns.get_loc("n_new_sentences")] = 0
            continue

        # Compute forward + specificity for NEW sentences only
        # Use cache where possible
        fwd_scores, spec_scores = [], []
        uncached_idx, uncached_sents = [], []
        for j, (sent, h) in enumerate(zip(new_sentences, new_hashes)):
            if h in cache and "fwd_spec" in cache[h]:
                fwd_scores.append(cache[h]["fwd"])
                spec_scores.append(cache[h]["spec"])
            else:
                uncached_idx.append(j)
                uncached_sents.append(sent)
                fwd_scores.append(None)
                spec_scores.append(None)

        # NER-based specificity for uncached (batch through spaCy for speed)
        if uncached_sents and nlp is not None:
            try:
                # pipe handles batching efficiently
                docs = list(nlp.pipe(uncached_sents))
            except Exception:
                docs = [None] * len(uncached_sents)
        else:
            docs = [None] * len(uncached_sents)

        for k, (j, sent, doc) in enumerate(zip(uncached_idx, uncached_sents, docs)):
            fwd, spec = _forward_specificity_score(sent, doc)
            fwd_scores[j] = fwd
            spec_scores[j] = spec

        # FinBERT sentiment for NEW uncached sentences only
        # First check cache
        finbert_sents = []
        uncached_idx2, uncached_sents2 = [], []
        for j, (sent, h) in enumerate(zip(new_sentences, new_hashes)):
            if h in cache and "finbert_sent" in cache[h]:
                finbert_sents.append(cache[h]["finbert_sent"])
            else:
                uncached_idx2.append(j)
                uncached_sents2.append(sent)
                finbert_sents.append(None)

        if uncached_sents2:
            probs = _finbert_sentiments(uncached_sents2)
            # Net sentiment = positive - negative (range [-1, 1])
            for k, j in enumerate(uncached_idx2):
                p_pos, p_neg, p_neu = float(probs[k, 0]), float(probs[k, 1]), float(probs[k, 2])
                net_sent = p_pos - p_neg
                finbert_sents[j] = net_sent

        # Write all uncached entries to cache
        for j, (sent, h) in enumerate(zip(new_sentences, new_hashes)):
            if h not in cache:
                cache[h] = {}
            cache[h]["fwd"] = fwd_scores[j]
            cache[h]["spec"] = spec_scores[j]
            cache[h]["finbert_sent"] = finbert_sents[j]
            cache[h]["fwd_spec"] = True  # marker for "computed"

        # Compute per-sentence novelty score = forward * specificity * |sentiment|
        # Higher score = more novel content
        novelty_scores = [
            float(fwd_scores[j]) * (0.5 + 0.5 * float(spec_scores[j])) *
            (1.0 + abs(float(finbert_sents[j])))
            for j in range(len(new_sentences))
        ]

        # Filter to TRULY new sentences (above threshold)
        truly_new = [(idx, novelty_scores[idx]) for idx in range(len(new_sentences))
                     if novelty_scores[idx] >= NEW_NOVELTY_THRESHOLD]

        # Aggregate to filing-level features
        if truly_new:
            tn_idx = [t[0] for t in truly_new]
            avg_sent = float(np.mean([finbert_sents[k] for k in tn_idx]))
            avg_fs = float(np.mean([fwd_scores[k] * spec_scores[k] for k in tn_idx]))
            out.iloc[i, out.columns.get_loc("n_new_sentences")] = len(truly_new)
            out.iloc[i, out.columns.get_loc("avg_finbert_sent_new")] = avg_sent
            out.iloc[i, out.columns.get_loc("avg_fwd_specificity_new")] = avg_fs
        else:
            out.iloc[i, out.columns.get_loc("n_new_sentences")] = 0
            # Keep NaN — model will impute

        # Cross-company novelty (if peer pool provided)
        if peer_sentence_pool is not None and filed_dates[i] is not None and truly_new:
            tn_hashes = {new_hashes[k] for k, _ in truly_new}
            # Look at peer filings within ±45 days
            filing_date = pd.Timestamp(filed_dates[i])
            peer_set = set()
            for d, hash_set in peer_sentence_pool.items():
                if abs((pd.Timestamp(d) - filing_date).days) <= 45:
                    peer_set |= hash_set
            if tn_hashes:
                overlap = len(tn_hashes & peer_set) / len(tn_hashes)
                out.iloc[i, out.columns.get_loc("peer_sentence_overlap")] = overlap

    # Flush cache
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"Cache write failed for {ticker}: {e}")

    # GPU memory cleanup
    if _finbert_dev == "mps":
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

    return out


def build_sector_peer_pool(
    full_df: pd.DataFrame,
    sector_map: Dict[str, str],
    text_col: str = "section_Item 1A",
) -> Dict[str, Dict[pd.Timestamp, set]]:
    """
    Build {sector: {filing_date: {sent_hashes}}} for peer-overlap lookup.

    Called once across the entire dataset before the per-ticker loop.
    Uses sentence hashes only (no FinBERT) so it's fast.
    """
    logger.info("Building sector peer sentence pool (one-time)...")
    pool: Dict[str, Dict[pd.Timestamp, set]] = {}
    nlp = _load_spacy()
    if nlp is None:
        return pool

    grouped = full_df.groupby("ticker")
    n_total = len(full_df)
    n_done = 0
    for ticker, g in grouped:
        sector = sector_map.get(ticker, "Other")
        if sector == "Other":
            n_done += len(g)
            continue
        for _, row in g.iterrows():
            n_done += 1
            text = row.get(text_col, "")
            if not isinstance(text, str) or not text.strip():
                continue
            sents = _segment_sentences(text)
            if not sents:
                continue
            hashes = {_hash_sentence(s) for s in sents}
            d = pd.Timestamp(row["filed_at"])
            pool.setdefault(sector, {}).setdefault(d, set()).update(hashes)
        if n_done % 50 == 0:
            logger.info(f"  peer pool progress: {n_done}/{n_total}")
    logger.info(f"Peer pool built: {sum(len(v) for v in pool.values())} sector-dates")
    return pool
