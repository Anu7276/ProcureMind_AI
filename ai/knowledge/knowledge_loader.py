"""
Knowledge loader — loads all static data files into memory at startup.

Loaded once and kept as module-level singletons.
All pipeline nodes import from here — no file I/O during request processing.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

from backend.config.settings import settings

logger = logging.getLogger(__name__)


# ── BM25 Lexical Index ───────────────────────────────────────────────────────

class BM25Hit(NamedTuple):
    score: float       # raw BM25 score
    doc_key: str
    ideal_score: float
    coverage: float

    @property
    def raw_score(self) -> float:
        return self.score

    @property
    def key(self) -> str:
        return self.doc_key


class BM25Index:
    """
    Pure Python BM25 lexical index with field weighting (BM25F-style).
    Weights:
      - title x 3.0
      - keywords x 2.0
      - scope x 1.5
      - category / subcategory x 1.0
      - title_hindi x 1.0
      - key / display_code x 2.0
    Parameters:
      k1 = 1.5, b = 0.75
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, float] = {}
        self.avg_doc_len: float = 1.0
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.corpus_size: int = 0

    def build(self, standards_by_key: Dict[str, Dict[str, Any]]) -> None:
        from ai.knowledge.text_utils import tokenize

        self.corpus_size = len(standards_by_key)
        if self.corpus_size == 0:
            return

        self.doc_len.clear()
        self.inverted_index.clear()
        self.idf.clear()

        field_specs = [
            (lambda s: s.get("title") or "", 3.0),
            (lambda s: " ".join(s.get("keywords") or []), 2.0),
            (lambda s: s.get("scope") or "", 1.5),
            (lambda s: f"{s.get('category') or ''} {s.get('subcategory') or ''}", 1.0),
            (lambda s: s.get("title_hindi") or "", 1.0),
            (lambda s: f"{s.get('key') or ''} {s.get('display_code') or ''}", 2.0),
        ]

        doc_frequencies: Dict[str, int] = defaultdict(int)
        total_length = 0.0

        for key, std in standards_by_key.items():
            doc_tf: Dict[str, float] = defaultdict(float)
            length = 0.0

            for extract_fn, weight in field_specs:
                text = extract_fn(std)
                tokens = tokenize(text)
                if not tokens:
                    continue
                length += len(tokens) * weight
                for t in tokens:
                    doc_tf[t] += weight

            self.doc_len[key] = length
            total_length += length

            for t, w_tf in doc_tf.items():
                self.inverted_index[t][key] = w_tf
                doc_frequencies[t] += 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 1.0

        for term, df in doc_frequencies.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query_text: str, top_k: int = 10) -> List[BM25Hit]:
        from ai.knowledge.text_utils import tokenize

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        doc_scores: Dict[str, float] = defaultdict(float)

        # Default IDF for OOV tokens (when df=0)
        default_idf = math.log((self.corpus_size + 0.5) / 0.5 + 1.0) if self.corpus_size > 0 else 1.0

        # ideal_score: sum over query tokens of the best possible single-term score for that token given idf and k1
        token_idfs = {t: self.idf.get(t, default_idf) for t in query_tokens}
        ideal_score = sum((self.k1 + 1.0) * token_idfs[t] for t in query_tokens)
        if ideal_score <= 0.0:
            ideal_score = 1.0

        for t in query_tokens:
            if t not in self.inverted_index:
                continue
            idf_val = self.idf.get(t, 0.0)
            if idf_val <= 0:
                continue

            postings = self.inverted_index[t]
            for doc_key, w_tf in postings.items():
                dl = self.doc_len.get(doc_key, self.avg_doc_len)
                denom = w_tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                term_score = idf_val * (w_tf * (self.k1 + 1.0)) / denom
                doc_scores[doc_key] += term_score

        if not doc_scores:
            return []

        unique_tokens = set(query_tokens)
        total_token_idf = sum(token_idfs[t] for t in unique_tokens)
        if total_token_idf <= 0:
            total_token_idf = 1.0

        scored = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        hits: List[BM25Hit] = []
        for doc_key, score in scored[:top_k]:
            matched_idf = sum(
                token_idfs[t] for t in unique_tokens if doc_key in self.inverted_index.get(t, {})
            )
            coverage = matched_idf / total_token_idf
            hits.append(BM25Hit(
                score=score,
                doc_key=doc_key,
                ideal_score=ideal_score,
                coverage=coverage,
            ))
        return hits

    def score_document(self, query_text: str, doc_key: str) -> Tuple[float, float, float]:
        """
        Compute (raw_score, ideal_score, coverage) for a single document key against query_text.
        """
        from ai.knowledge.text_utils import tokenize

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return (0.0, 1.0, 0.0)

        default_idf = math.log((self.corpus_size + 0.5) / 0.5 + 1.0) if self.corpus_size > 0 else 1.0
        token_idfs = {t: self.idf.get(t, default_idf) for t in query_tokens}
        ideal_score = sum((self.k1 + 1.0) * token_idfs[t] for t in query_tokens)
        if ideal_score <= 0.0:
            ideal_score = 1.0

        dl = self.doc_len.get(doc_key, self.avg_doc_len)
        raw_score = 0.0
        for t in query_tokens:
            if t not in self.inverted_index:
                continue
            idf_val = self.idf.get(t, 0.0)
            if idf_val <= 0:
                continue
            w_tf = self.inverted_index[t].get(doc_key, 0.0)
            if w_tf > 0:
                denom = w_tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                raw_score += idf_val * (w_tf * (self.k1 + 1.0)) / denom

        unique_tokens = set(query_tokens)
        total_token_idf = sum(token_idfs[t] for t in unique_tokens)
        if total_token_idf <= 0:
            total_token_idf = 1.0
        matched_idf = sum(
            token_idfs[t] for t in unique_tokens if doc_key in self.inverted_index.get(t, {})
        )
        coverage = matched_idf / total_token_idf
        return (raw_score, ideal_score, coverage)


# ── Module-level singletons (populated by load_all()) ────────────────────────

BM25_INDEX = BM25Index(k1=1.5, b=0.75)

# Full standards dict: key → full record
STANDARDS_BY_KEY: Dict[str, Dict[str, Any]] = {}

# Whitelist: set of canonical display_codes for hallucination guard
IS_CODE_WHITELIST: Set[str] = set()

# Procurement thesaurus sub-maps
CPWD_DSR_MAP: Dict[str, Any] = {}       # item_code → {standards, keywords, ...}
GEM_TAXONOMY_MAP: Dict[str, Any] = {}   # product → {standards, ...}
HINDI_SYNONYMS: Dict[str, List[str]] = {}  # hindi_term → [english_terms]

# Category keyword maps
DOMAIN_KEYWORDS: Dict[str, List[str]] = {}   # domain → [keywords]
CATEGORY_KEYWORDS: Dict[str, Any] = {}        # BIS AI category → ...
DOMAIN_TO_CATALOG_CATEGORIES: Dict[str, List[str]] = {} # domain → [catalog category strings]
NEGATIVE_KEYWORD_RULES: List[Dict[str, Any]] = []      # trigger_keywords → penalize_standards

# QCO in-memory index: standard_key → qco record (for Node04 fast lookup)
QCO_BY_STANDARD_KEY: Dict[str, Dict[str, Any]] = {}

# Certification schemes index: scheme_code → record
CERT_SCHEMES: Dict[str, Dict[str, Any]] = {}

# Relationships index: standard_key → list of related standards
RELATIONSHIPS_BY_KEY: Dict[str, List[Dict[str, Any]]] = {}

# Bidirectional successors and predecessors index
SUCCESSORS_BY_KEY: Dict[str, List[str]] = {}
PREDECESSORS_BY_KEY: Dict[str, List[str]] = {}

# Amendments index
AMENDMENTS_BY_KEY: Dict[str, List[Dict[str, Any]]] = {}

_loaded = False


def load_all() -> None:
    """
    Call once at application startup (FastAPI lifespan).
    Idempotent — safe to call multiple times.
    """
    global _loaded
    if _loaded:
        return

    _load_standards()
    _load_whitelist()
    _load_thesaurus()
    _load_category_maps()
    _load_qco()
    _load_cert_schemes()
    _load_relationships()
    _build_bm25_index()
    _build_version_indices()

    logger.info(
        "Knowledge loaded: %d standards, %d whitelist codes, "
        "%d QCO records, %d cert schemes, %d standards with relationships, BM25 index built (%d docs)",
        len(STANDARDS_BY_KEY),
        len(IS_CODE_WHITELIST),
        len(QCO_BY_STANDARD_KEY),
        len(CERT_SCHEMES),
        len(RELATIONSHIPS_BY_KEY),
        BM25_INDEX.corpus_size,
    )
    _loaded = True


def _build_version_indices() -> None:
    global SUCCESSORS_BY_KEY, PREDECESSORS_BY_KEY, AMENDMENTS_BY_KEY
    succ_map = defaultdict(set)
    pred_map = defaultdict(set)

    for key, std in STANDARDS_BY_KEY.items():
        # 1. From superseded_by
        raw_succ = std.get("superseded_by") or []
        if isinstance(raw_succ, str):
            raw_succ = [raw_succ]
        for s in raw_succ:
            if s:
                succ_map[key].add(s)
                pred_map[s].add(key)

        # 2. From supersedes
        raw_pred = std.get("supersedes") or []
        if isinstance(raw_pred, str):
            raw_pred = [raw_pred]
        for p in raw_pred:
            if p:
                pred_map[key].add(p)
                succ_map[p].add(key)

        # 3. From relationships
        for r in RELATIONSHIPS_BY_KEY.get(key, []):
            rel_type = (r.get("relationship_type") or "").lower()
            t_key = r.get("key")
            if not t_key:
                continue
            if rel_type == "superseded_by":
                succ_map[key].add(t_key)
                pred_map[t_key].add(key)
            elif rel_type == "supersedes":
                pred_map[key].add(t_key)
                succ_map[t_key].add(key)

    SUCCESSORS_BY_KEY = {k: sorted(list(v)) for k, v in succ_map.items()}
    PREDECESSORS_BY_KEY = {k: sorted(list(v)) for k, v in pred_map.items()}

    # Load optional amendments.json
    amend_path = Path("BIS_Sahayak_Clean_Data/clean/standards/amendments.json")
    AMENDMENTS_BY_KEY = defaultdict(list)
    if amend_path.exists():
        try:
            with open(amend_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "key" in item:
                            AMENDMENTS_BY_KEY[item["key"]].append(item)
        except Exception as exc:
            logger.warning("Could not load amendments.json: %s", exc)


def _build_bm25_index() -> None:
    global BM25_INDEX
    BM25_INDEX.build(STANDARDS_BY_KEY)


def _load_standards() -> None:
    path = settings.standards_json
    with open(path, encoding="utf-8") as f:
        data: List[Dict] = json.load(f)
    for rec in data:
        key = rec.get("key", "")
        if key:
            STANDARDS_BY_KEY[key] = rec


def _load_whitelist() -> None:
    path = settings.whitelist_json
    with open(path, encoding="utf-8") as f:
        codes: List[str] = json.load(f)
    IS_CODE_WHITELIST.update(codes)
    # Also add plain keys like "IS 1786" (stripped of year)
    for code in codes:
        base = code.split(":")[0].strip()
        IS_CODE_WHITELIST.add(base)


def _load_thesaurus() -> None:
    path = settings.thesaurus_json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    CPWD_DSR_MAP.update(data.get("cpwd_dsr_2023_mapping", {}))
    GEM_TAXONOMY_MAP.update(data.get("gem_taxonomy_mapping", {}))
    HINDI_SYNONYMS.update(data.get("hindi_synonyms", {}))


def _load_category_maps() -> None:
    path = settings.category_map_json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    DOMAIN_KEYWORDS.update(data.get("sure_domain_taxonomy", {}))
    CATEGORY_KEYWORDS.update(data.get("bis_ai_category_keywords", {}))
    DOMAIN_TO_CATALOG_CATEGORIES.update(data.get("domain_to_catalog_categories", {}))
    NEGATIVE_KEYWORD_RULES.clear()
    NEGATIVE_KEYWORD_RULES.extend(data.get("negative_keyword_rules", []))


def _load_qco() -> None:
    path = settings.qco_json
    with open(path, encoding="utf-8") as f:
        data: List[Dict] = json.load(f)
    for rec in data:
        key = rec.get("standard_key", "")
        if key:
            QCO_BY_STANDARD_KEY[key] = rec


def _load_cert_schemes() -> None:
    path = settings.certification_schemes_json
    with open(path, encoding="utf-8") as f:
        data: List[Dict] = json.load(f)
    for rec in data:
        code = rec.get("scheme_code", "")
        if code:
            CERT_SCHEMES[code] = rec


def _load_relationships() -> None:
    path = settings.relationships_json
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data: List[Dict] = json.load(f)
        for rel in data:
            frm = (rel.get("frm") or "").strip()
            to = (rel.get("to") or "").strip()
            if frm and to:
                RELATIONSHIPS_BY_KEY.setdefault(frm, []).append({
                    "key": to,
                    "display_code": to,
                    "title": rel.get("to_title") or "",
                    "relationship_type": rel.get("type", "references"),
                })
    except Exception as exc:
        logger.warning("Could not load relationships JSON: %s", exc)


# ── Helper functions used by pipeline nodes ───────────────────────────────────

def get_standard(key: str) -> Optional[Dict[str, Any]]:
    """Fast in-memory lookup — no DB round-trip."""
    return STANDARDS_BY_KEY.get(key)


def get_related_standards_in_memory(key: str) -> List[Dict[str, Any]]:
    """Return related standards from in-memory index."""
    return RELATIONSHIPS_BY_KEY.get(key, [])


def validate_whitelist(code: str) -> bool:
    """
    Returns True if code is on the whitelist.
    Checks both display_code (with year) and bare key (without year).
    """
    if code in IS_CODE_WHITELIST:
        return True
    # Try stripping year
    bare = code.split(":")[0].strip()
    return bare in IS_CODE_WHITELIST


def expand_query_with_thesaurus(text: str) -> List[str]:
    """
    Returns additional search terms derived from the thesaurus for the
    procurement text. Used in Node01.
    """
    text_lower = text.lower()
    expansions: List[str] = []

    # Check CPWD DSR entries
    for _item_code, entry in CPWD_DSR_MAP.items():
        keywords = entry.get("keywords", [])
        if any(kw.lower() in text_lower for kw in keywords):
            expansions.extend(entry.get("standards", []))

    # Check GeM taxonomy
    for product, entry in GEM_TAXONOMY_MAP.items():
        if product.lower() in text_lower:
            expansions.extend(entry.get("standards", []))

    # Hindi synonym expansion
    for hindi_term, english_terms in HINDI_SYNONYMS.items():
        if hindi_term in text:
            expansions.extend(english_terms)

    return list(dict.fromkeys(expansions))  # deduplicate, preserve order


def infer_category_from_text(text: str) -> Optional[str]:
    """
    Simple keyword vote — returns best-matching domain category.
    Used in Node02 as disambiguation hint for the LLM.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return None
    return max(scores, key=scores.__getitem__)


def map_category_hint_to_catalog(category_hint: Optional[str]) -> List[str]:
    """Map a domain category hint (from Node 02) to corresponding catalog category strings."""
    if not category_hint:
        return []
    hint = category_hint.strip()
    if hint in DOMAIN_TO_CATALOG_CATEGORIES:
        return DOMAIN_TO_CATALOG_CATEGORIES[hint]
    for dom, cats in DOMAIN_TO_CATALOG_CATEGORIES.items():
        if dom.lower() == hint.lower() or dom.lower() in hint.lower():
            return cats
    return [hint]


def search_standards_in_memory(query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    In-memory fallback search when external vector/Postgres stores are unreachable.
    Scores standards using pure Python BM25 over weighted fields with stopword removal.
    Normalises scores to [0, 1] by dividing by the max score in the result list.
    Also returns raw_score, ideal_score, coverage, and absolute match_strength.
    """
    from ai.knowledge.text_utils import tokenize

    tokens = tokenize(query_text)
    if not tokens:
        return []

    global BM25_INDEX
    if BM25_INDEX.corpus_size == 0 and STANDARDS_BY_KEY:
        BM25_INDEX.build(STANDARDS_BY_KEY)

    scored = BM25_INDEX.search(query_text, top_k=top_k)
    if not scored:
        return []

    max_score = scored[0].score
    results = []
    for hit in scored:
        std = STANDARDS_BY_KEY.get(hit.doc_key)
        if not std:
            continue
        dq = std.get("data_quality") or {}
        norm_score = round(hit.score / max_score, 4) if max_score > 0 else 0.0
        ms = 0.5 * (hit.score / hit.ideal_score if hit.ideal_score > 0 else 0.0) + 0.5 * hit.coverage
        match_strength = round(min(1.0, max(0.0, ms)), 4)
        results.append({
            "key": std["key"],
            "display_code": std.get("display_code", std["key"]),
            "title": std.get("title", ""),
            "category": std.get("category"),
            "subcategory": std.get("subcategory"),
            "verification_level": dq.get("verification_level", "single_source_unconfirmed"),
            "status": std.get("status", "ACTIVE"),
            "superseded_by": std.get("superseded_by") or [],
            "flags": dq.get("flags", []),
            "score": norm_score,
            "raw_score": round(hit.score, 4),
            "ideal_score": round(hit.ideal_score, 4),
            "coverage": round(hit.coverage, 4),
            "match_strength": match_strength,
            "source": "in_memory_catalog",
            "related_standards": RELATIONSHIPS_BY_KEY.get(std["key"], []),
        })
    return results


def get_qco_for_key(standard_key: str) -> Optional[Dict[str, Any]]:
    """In-memory QCO lookup for Node04 (Postgres-free path)."""
    return QCO_BY_STANDARD_KEY.get(standard_key)


def get_cert_scheme(scheme_code: str) -> Optional[Dict[str, Any]]:
    """In-memory cert scheme lookup."""
    return CERT_SCHEMES.get(scheme_code)


def check_standard_version(candidate_record: Dict[str, Any], cited_year: Optional[int] = None) -> Dict[str, Any]:
    """Helper to check version, successors, and amendment status for a standard."""
    from ai.knowledge.version_checker import check_version
    global SUCCESSORS_BY_KEY, PREDECESSORS_BY_KEY, AMENDMENTS_BY_KEY
    return check_version(
        candidate_record=candidate_record,
        cited_year=cited_year,
        all_records=STANDARDS_BY_KEY,
        successors_map=SUCCESSORS_BY_KEY,
        predecessors_map=PREDECESSORS_BY_KEY,
        amendments_map=AMENDMENTS_BY_KEY,
    )


