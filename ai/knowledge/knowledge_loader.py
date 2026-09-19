"""
Knowledge loader — loads all static data files into memory at startup.

Loaded once and kept as module-level singletons.
All pipeline nodes import from here — no file I/O during request processing.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from backend.config.settings import settings

logger = logging.getLogger(__name__)


# ── Module-level singletons (populated by load_all()) ────────────────────────

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

# QCO in-memory index: standard_key → qco record (for Node04 fast lookup)
QCO_BY_STANDARD_KEY: Dict[str, Dict[str, Any]] = {}

# Certification schemes index: scheme_code → record
CERT_SCHEMES: Dict[str, Dict[str, Any]] = {}

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

    logger.info(
        "Knowledge loaded: %d standards, %d whitelist codes, "
        "%d QCO records, %d cert schemes",
        len(STANDARDS_BY_KEY),
        len(IS_CODE_WHITELIST),
        len(QCO_BY_STANDARD_KEY),
        len(CERT_SCHEMES),
    )
    _loaded = True


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


# ── Helper functions used by pipeline nodes ───────────────────────────────────

def get_standard(key: str) -> Optional[Dict[str, Any]]:
    """Fast in-memory lookup — no DB round-trip."""
    return STANDARDS_BY_KEY.get(key)


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


def get_qco_for_key(standard_key: str) -> Optional[Dict[str, Any]]:
    """In-memory QCO lookup for Node04 (Postgres-free path)."""
    return QCO_BY_STANDARD_KEY.get(standard_key)


def get_cert_scheme(scheme_code: str) -> Optional[Dict[str, Any]]:
    """In-memory cert scheme lookup."""
    return CERT_SCHEMES.get(scheme_code)
