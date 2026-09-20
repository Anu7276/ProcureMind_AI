"""
Node 01 — Ingest

Responsibilities:
  1. Language detection (langdetect)
  2. Thesaurus expansion — CPWD DSR / GeM taxonomy / Hindi synonyms
     → procurement_thesaurus.json (all 3 sub-maps)
  3. Literal IS code extraction (catches "IS 1786"-style mentions)
  4. Normalises text for Node02
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

# Regex capturing prefix (IS | IS/IEC | IS/ISO | IS/ISO/IEC), standard number,
# optional (Part N) or (Part N/Sec M), and optional publication year after ':'.
_IS_CODE_RE = re.compile(
    r"\b(?P<prefix>IS(?:/(?:ISO/IEC|IEC|ISO))?)\s+(?P<number>\d+)(?:\s*(?P<part>\(Part\s+\d+(?:/Sec\s+\d+)?\)))?(?:\s*:\s*(?P<year>\d{4}))?",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def extract_literal_codes(text: str) -> List[Dict[str, Any]]:
    """
    Extract IS codes literally mentioned in text.
    Captures: prefix (IS | IS/IEC | IS/ISO | IS/ISO/IEC), number, optional part, optional year.
    Resolves each match to catalog key(s).
    """
    matches = list(_IS_CODE_RE.finditer(text))
    if not matches:
        return []

    if not kl.STANDARDS_BY_KEY:
        kl.load_all()

    resolved_items: List[Dict[str, Any]] = []
    seen_keys = set()

    for m in matches:
        raw = m.group(0).strip()
        prefix = m.group("prefix").upper()
        prefix = re.sub(r"\s*/\s*", "/", prefix)
        number = m.group("number")
        part_raw = m.group("part")
        part = re.sub(r"\s+", " ", part_raw.strip()) if part_raw else None
        year_raw = m.group("year")
        cited_year = int(year_raw) if year_raw else None

        exact_candidate = f"{prefix} {number} {part}" if part else f"{prefix} {number}"
        family_candidate = f"{prefix} {number}"

        # 1. Try exact key in catalog
        if exact_candidate in kl.STANDARDS_BY_KEY:
            if exact_candidate not in seen_keys:
                seen_keys.add(exact_candidate)
                resolved_items.append({
                    "raw": raw,
                    "key": exact_candidate,
                    "part": part,
                    "cited_year": cited_year,
                })
        # 2. Try key without part (family)
        elif family_candidate in kl.STANDARDS_BY_KEY:
            if family_candidate not in seen_keys:
                seen_keys.add(family_candidate)
                resolved_items.append({
                    "raw": raw,
                    "key": family_candidate,
                    "part": part,
                    "cited_year": cited_year,
                })
        # 3. Family has multiple parts in catalog
        else:
            parts_in_cat = [
                k for k in kl.STANDARDS_BY_KEY
                if k.startswith(family_candidate + " (") or k.startswith(family_candidate + " /")
            ]
            if len(parts_in_cat) == 1:
                k = parts_in_cat[0]
                if k not in seen_keys:
                    seen_keys.add(k)
                    resolved_items.append({
                        "raw": raw,
                        "key": k,
                        "part": part,
                        "cited_year": cited_year,
                    })
            elif len(parts_in_cat) > 1:
                for k in parts_in_cat:
                    if k not in seen_keys:
                        seen_keys.add(k)
                        resolved_items.append({
                            "raw": raw,
                            "key": k,
                            "part": part,
                            "cited_year": cited_year,
                            "ambiguous_part": True,
                        })
            else:
                # Key not in catalog, but user explicitly cited it
                if exact_candidate not in seen_keys:
                    seen_keys.add(exact_candidate)
                    resolved_items.append({
                        "raw": raw,
                        "key": exact_candidate,
                        "part": part,
                        "cited_year": cited_year,
                        "not_in_catalog": True,
                    })

    return resolved_items


async def node_01_ingest(state: PipelineState) -> dict:
    """Normalise input, detect language, extract literal codes, expand via thesaurus."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))

    # Resolve input text — prefer extracted_text (from Node00) if available
    raw_text = state.get("extracted_text") or state.get("raw_input", "")

    if not raw_text or not raw_text.strip():
        warnings.append("Node01: Empty input text received")
        stages.append("ingest")
        return {
            "normalized_text": "",
            "language": "en",
            "literal_codes": [],
            "thesaurus_expansions": [],
            "thesaurus_hint": None,
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    # 1. Language detection
    lang = _detect_language(raw_text)
    logger.info("Node01: detected language=%s, text_len=%d", lang, len(raw_text))

    # 2. Thesaurus expansion (kept strictly separate from literal mentions)
    expansions: List[str] = kl.expand_query_with_thesaurus(raw_text)

    # 3. Literal IS code extraction with cited year and catalog resolution
    literal_codes = extract_literal_codes(raw_text)
    if literal_codes:
        logger.info("Node01: found literal IS codes: %s", [c["key"] for c in literal_codes])

    # 4. Clean normalization (extra whitespace removed, no thesaurus hints appended)
    normalized = " ".join(raw_text.split())

    thesaurus_hint = ", ".join(expansions) if expansions else None

    stages.append("ingest")
    return {
        "normalized_text": normalized,
        "language": lang,
        "literal_codes": literal_codes,
        "thesaurus_expansions": expansions,
        "thesaurus_hint": thesaurus_hint,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
