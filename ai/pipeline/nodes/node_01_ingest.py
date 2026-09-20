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
from typing import Any, Dict, List

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

REJECTED_UNITS = {
    "mm", "cm", "m", "mtr", "mtrs", "meter", "meters", "metre", "metres",
    "kg", "kgs", "gm", "gms", "gram", "grams",
    "v", "kv", "volt", "volts", "voltage",
    "a", "amp", "amps", "ampere", "amperes", "ka",
    "kw", "kva", "w", "watt", "watts", "mw", "hp",
    "sq", "sqmm", "sqm", "sqft", "sq.mm",
    "nos", "no", "pcs", "pieces",
    "ltr", "ltrs", "litre", "litres", "liter", "liters", "ml",
    "hz", "mhz", "khz", "ghz",
    "rpm", "kn", "mpa", "n/mm2", "bar", "psi"
}

_LITERAL_PATTERN = re.compile(
    r"""
    (?:(?P<cue>\b(?:as\s+per|conforming\s+to|confirming\s+to|conform\s+to|in\s+accordance\s+with|per|to)\b)\s+)?
    (?P<prefix>\bIS(?:/(?:ISO/IEC|IEC|ISO))?|\bis(?:/(?:ISO/IEC|IEC|ISO))?)
    [\s:\-]+
    (?P<number>\d{2,5})
    (?:\s*(?P<part>\(?(?:Part|Pt)[\s\-]*\d+(?:[\s\-/]*(?:Sec|Section)[\s\-]*\d+)?(?:\s*to\s*\d+)?\)?))?
    (?:\s*[:\-\(]\s*(?P<year>(?:19|20)\d{2})(?:\s*\(?Reaffirmed\s*\d{4}\)?)?\s*\)?)?
    """,
    re.VERBOSE | re.IGNORECASE,
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
    Rejects matches followed by units of measurement or lowercase 'is' without cue phrases.
    Resolves each match to catalog key(s).
    """
    if not text:
        return []

    if not kl.STANDARDS_BY_KEY:
        kl.load_all()

    resolved_items: List[Dict[str, Any]] = []
    seen_keys = set()

    for m in _LITERAL_PATTERN.finditer(text):
        raw = text[m.start("prefix"):m.end()].strip()
        cue = m.group("cue")
        prefix_raw = m.group("prefix")
        number = m.group("number")
        part_raw = m.group("part")
        year_raw = m.group("year")

        # 1. Reject lowercase 'is' unless preceded by explicit cue phrase
        is_uppercase = prefix_raw.startswith("IS")
        if not is_uppercase and not cue:
            continue

        # 2. Reject if immediately followed by a unit of measurement
        trailing_text = text[m.end():]
        nw_match = re.match(r"^\s*([a-zA-Z\./]+)", trailing_text)
        if nw_match:
            nw = nw_match.group(1).lower().rstrip(".,;:")
            if nw in REJECTED_UNITS:
                continue

        # 3. Format prefix
        prefix = re.sub(r"\s*/\s*", "/", prefix_raw.upper())

        # 4. Normalize part
        part = None
        if part_raw:
            p_clean = part_raw.strip().strip("()")
            sec_m = re.search(r"(?:Part|Pt)[\s\-]*(\d+)[\s\-/]*(?:Sec|Section)[\s\-]*(\d+)", p_clean, re.IGNORECASE)
            if sec_m:
                part = f"(Part {sec_m.group(1)}/Sec {sec_m.group(2)})"
            else:
                part_m = re.search(r"(?:Part|Pt)[\s\-]*(\d+)", p_clean, re.IGNORECASE)
                if part_m:
                    part = f"(Part {part_m.group(1)})"

        # 5. Normalize year
        cited_year = None
        if year_raw:
            try:
                y = int(year_raw)
                if 1947 <= y <= 2026:
                    cited_year = y
            except ValueError:
                pass

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
