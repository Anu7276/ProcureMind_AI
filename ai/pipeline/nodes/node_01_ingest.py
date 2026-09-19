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
from typing import List

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

# Pattern to catch explicit "IS XXXX" / "IS/IEC XXXX" mentions in text
_IS_CODE_RE = re.compile(
    r"\bIS(?:/IEC)?\s+\d[\d\s\(\)Part/Sec\.]+",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _extract_literal_codes(text: str) -> List[str]:
    """Extract any IS codes literally mentioned in the text."""
    matches = _IS_CODE_RE.findall(text)
    # Normalise: strip extra whitespace
    return [re.sub(r"\s+", " ", m.strip()) for m in matches]


async def node_01_ingest(state: PipelineState) -> dict:
    """Normalise input, detect language, expand via thesaurus."""
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
            "thesaurus_expansions": [],
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    # 1. Language detection
    lang = _detect_language(raw_text)
    logger.info("Node01: detected language=%s, text_len=%d", lang, len(raw_text))

    # 2. Thesaurus expansion
    expansions: List[str] = kl.expand_query_with_thesaurus(raw_text)

    # 3. Literal IS code extraction
    literal_codes = _extract_literal_codes(raw_text)
    if literal_codes:
        logger.info("Node01: found literal IS codes: %s", literal_codes)
        # Add any found keys not already in expansions
        for code in literal_codes:
            normalised_key = code.split(":")[0].strip()
            if normalised_key not in expansions:
                expansions.append(normalised_key)

    # 4. Normalise: strip extra whitespace, unify unicode spaces
    normalized = " ".join(raw_text.split())

    # Append expansion keywords to help Node02 extraction
    if expansions:
        expansion_hint = " [Related IS codes from thesaurus: " + ", ".join(expansions) + "]"
        normalized_with_hint = normalized + expansion_hint
    else:
        normalized_with_hint = normalized

    stages.append("ingest")
    return {
        "normalized_text": normalized_with_hint,
        "language": lang,
        "thesaurus_expansions": expansions,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
