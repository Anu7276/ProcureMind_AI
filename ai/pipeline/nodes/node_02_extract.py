"""
Node 02 — Extract

LLM call: converts normalised text → structured requirement dict.
Uses category_keyword_maps.json as a disambiguation hint passed to the prompt.
Output shape: {product, material, specifications, performance_requirements,
               safety_requirements, application, category_hint}
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from ai.knowledge import knowledge_loader as kl
from ai.llm.llm_factory import get_llm
from ai.llm.prompts import EXTRACTION_PROMPT
from ai.pipeline.state import PipelineState

logger = logging.getLogger(__name__)

_EMPTY_REQ: Dict[str, Any] = {
    "product": None,
    "material": None,
    "specifications": None,
    "performance_requirements": None,
    "safety_requirements": None,
    "application": None,
    "category_hint": None,
}


def _build_category_hints(text: str) -> str:
    """Build a brief hint string from category maps for the prompt."""
    # Infer likely category from keyword voting
    inferred = kl.infer_category_from_text(text)
    domains = list(kl.DOMAIN_KEYWORDS.keys())
    hint = f"Likely domain: {inferred}. " if inferred else ""
    hint += f"Available domains: {', '.join(domains)}"
    return hint


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as exc:
        logger.warning("Node02: JSON parse failed: %s | raw=%s", exc, raw[:200])
    return None


async def node_02_extract(state: PipelineState) -> dict:
    """LLM extraction of structured requirement from normalised text."""
    warnings = list(state.get("pipeline_warnings", []))
    stages = list(state.get("stages_completed", []))

    text = state.get("normalized_text", "")
    if not text.strip():
        warnings.append("Node02: Empty text — returning empty requirement")
        stages.append("extract")
        return {
            "structured_requirement": _EMPTY_REQ,
            "stages_completed": stages,
            "pipeline_warnings": warnings,
        }

    category_hints = _build_category_hints(text)

    try:
        llm = get_llm()
        chain = EXTRACTION_PROMPT | llm
        response = await chain.ainvoke(
            {"text": text[:4000], "category_hints": category_hints}
        )
        raw_content = response.content if hasattr(response, "content") else str(response)
        requirement = _parse_llm_json(raw_content)

        if requirement is None:
            warnings.append(
                "Node02: LLM returned unparseable JSON — using empty requirement. "
                f"Raw (first 200 chars): {raw_content[:200]}"
            )
            requirement = _EMPTY_REQ
        else:
            # Fill in missing keys
            for k in _EMPTY_REQ:
                requirement.setdefault(k, None)
            logger.info(
                "Node02: extracted requirement — product=%s category_hint=%s",
                requirement.get("product"),
                requirement.get("category_hint"),
            )

    except Exception as exc:
        logger.error("Node02: LLM call failed: %s", exc)
        warnings.append(f"Node02: LLM extraction failed ({exc}) — using empty requirement")
        requirement = _EMPTY_REQ

    stages.append("extract")
    return {
        "structured_requirement": requirement,
        "stages_completed": stages,
        "pipeline_warnings": warnings,
    }
