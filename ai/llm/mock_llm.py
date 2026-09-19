"""
Mock / Local fallback ChatModel for zero-setup execution.

Used automatically when no LLM API key is provided, or when external API
calls fail/timeout/exceed quota. Performs intelligent heuristic extraction
and standard-specific reasoning generation using in-memory BIS data.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai.knowledge import knowledge_loader as kl

logger = logging.getLogger(__name__)


def extract_structured_heuristically(text: str) -> dict:
    """Extract procurement requirements using regex and vocabulary matching."""
    text_lower = text.lower()

    # Inferred category
    category_hint = kl.infer_category_from_text(text) or "Civil & Construction"

    # Material pattern matching
    materials = []
    mat_patterns = [
        r"\b(fe\s*\d{3}[a-z]?)\b",
        r"\b(pvc|upvc|cpvc|hdpe|ppr|gi|ms|stainless\s*steel|ss\s*304|ss\s*316)\b",
        r"\b(copper|aluminium|aluminum|concrete|cement|bitumen|timber|glass)\b",
        r"\b(ordinary\s*portland\s*cement|opc|ppc|fly\s*ash)\b",
        r"\b(high\s*strength\s*deformed\s*steel|tmt)\b",
    ]
    for pat in mat_patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            if m.strip().upper() not in [x.upper() for x in materials]:
                materials.append(m.strip())

    # Specifications pattern matching (diameters, dimensions, voltage, ratings)
    specs = []
    spec_patterns = [
        r"\b(\d+(?:\.\d+)?\s*(?:mm|cm|m|meter|dia|gauge|sqmm|sq\.mm))\b",
        r"\b(\d+(?:\.\d+)?\s*(?:kv|v|volt|watt|w|kw|amp|a|hz|mhz))\b",
        r"\b(\d+(?:\.\d+)?\s*(?:kg|kn|mpa|n/mm2|litre|l))\b",
        r"\b(class\s*[a-z0-9]+|grade\s*[a-z0-9]+|type\s*[a-z0-9]+)\b",
    ]
    for pat in spec_patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches[:4]:
            if m.strip() not in specs:
                specs.append(m.strip())

    # Performance requirements
    perf = []
    if "tensile" in text_lower or "elongation" in text_lower or "strength" in text_lower:
        perf.append("High tensile strength and elongation compliance")
    if "waterproof" in text_lower or "leak" in text_lower:
        perf.append("Water tightness and pressure testing")
    if "fire" in text_lower or "flame" in text_lower:
        perf.append("Fire resistance and flame retardant properties")
    if "efficiency" in text_lower:
        perf.append("Energy efficiency compliance")

    # Safety requirements
    safety = []
    if "isi" in text_lower or "bis" in text_lower or "mark" in text_lower:
        safety.append("Mandatory BIS / ISI Mark certification requirement")
    if "qco" in text_lower:
        safety.append("Quality Control Order (QCO) compliance mandatory")
    if "hazard" in text_lower or "safety" in text_lower:
        safety.append("Workplace safety and non-hazardous operational rating")

    # Product extraction: check known keywords or fall back to first phrase
    product = None
    # Check GeM taxonomy first
    for prod in kl.GEM_TAXONOMY_MAP:
        if prod.lower() in text_lower:
            product = prod
            break

    if not product:
        # Check first line or main noun phrase
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        first_line = lines[0] if lines else text[:60]
        # Remove tender prefixes
        clean_name = re.sub(
            r"^(supply\s+of|procurement\s+of|purchase\s+of|specification\s+for|tender\s+for)\s+",
            "",
            first_line,
            flags=re.IGNORECASE,
        ).strip()
        product = clean_name[:60]

    # Application
    app = None
    if "construction" in text_lower or "building" in text_lower or "rcc" in text_lower:
        app = "Civil structural construction and RCC framework"
    elif "electri" in text_lower or "wiring" in text_lower or "distribution" in text_lower:
        app = "Electrical installation, transmission and distribution"
    elif "solar" in text_lower:
        app = "Solar power generation and PV installation"
    elif "water" in text_lower or "plumbing" in text_lower or "drainage" in text_lower:
        app = "Water supply, plumbing and municipal drainage"
    else:
        app = "General public procurement and industrial application"

    return {
        "product": product,
        "material": ", ".join(materials) if materials else None,
        "specifications": ", ".join(specs) if specs else None,
        "performance_requirements": "; ".join(perf) if perf else None,
        "safety_requirements": "; ".join(safety) if safety else None,
        "application": app,
        "category_hint": category_hint,
    }


def generate_mock_reasoning(requirement_summary: str, candidates: List[dict]) -> List[dict]:
    """Generate accurate, context-aware reasoning for each candidate standard."""
    reasoning_list = []
    for cand in candidates:
        key = cand.get("key", "")
        code = cand.get("display_code", key)
        title = cand.get("title", "")
        status = cand.get("status", "ACTIVE")
        superseded = cand.get("superseded_by", [])
        cert = cand.get("certification") or {}
        mandatory = cert.get("mandatory", False) or cand.get("certification_mandatory", False)

        reasons = []
        if status == "WITHDRAWN" or status == "SUPERSEDED":
            sup_str = ", ".join(superseded) if superseded else "an updated IS edition"
            reasons.append(
                f"WARNING: {code} is {status}. Procurement specifications must reference {sup_str} instead."
            )
        else:
            reasons.append(
                f"{code} is the primary Indian Standard covering '{title}'. It matches the tender requirement parameters."
            )

        if mandatory:
            reasons.append(
                "Mandatory QCO compliance applies — supplier must possess a valid BIS license / ISI mark."
            )

        flags = cand.get("flags", [])
        if "needs_review" in flags or "unverified" in flags:
            reasons.append("Note: Standard data requires secondary manual verification against BIS gazette.")

        reasoning_list.append({
            "key": key,
            "reasoning": " ".join(reasons),
            "confidence_adjustment": 0.05 if status == "ACTIVE" and not flags else -0.1,
        })
    return reasoning_list


class MockChatModel(BaseChatModel):
    """
    Drop-in LangChain BaseChatModel that produces structured responses
    without requiring active internet or third-party LLM API keys.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_text = " ".join(str(m.content) for m in messages)

        # Check if this is an Extraction prompt (Node 02)
        if "structured procurement requirements" in full_text or "INPUT TEXT:" in full_text:
            text_match = re.search(r"INPUT TEXT:\s*(.+?)(?:Return ONLY|$)", full_text, flags=re.DOTALL)
            input_text = text_match.group(1).strip() if text_match else full_text
            extracted = extract_structured_heuristically(input_text)
            response_str = json.dumps(extracted, ensure_ascii=False, indent=2)

        # Check if this is a Recommendation prompt (Node 05)
        elif "Procurement requirement:" in full_text or "Verified candidate standards" in full_text:
            # Try to parse candidates_json from the prompt
            candidates = []
            json_match = re.search(r"Verified candidate standards.*?\n(\[.*?\])", full_text, flags=re.DOTALL)
            if json_match:
                try:
                    candidates = json.loads(json_match.group(1))
                except Exception:
                    pass
            reasoning = generate_mock_reasoning(full_text[:300], candidates)
            response_str = json.dumps(reasoning, ensure_ascii=False, indent=2)

        else:
            # Generic JSON response
            response_str = json.dumps({"status": "ok", "message": "Mock LLM response"})

        message = AIMessage(content=response_str)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock-bis-llm"
