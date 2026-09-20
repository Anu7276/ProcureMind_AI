"""
Unit tests for Phase 7: Node 05 Evidence-Based Explanations.
"""
from __future__ import annotations

import json
import pytest

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_05_recommend import (
    _candidate_summary_for_prompt,
    _deterministic_fallback_reasoning,
    _merge_reasoning,
)


@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


def test_extract_evidence_fields_and_prompt_summary():
    cand = {
        "key": "IS 1786",
        "display_code": "IS 1786",
        "title": "High strength deformed steel bars and wires for concrete reinforcement",
        "status": "ACTIVE",
        "score": 0.95,
        "certification": {"mandatory": True, "scheme_name": "Scheme-I ISI Mark"},
    }
    candidates = [cand]
    query = "high strength deformed steel bars TMT for concrete reinforcement"

    summary_json = _candidate_summary_for_prompt(candidates, requirement_text=query)
    parsed = json.loads(summary_json)

    assert len(parsed) == 1
    item = parsed[0]
    assert item["key"] == "IS 1786"
    assert "scope" in item
    assert "keywords" in item
    assert "matched_terms" in item
    assert "evidence_clause" in item
    assert "version_info" in item
    assert "certification" in item

    # Check that matched_terms captured key overlapping terms
    matched = item["matched_terms"]
    assert any("steel" in t.lower() or "bars" in t.lower() or "concrete" in t.lower() for t in matched)
    assert item["certification"]["mandatory"] is True


def test_deterministic_fallback_reasoning_evidence_and_clause():
    cand = {
        "key": "IS 4984",
        "display_code": "IS 4984",
        "title": "High Density Polyethylene Pipes for Water Supply",
        "status": "ACTIVE",
        "score": 0.90,
        "evidence_clause": "Pipes shall be manufactured from virgin high density polyethylene material.",
        "certification": {"mandatory": True, "scheme_name": "Scheme-I"},
    }
    req_summary = "Product: HDPE pipes; Material: high density polyethylene; Application: water supply"

    reasoning = _deterministic_fallback_reasoning(cand, req_summary)
    assert "IS 4984" in reasoning
    assert "pipes" in reasoning.lower() or "polyethylene" in reasoning.lower()
    # Evidence clause is quoted / cited
    assert "virgin high density polyethylene" in reasoning or "clause specifies" in reasoning.lower()
    assert "Mandatory QCO" in reasoning


def test_deterministic_fallback_reasoning_withdrawn_warning():
    cand = {
        "key": "IS 8828",
        "display_code": "IS 8828",
        "title": "Miniature Circuit Breakers",
        "status": "WITHDRAWN",
        "superseded_by": ["IS/IEC 60898 (Part 1)"],
        "score": 0.85,
    }
    req_summary = "Product: MCB circuit breaker"

    reasoning = _deterministic_fallback_reasoning(cand, req_summary)
    assert "WARNING" in reasoning
    assert "WITHDRAWN" in reasoning
    assert "IS/IEC 60898 (Part 1)" in reasoning


def test_merge_reasoning_preserves_evidence_in_output():
    candidates = [
        {
            "key": "IS 1239 (Part 1)",
            "display_code": "IS 1239 (Part 1)",
            "title": "Steel Tubes, Tubulars and Other Wrought Steel Fittings",
            "status": "ACTIVE",
            "score": 0.88,
            "evidence_clause": "Tubes shall be designated by their nominal bore.",
            "certification": {"mandatory": True},
        }
    ]
    reasoning_items = []  # Empty, triggers fallback
    warnings = []
    recs = _merge_reasoning(
        candidates=candidates,
        reasoning_items=reasoning_items,
        req_summary="Steel tubes for structural purposes",
        warnings=warnings,
        req_text="mild steel tubes and pipes",
    )[0]  # unpack first element: (recommendations, abstained, abstain_reason, closest_matches)

    assert len(recs) == 1
    r = recs[0]
    assert r["is_code"] == "IS 1239 (Part 1)"
    assert "matched_terms" in r
    assert "evidence_clause" in r
    assert "scope" in r
    assert "reasoning" in r
    assert len(r["reasoning"]) > 10
