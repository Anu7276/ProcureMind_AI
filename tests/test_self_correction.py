"""
Unit tests for Phase 9 & 10: Self-Correction Loop, Fallback Expansion & Clarification Prompting.
"""
from __future__ import annotations

import pytest

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_03_retrieve import _build_fallback_query
from ai.pipeline.nodes.node_05_recommend import _generate_clarification_prompt, _merge_reasoning


@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


def test_build_fallback_query_extracts_product_and_domain():
    req = {
        "product": "transformer",
        "material": "copper",
        "application": "outdoor distribution",
    }
    fallback = _build_fallback_query(
        query_text="special nonstandard equipment",
        req=req,
        category_hint="Electrical",
    )
    assert "transformer" in fallback
    assert "copper" in fallback
    assert "outdoor distribution" in fallback
    # Should contain some electrical domain keywords
    assert any(w in fallback for w in ["motor", "induction", "transformer", "cable", "switchgear"])


def test_low_confidence_triggers_clarification_prompt():
    cand = {
        "key": "IS 100",
        "display_code": "IS 100",
        "title": "General Test Code",
        "score": 0.40,
        "category": "Electrical",
    }
    candidates = [cand]
    warnings = []
    recs = _merge_reasoning(
        candidates=candidates,
        reasoning_items=[],
        req_summary="Special equipment",
        warnings=warnings,
    )

    assert len(recs) == 1
    r = recs[0]
    assert r["confidence"] < 0.60
    assert r["is_low_confidence"] is True
    assert r["clarification_prompt"] is not None
    assert "material grade" in r["clarification_prompt"]
    assert any("Low confidence advisory" in w for w in warnings)


def test_high_confidence_has_no_clarification_prompt():
    cand = {
        "key": "IS 1786",
        "display_code": "IS 1786",
        "title": "High strength deformed steel bars",
        "score": 0.95,
        "category": "Civil & Construction",
    }
    candidates = [cand]
    warnings = []
    recs = _merge_reasoning(
        candidates=candidates,
        reasoning_items=[],
        req_summary="TMT steel bars Fe 500D for concrete",
        warnings=warnings,
    )

    assert len(recs) == 1
    r = recs[0]
    assert r["confidence"] >= 0.80
    assert r["is_low_confidence"] is False
    assert r["clarification_prompt"] is None
    assert not any("Low confidence advisory" in w for w in warnings)
