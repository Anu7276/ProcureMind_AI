"""
Unit tests for Phase 8: Negative Keyword Filtering & False-Positive Suppression.
"""
from __future__ import annotations

import pytest

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.reranker import _apply_negative_keyword_rules, rerank


@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


def test_drinking_water_penalizes_sewerage_pipes():
    candidates = [
        {"key": "IS 4984", "relevance_score": 0.85, "score": 0.85, "category": "Water & Environment - Pipes"},
        {"key": "IS 14333", "relevance_score": 0.85, "score": 0.85, "category": "Water & Environment - Pipes"},
    ]
    query = "hdpe pipes for drinking water supply"
    penalized, warnings = _apply_negative_keyword_rules(query, candidates, literal_keys=set())

    is_4984 = next(c for c in penalized if c["key"] == "IS 4984")
    is_14333 = next(c for c in penalized if c["key"] == "IS 14333")

    assert is_4984["score"] == 0.85
    assert is_14333["score"] < 0.85
    assert is_14333.get("negative_penalty_applied") == 0.25
    assert any("IS 14333" in w for w in warnings)


def test_metallic_steel_penalizes_plastic_pipes():
    candidates = [
        {"key": "IS 1239 (Part 1)", "relevance_score": 0.80, "score": 0.80, "category": "Structural Steel - Pipes"},
        {"key": "IS 4984", "relevance_score": 0.82, "score": 0.82, "category": "Water & Environment - Pipes"},
    ]
    query = "galvanized iron gi pipes for plumbing"
    penalized, warnings = _apply_negative_keyword_rules(query, candidates, literal_keys=set())

    is_1239 = next(c for c in penalized if c["key"] == "IS 1239 (Part 1)")
    is_4984 = next(c for c in penalized if c["key"] == "IS 4984")

    assert is_1239["score"] == 0.80
    assert is_4984["score"] < 0.80
    assert is_4984.get("negative_penalty_applied") == 0.20


def test_pozzolana_penalizes_opc_cement():
    candidates = [
        {"key": "IS 1489 (Part 1)", "relevance_score": 0.85, "score": 0.85, "category": "Construction Materials - Cement"},
        {"key": "IS 8112", "relevance_score": 0.85, "score": 0.85, "category": "Construction Materials - Cement"},
    ]
    query = "portland pozzolana cement fly ash based for plastering"
    penalized, warnings = _apply_negative_keyword_rules(query, candidates, literal_keys=set())

    is_ppc = next(c for c in penalized if c["key"] == "IS 1489 (Part 1)")
    is_opc = next(c for c in penalized if c["key"] == "IS 8112")

    assert is_ppc["score"] == 0.85
    assert is_opc["score"] < 0.85
    assert is_opc.get("negative_penalty_applied") == 0.20


def test_negative_rules_never_penalize_literal_mention():
    candidates = [
        {
            "key": "IS 14333",
            "relevance_score": 0.90,
            "score": 0.90,
            "is_literal_mention": True,
            "category": "Water & Environment - Pipes",
        },
    ]
    # Even if query says "drinking water", literal mention must never be penalized
    query = "drinking water supply tender referencing IS 14333"
    penalized, warnings = _apply_negative_keyword_rules(query, candidates, literal_keys={"IS 14333"})

    is_14333 = penalized[0]
    assert is_14333["score"] == 0.90
    assert "negative_penalty_applied" not in is_14333
    assert len(warnings) == 0
