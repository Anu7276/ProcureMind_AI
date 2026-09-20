"""
Unit tests for Phase 6: Reranker and Category Filter.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.reranker import rerank, _min_max_scale


@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


def test_min_max_scale():
    scores = [10.0, 20.0, 30.0]
    scaled = _min_max_scale(scores)
    assert scaled == [0.0, 0.5, 1.0]

    # All identical
    same = [5.0, 5.0, 5.0]
    scaled_same = _min_max_scale(same)
    assert scaled_same == [1.0, 1.0, 1.0]


def test_lexical_rerank_fallback_and_warning():
    candidates = [
        {"key": "IS 1239 (Part 1)", "score": 0.8, "category": "Structural Steel - Pipes"},
        {"key": "IS 4984", "score": 0.9, "category": "Water & Environment - Pipes"},
    ]
    # In mock mode, CrossEncoder is not loaded, uses lexical BM25 fallback
    reranked, warnings = rerank(
        query="mild steel pipes and tubes",
        candidates=candidates,
        top_n=5,
    )
    assert len(reranked) == 2
    assert any("lexical rerank" in w.lower() for w in warnings)
    for c in reranked:
        assert "relevance_score" in c
        assert 0.0 <= c["relevance_score"] <= 1.0


def test_category_soft_boost():
    candidates = [
        {"key": "IS 100", "score": 0.80, "category": "Electrical - Cables"},
        {"key": "IS 200", "score": 0.80, "category": "Civil & Construction"},
    ]
    reranked, _ = rerank(
        query="cables",
        candidates=candidates,
        top_n=5,
        category_hint="Electrical",
    )
    # IS 100 should get category soft boost (+0.05)
    is_100 = next(c for c in reranked if c["key"] == "IS 100")
    is_200 = next(c for c in reranked if c["key"] == "IS 200")
    assert is_100.get("category_boosted") is True
    assert is_100["score"] > is_200["score"]


def test_category_soft_boost_raises_matching_candidates():
    # Phase 4.2: Category hint gives a soft BOOST only — hard filter was removed.
    # Non-matching candidates are NOT dropped; matching ones get a +0.05 score boost.
    candidates = [
        {"key": f"IS {i}", "score": 0.80, "category": "Electrical - Motors"}
        for i in range(1, 6)
    ]
    candidates.append({"key": "IS 999", "score": 0.75, "category": "Food & Agriculture"})

    reranked, _ = rerank(
        query="motor",
        candidates=candidates,
        top_n=10,
        category_hint="Electrical",
    )
    # IS 999 should NOT be dropped (hard filter removed in Phase 4.2)
    assert any(c["key"] == "IS 999" for c in reranked), "Non-matching candidate must still appear (soft boost only)"
    # Electrical - Motors candidates should have higher scores than IS 999 (boosted by category)
    is_999 = next(c for c in reranked if c["key"] == "IS 999")
    is_1 = next(c for c in reranked if c["key"] == "IS 1")
    assert is_1["score"] > is_999["score"], "Category-matching IS 1 should outscore non-matching IS 999"


def test_category_filter_preserves_literal_mention():
    # Literal mentions must NEVER be dropped even if category does not match
    candidates = [
        {"key": f"IS {i}", "score": 0.80, "category": "Electrical"}
        for i in range(1, 6)
    ]
    candidates.append({
        "key": "IS 1786",
        "score": 0.90,
        "category": "Civil & Construction",
        "is_literal_mention": True,
    })

    reranked, _ = rerank(
        query="motor with IS 1786",
        candidates=candidates,
        top_n=10,
        category_hint="Electrical",
        literal_keys={"IS 1786"},
    )
    # IS 1786 must be preserved because it is a literal mention!
    assert any(c["key"] == "IS 1786" for c in reranked)
