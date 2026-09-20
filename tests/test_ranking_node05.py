"""
Unit tests for Phase 3: Node 05 ranking, monotonic confidence, and reasoning validation.
"""
from ai.pipeline.nodes.node_05_recommend import (
    _compute_confidence,
    _merge_reasoning,
)


def test_compute_confidence_monotonicity_and_bounds():
    # Strictly monotonic in score
    assert _compute_confidence(0.9, 0.1) > _compute_confidence(0.7, 0.1)
    # Rewarded by gap
    assert _compute_confidence(0.8, 0.4) > _compute_confidence(0.8, 0.1)
    # Never clipped to 1.0; capped at 0.99
    assert _compute_confidence(1.0, 1.0) <= 0.99
    assert _compute_confidence(1.0, 0.0) < 1.0


def test_ranking_relevance_first_and_data_quality_independence():
    candidates = [
        {
            "key": "IS 200",
            "score": 0.6,
            "verification_level": "verified_multi_source",
            "flags": [],
            "is_literal_mention": False,
        },
        {
            "key": "IS 100",
            "score": 0.9,
            "verification_level": "single_source_unconfirmed",
            "flags": ["needs_review"],
            "is_literal_mention": False,
        },
    ]
    warnings = []
    result = _merge_reasoning(candidates, [], "test requirement", warnings)
    # _merge_reasoning returns (recommendations, abstained, abstain_reason, closest_matches)
    recs = result[0]
    # IS 100 has higher relevance score (0.9 vs 0.6), so it MUST rank #1 despite flags
    assert recs[0]["key"] == "IS 100"
    assert recs[1]["key"] == "IS 200"
    assert "data_quality_note" in recs[0]
    assert "single-source" in recs[0]["data_quality_note"].lower() or "review" in recs[0]["data_quality_note"].lower()


def test_sorting_tie_breaking():
    # Both score 0.8, but IS 300 is literal_mention
    candidates = [
        {"key": "IS 200", "score": 0.8, "is_literal_mention": False},
        {"key": "IS 300", "score": 0.8, "is_literal_mention": True},
        {"key": "IS 100", "score": 0.8, "is_literal_mention": False},
    ]
    warnings = []
    result = _merge_reasoning(candidates, [], "test", warnings)
    recs = result[0]
    # IS 300 first (literal mention), then IS 100 before IS 200 (alphabetical)
    assert recs[0]["key"] == "IS 300"
    assert recs[1]["key"] == "IS 100"
    assert recs[2]["key"] == "IS 200"


def test_llm_validation_drops_hallucinated_codes_and_fills_missing():
    candidates = [
        {"key": "IS 325", "score": 0.9},
        {"key": "IS 12615", "score": 0.7},
    ]
    # LLM hallucinated IS 9999 and missed IS 12615
    llm_reasoning = [
        {"key": "IS 325", "reasoning": "Valid reasoning for IS 325."},
        {"key": "IS 9999", "reasoning": "Fabricated standard."},
    ]
    warnings = []
    result = _merge_reasoning(candidates, llm_reasoning, "motor specs", warnings)
    recs = result[0]
    rec_keys = [r["key"] for r in recs]
    # IS 9999 must NOT be present
    assert "IS 9999" not in rec_keys
    # IS 12615 must be present with fallback reasoning
    assert "IS 12615" in rec_keys
    is_12615 = next(r for r in recs if r["key"] == "IS 12615")
    assert len(is_12615["reasoning"]) > 10
    # Warning must be logged
    assert any("missed" in w.lower() for w in warnings)
