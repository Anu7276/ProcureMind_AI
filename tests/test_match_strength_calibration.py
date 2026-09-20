"""
Unit tests for Phase R4: Match Strength Calibration, Boilerplate Handling, and Withdrawn Confidence.
"""
import pytest
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline

@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


@pytest.mark.asyncio
async def test_scenario_a_match_strength_above_low_match_threshold():
    """Scenario A must return IS 3854 at rank 1 with match_strength >= 0.50 and low_match=False."""
    query = "We need switches for fixed domestic electrical wiring in a residential government housing project."
    res = await run_pipeline(raw_input=query, input_type="text")
    assert not res.get("abstained")
    recs = res.get("recommendations", [])
    assert len(recs) >= 1
    top = recs[0]
    assert top["key"] == "IS 3854"
    assert top["match_strength"] >= 0.50
    assert top["low_match"] is False


@pytest.mark.asyncio
async def test_scenario_b_successor_promotion_and_withdrawn_confidence_cap():
    """Scenario B: promoted successor IS/IEC 60898 (Part 1) has match_strength >= withdrawn IS 8828,
    and withdrawn item's confidence is capped below the successor's confidence."""
    query = "Miniature Circuit Breakers (MCB) 10kA breaking capacity conforming to IS 8828"
    res = await run_pipeline(raw_input=query, input_type="text")
    assert not res.get("abstained")
    recs = res.get("recommendations", [])

    succ = next((r for r in recs if "60898" in r["key"]), None)
    withdrawn = next((r for r in recs if r["key"] == "IS 8828"), None)

    assert succ is not None, "Promoted successor IS/IEC 60898 (Part 1) must be in recommendations"
    assert withdrawn is not None, "Cited withdrawn IS 8828 must be in recommendations"
    assert succ["match_strength"] >= withdrawn["match_strength"]
    assert withdrawn["confidence"] <= round(succ["confidence"] - 0.05 + 1e-4, 3)


@pytest.mark.asyncio
async def test_hard_negative_out_of_scope_abstains():
    """Genuinely absent BIS products must cleanly abstain rather than hallucinating unrelated standards."""
    query = "Aviation turbine fuel Jet A-1 kerosene grade for civil aircraft refueling at international airport"
    res = await run_pipeline(raw_input=query, input_type="text")
    assert res.get("abstained") is True
    assert len(res.get("recommendations", [])) == 0
    assert len(res.get("closest_matches", [])) > 0
