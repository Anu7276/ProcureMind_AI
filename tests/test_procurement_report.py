"""
Comprehensive end-to-end verification of the 8 core procurement report scenarios.
"""
import pytest
from ai.pipeline.graph import run_pipeline
from ai.pipeline.bid_check import verify_bid, VendorSubmission
from ai.knowledge import knowledge_loader as kl


@pytest.fixture(autouse=True)
def ensure_knowledge():
    kl.load_all()


@pytest.mark.asyncio
async def test_scenario_1_opc_43_grade_cement():
    """Scenario 1: Ordinary Portland Cement 43 Grade -> IS 8112."""
    query = "Supply and delivery of Ordinary Portland Cement 43 Grade for general RCC construction."
    res = await run_pipeline(raw_input=query, input_type="text")
    assert not res.get("abstained")
    recs = res.get("recommendations", [])
    assert len(recs) > 0
    top_key = recs[0].get("key")
    assert top_key == "IS 8112"
    assert recs[0].get("confidence", 0) >= 0.50
    assert "IS 8112" in recs[0].get("spec_line", "")


@pytest.mark.asyncio
async def test_scenario_2_fe_500d_tmt_rebars():
    """Scenario 2: High strength Fe 500D TMT rebars -> IS 1786."""
    query = "Procurement of High Strength Deformed Steel Bars Fe 500D for bridges and flyovers."
    res = await run_pipeline(raw_input=query, input_type="text")
    assert not res.get("abstained")
    recs = res.get("recommendations", [])
    assert len(recs) > 0
    top_key = recs[0].get("key")
    assert top_key == "IS 1786"
    assert recs[0].get("confidence", 0) >= 0.50


@pytest.mark.asyncio
async def test_scenario_3_structural_steel_sections():
    """Scenario 3: Structural steel hollow sections -> IS 4923 / IS 1161."""
    query = "Hollow steel sections for structural use in industrial trusses."
    res = await run_pipeline(raw_input=query, input_type="text")
    assert not res.get("abstained")
    recs = res.get("recommendations", [])
    keys = [r["key"] for r in recs[:3]]
    assert any(k in keys for k in ("IS 4923", "IS 1161", "IS 2062"))


@pytest.mark.asyncio
async def test_scenario_4_withdrawn_standard_successor_mcb():
    """Scenario 4: Obsolete IS 8828 cited -> Successor IS/IEC 60898 (Part 1) promoted."""
    query = "Miniature Circuit Breakers 32A 10kA conforming to IS 8828."
    res = await run_pipeline(raw_input=query, input_type="text")
    recs = res.get("recommendations", [])
    assert len(recs) > 0
    # Successor must be rank 1
    assert recs[0]["key"] == "IS/IEC 60898 (Part 1)"
    assert recs[0]["status"] == "ACTIVE"
    assert recs[0]["confidence"] >= 0.85
    # IS 8828 should be present but marked withdrawn
    is_8828_items = [r for r in recs if r["key"] == "IS 8828"]
    if is_8828_items:
        assert is_8828_items[0]["status"] == "WITHDRAWN"


@pytest.mark.asyncio
async def test_scenario_5_negative_keyword_disambiguation():
    """Scenario 5: Negative keyword rules prevent cross-domain collision."""
    query = "Industrial safety helmets for electrical workers head protection."
    res = await run_pipeline(raw_input=query, input_type="text")
    recs = res.get("recommendations", [])
    keys = [r["key"] for r in recs]
    assert "IS 2925" in keys
    # Should not recommend civil cement or water pipes
    assert "IS 8112" not in keys
    assert "IS 4985" not in keys


@pytest.mark.asyncio
async def test_scenario_6_multi_item_tender_segmentation():
    """Scenario 6: Multi-item tender splits into distinct per-item recommendations."""
    tender_text = """
    Tender for Infrastructure Materials:
    1. 500 bags Ordinary Portland Cement 43 Grade conforming to IS 8112.
    2. 20 MT Fe 500D TMT reinforcement rebars per IS 1786.
    """
    res = await run_pipeline(raw_input=tender_text, input_type="text")
    per_item = res.get("per_item_results")
    assert per_item is not None
    assert len(per_item) == 2
    assert "IS 8112" in [r["key"] for r in per_item[0]["recommendations"]]
    assert "IS 1786" in [r["key"] for r in per_item[1]["recommendations"]]


@pytest.mark.asyncio
async def test_scenario_7_out_of_scope_abstention():
    """Scenario 7: Out-of-scope tender (Aviation Turbine Fuel / Jet A-1) abstains."""
    query = "Aviation Turbine Fuel Jet A-1 bulk procurement for international airport fueling."
    res = await run_pipeline(raw_input=query, input_type="text")
    assert res.get("abstained") is True
    assert "abstain_reason" in res


def test_scenario_8_bid_check_verification():
    """Scenario 8: Bid check engine catches missing mandatory QCO and validates CM/L."""
    # Test valid bid
    valid_sub = VendorSubmission(
        vendor_name="Apex Construction Supplies",
        submitted_codes=["IS 1786:2008", "IS 8112:2013"],
        license_numbers=["CM/L-1234567890", "CM/L-9876543210"],
        test_certificate_dates=["2026-08-01", "2026-08-01"],
    )
    resp_valid = verify_bid(["IS 1786", "IS 8112"], valid_sub)
    assert resp_valid.overall_verdict == "COMPLIANT"

    # Test missing mandatory standard
    invalid_sub = VendorSubmission(
        vendor_name="Incomplete Bidder",
        submitted_codes=["IS 8112:2013"],
    )
    resp_invalid = verify_bid(["IS 1786", "IS 8112"], invalid_sub)
    assert resp_invalid.overall_verdict == "NON_COMPLIANT"
    assert any("IS 1786" in g for g in resp_invalid.critical_gaps)
