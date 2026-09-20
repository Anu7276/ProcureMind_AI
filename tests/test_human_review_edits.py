"""
Tests for Phase R2 — Human Review flow, edits, and lightweight ingest_codes_only.
"""
import pytest
from ai.pipeline.graph import run_pipeline, run_extract_only, run_pipeline_from_requirement
from ai.knowledge import knowledge_loader as kl


@pytest.fixture(autouse=True)
def setup_knowledge():
    kl.load_all()


@pytest.mark.asyncio
async def test_edited_product_changes_search_source_of_truth():
    """
    (a) Editing product to 'Fire extinguisher dry powder 9 kg ABC' (and clearing others)
    must return IS 15683 / IS 4308 in the top 3, and NOT IS 1786 in the top 3,
    even if the original normalized_text had TMT bars.
    """
    structured_req = {
        "product": "Fire extinguisher dry powder 9 kg ABC",
        "material": "",
        "specifications": "",
        "performance_requirements": "",
        "safety_requirements": "",
        "application": "",
    }
    # Original text had TMT steel bars
    original_text = "Supply of TMT steel bars Fe 500D conforming to IS 1786"

    res = await run_pipeline_from_requirement(
        structured_requirement=structured_req,
        normalized_text=original_text,
        user_edited=True,
    )

    recs = res.get("recommendations", [])
    top_keys = [r["key"] for r in recs[:3]]

    # IS 15683 or IS 4308 must be in top 3
    assert any(k in ["IS 15683", "IS 4308"] for k in top_keys)
    # IS 1786 must NOT be in top 3
    assert "IS 1786" not in top_keys


@pytest.mark.asyncio
async def test_scenario_b_ingest_to_recommend_carries_literal_and_withdrawn_status():
    """
    (b) Scenario B via ingest -> recommend (run_extract_only -> run_pipeline_from_requirement):
    Tender citing IS 8828 has IS 8828 present with status WITHDRAWN, replaced_by populated,
    and promoted successor IS/IEC 60898 (Part 1).
    """
    tender_text = "Miniature Circuit Breakers (MCB) 10kA breaking capacity conforming to IS 8828"
    extract_state = await run_extract_only(raw_input=tender_text)
    struct_req = extract_state.get("structured_requirement", {})

    res = await run_pipeline_from_requirement(
        structured_requirement=struct_req,
        normalized_text=extract_state.get("normalized_text", tender_text),
        audit_id=extract_state.get("audit_id"),
    )

    recs = res.get("recommendations", [])
    rec_keys = [r["key"] for r in recs]

    assert "IS 8828" in rec_keys
    is_8828_rec = next(r for r in recs if r["key"] == "IS 8828")
    assert is_8828_rec.get("status") == "WITHDRAWN"
    assert is_8828_rec.get("replaced_by") or is_8828_rec.get("superseded_by")


@pytest.mark.asyncio
async def test_scenario_b_cited_year_parsed():
    """
    (c) Tender with cited year 'IS 8828:1996' produces version_info.cited_year == 1996.
    """
    tender_text = "Supply of MCB conforming to IS 8828:1996 for distribution board"
    extract_state = await run_extract_only(raw_input=tender_text)
    struct_req = extract_state.get("structured_requirement", {})

    res = await run_pipeline_from_requirement(
        structured_requirement=struct_req,
        normalized_text=extract_state.get("normalized_text", tender_text),
        audit_id=extract_state.get("audit_id"),
    )

    recs = res.get("recommendations", [])
    is_8828_rec = next((r for r in recs if r["key"] == "IS 8828"), None)
    assert is_8828_rec is not None
    vinfo = is_8828_rec.get("version_info") or {}
    assert vinfo.get("cited_year") == 1996
