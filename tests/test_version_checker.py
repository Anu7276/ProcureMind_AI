"""
Unit tests for Phase 4: Version checking, edition mismatch detection,
successor promotion, and whitelist enforcement.
"""
import pytest
from pathlib import Path

from ai.knowledge import knowledge_loader as kl
from ai.knowledge.version_checker import check_version
from ai.pipeline.graph import run_pipeline
from ai.pipeline.nodes.node_04_verify import node_04_verify


@pytest.fixture(scope="module", autouse=True)
def setup_catalog():
    kl.load_all()


def test_version_checker_cited_year_and_successors():
    is_325 = kl.get_standard("IS 325")
    assert is_325 is not None
    info = kl.check_standard_version(is_325, cited_year=1996)
    assert info["status"] == "SUPERSEDED"
    assert info["is_current"] is False
    assert "IS 12615" in info["successors"]
    assert info["amendment_status"] == "not_available_in_dataset"


def test_version_checker_edition_mismatch():
    # IS 12615 current edition is 2018; tender citing 2011 should trigger mismatch
    is_12615 = kl.get_standard("IS 12615")
    assert is_12615 is not None
    info = kl.check_standard_version(is_12615, cited_year=2011)
    assert info["cited_year"] == 2011
    assert info["current_edition_year"] == 2018
    assert info["is_current"] is False
    assert any("cites" in m.lower() for m in info["messages"])


def test_whitelist_guard_drops_unrecognized_code():
    state = {
        "candidates": [
            {"key": "IS 12269", "display_code": "IS 12269", "score": 0.9},
            {"key": "IS 99999_FAKE", "display_code": "IS 99999_FAKE", "score": 0.95},
        ],
        "stages_completed": [],
        "pipeline_warnings": [],
    }
    import asyncio
    out = asyncio.run(node_04_verify(state))
    verified_keys = [c["key"] for c in out["verified_candidates"]]
    assert "IS 12269" in verified_keys
    assert "IS 99999_FAKE" not in verified_keys
    assert any("dropped" in w.lower() for w in out["pipeline_warnings"])


@pytest.mark.asyncio
async def test_golden_mcb_is_8828_withdrawn_promotes_successor():
    # Golden test 1: "MCB 32A as per IS 8828:1996"
    query = "MCB 32A as per IS 8828:1996 for distribution board"
    result = await run_pipeline(raw_input=query, input_type="text")
    recs = result["recommendations"]
    assert len(recs) >= 2

    # Rank 1 must be IS/IEC 60898 (Part 1)
    rank1 = recs[0]
    assert "60898" in rank1["key"]
    assert rank1["status"] == "ACTIVE"

    # IS 8828 must be present, flagged WITHDRAWN, and have replaced_by set
    is_8828 = next((r for r in recs if r["key"] == "IS 8828"), None)
    assert is_8828 is not None
    assert is_8828["status"] == "WITHDRAWN"
    assert len(is_8828["replaced_by"]) > 0
    assert any("60898" in x for x in is_8828["replaced_by"])


@pytest.mark.asyncio
async def test_golden_motor_is_325_superseded_ranks_below_is_12615():
    # Golden test 2: "Three phase induction motors 75 kW for drainage pumps as per IS 325:1996"
    query = "Three phase induction motors 75 kW for drainage pumps as per IS 325:1996"
    result = await run_pipeline(raw_input=query, input_type="text")
    recs = result["recommendations"]

    is_12615_idx = next((i for i, r in enumerate(recs) if r["key"] == "IS 12615"), None)
    is_325_idx = next((i for i, r in enumerate(recs) if r["key"] == "IS 325"), None)

    assert is_12615_idx is not None, "Successor IS 12615 must be recommended"
    assert is_325_idx is not None, "Original cited IS 325 must be returned"
    assert is_12615_idx < is_325_idx, "IS 12615 must rank higher than superseded IS 325"

    is_325_rec = recs[is_325_idx]
    assert is_325_rec["status"] == "SUPERSEDED"


@pytest.mark.asyncio
async def test_golden_sample_tender_with_obsolete_standards():
    # Golden test 3: The tender in samples/Sample_Tender_With_Obsolete_Standards.txt returns a version warning for IS 325
    sample_path = Path("BIS_Sahayak_Clean_Data/clean/samples/Sample_Tender_With_Obsolete_Standards.txt")
    assert sample_path.exists()
    content = sample_path.read_text(encoding="utf-8")

    result = await run_pipeline(raw_input=content, input_type="text")
    recs = result["recommendations"]

    is_325_rec = next((r for r in recs if r["key"] == "IS 325"), None)
    assert is_325_rec is not None
    assert is_325_rec["status"] == "SUPERSEDED"
    # Must have replaced_by / successor to IS 12615
    assert "IS 12615" in is_325_rec.get("replaced_by", []) or "IS 12615" in is_325_rec.get("superseded_by", [])
