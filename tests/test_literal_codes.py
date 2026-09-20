"""
Unit tests for literal code extraction, year preservation, and retrieval separation.
"""
import pytest
import asyncio
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_01_ingest import extract_literal_codes, node_01_ingest
from ai.pipeline.nodes.node_03_retrieve import node_03_retrieve


@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


def test_literal_code_is_325_with_year():
    text = "three phase induction motors 75 kW as per IS 325:1996 for drainage pumps"
    results = extract_literal_codes(text)
    assert len(results) >= 1
    item = next((r for r in results if r["key"] == "IS 325"), None)
    assert item is not None
    assert item["key"] == "IS 325"
    assert item["cited_year"] == 1996
    assert item["raw"] == "IS 325:1996"


def test_literal_code_is_iec_60898_part_1_with_year():
    text = "MCB 32A as per IS/IEC 60898 (Part 1) : 2002 for distribution board"
    results = extract_literal_codes(text)
    assert len(results) >= 1
    item = next((r for r in results if "60898" in r["key"]), None)
    assert item is not None
    assert item["key"] == "IS/IEC 60898 (Part 1)"
    assert item["cited_year"] == 2002
    assert "(Part 1)" in item["raw"]


def test_literal_code_fe_500d_not_tagged_literal_mention():
    text = "Supply of TMT reinforcement bars, Fe 500D grade, for RCC column and beam work"
    results = extract_literal_codes(text)
    # Fe 500D is a material specification, not a literal "IS XXXX" code
    assert results == []


@pytest.mark.asyncio
async def test_ingest_and_retrieve_separation():
    # Fe 500D query: thesaurus will suggest IS 1786, but it must NOT be marked literal_mention
    state = {
        "raw_input": "Supply of TMT reinforcement bars, Fe 500D grade, for RCC column and beam work",
        "stages_completed": [],
        "pipeline_warnings": [],
    }
    ingest_out = await node_01_ingest(state)
    assert ingest_out["literal_codes"] == []
    assert "IS 1786" in ingest_out["thesaurus_expansions"]
    # Verify normalized_text is NOT polluted with thesaurus hint string
    assert "[Related IS codes from thesaurus:" not in ingest_out["normalized_text"]

    # In retrieve step:
    state.update(ingest_out)
    retrieve_out = await node_03_retrieve(state)
    cands = retrieve_out["candidates"]
    is_1786_cand = next((c for c in cands if c["key"] == "IS 1786"), None)
    assert is_1786_cand is not None
    assert is_1786_cand.get("source") != "literal_mention"
    assert is_1786_cand.get("is_literal_mention") is not True


def test_false_positive_lowercase_is_with_units():
    # 'is 800 mm' or 'is 1786 volts' must NEVER match IS 800 or IS 1786
    res1 = extract_literal_codes("The GI conduit is 800 mm long and 25 mm diameter")
    assert res1 == []

    res2 = extract_literal_codes("The operating voltage rating is 1786 volts AC")
    assert res2 == []

    res3 = extract_literal_codes("Weight of the motor is 325 kg approximately")
    assert res3 == []


def test_literal_code_format_variants():
    v1 = extract_literal_codes("Conforming to IS:1786-2008")
    assert len(v1) == 1 and v1[0]["key"] == "IS 1786" and v1[0]["cited_year"] == 2008

    v2 = extract_literal_codes("Cables conforming to IS-694")
    assert len(v2) == 1 and v2[0]["key"] == "IS 694"

    v3 = extract_literal_codes("Design as per IS: 456")
    assert len(v3) == 1 and v3[0]["key"] == "IS 456"

    v4 = extract_literal_codes("Steel conforming to IS 1786 (2008)")
    assert len(v4) == 1 and v4[0]["key"] == "IS 1786" and v4[0]["cited_year"] == 2008

    v5 = extract_literal_codes("IS 1786 : 2008 (Reaffirmed 2021)")
    assert len(v5) == 1 and v5[0]["key"] == "IS 1786" and v5[0]["cited_year"] == 2008

    v6 = extract_literal_codes("Steel tubes conforming to IS 1239 Pt 1")
    assert len(v6) == 1 and v6[0]["key"] == "IS 1239 (Part 1)"

