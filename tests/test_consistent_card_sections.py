"""
Test for invariant: Every standard result card must contain the exact SAME four blocks
in the SAME order (amendments, version_check, certification, verification), whether or not
the underlying data exists. Never omit any block, and never fabricate placeholder values.
"""
import pytest
from fastapi.testclient import TestClient

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_04_verify import node_04_verify
from ai.pipeline.nodes.node_05_recommend import node_05_recommend, _merge_reasoning
from backend.main import app
from backend.schemas.api_schemas import RecommendationItem


@pytest.fixture(scope="module", autouse=True)
def setup_knowledge():
    kl.load_all()


@pytest.mark.asyncio
async def test_node_04_and_05_populate_all_four_blocks_for_with_and_without_standards():
    """
    Test two standards:
      1. With confirmed certification/QCO data: IS 1786 (TMT bars)
      2. Without certification data & without edition year: IS 158 (Ready Mixed Paint / unconfirmed catalog entry)
    """
    state = {
        "candidates": [
            {
                "key": "IS 1786",
                "display_code": "IS 1786:2008",
                "title": "High strength deformed steel bars and wires for concrete reinforcement",
                "score": 0.95,
                "is_literal_mention": True,
            },
            {
                "key": "IS 158",
                "display_code": "IS 158",
                "title": "Ready mixed paint, brushing, bituminous, black, lead-free",
                "score": 0.75,
                "is_literal_mention": False,
            },
        ],
        "stages_completed": [],
        "pipeline_warnings": [],
    }

    # Run Node 04
    v_out = await node_04_verify(state)
    verified_candidates = v_out["verified_candidates"]
    assert len(verified_candidates) >= 2

    is_1786_cand = next(c for c in verified_candidates if c["key"] == "IS 1786")
    is_158_cand = next(c for c in verified_candidates if c["key"] == "IS 158")

    # Assert all four keys are present in Node 04 output
    for cand in (is_1786_cand, is_158_cand):
        assert "certification" in cand, f"certification missing for {cand['key']}"
        assert "version_check" in cand, f"version_check missing for {cand['key']}"
        assert "amendments" in cand, f"amendments missing for {cand['key']}"
        assert "verification" in cand, f"verification missing for {cand['key']}"

    # Verify IS 1786 has confirmed certification data
    assert is_1786_cand["certification"]["mandatory"] is True
    assert len(is_1786_cand["certification"]["schemes"]) > 0
    assert is_1786_cand["version_check"]["current_edition_year"] == 2008
    assert is_1786_cand["version_check"]["is_current"] is True

    # Verify IS 158 without certification and without edition year has explicit "not confirmed" / "unknown" values
    assert is_158_cand["certification"]["mandatory"] is None, "mandatory must be None (not False) when unconfirmed"
    assert is_158_cand["certification"]["schemes"] == [], "schemes must be empty list when unconfirmed"
    assert is_158_cand["version_check"]["current_edition_year"] is None
    assert is_158_cand["version_check"]["is_current"] is None, "is_current must be None when edition year is None"
    assert any("Edition year not available in dataset" in m for m in is_158_cand["version_check"]["messages"])
    assert is_158_cand["amendments"]["status"] == "not_available_in_dataset"
    assert is_158_cand["amendments"]["entries"] == []
    assert is_158_cand["verification"]["level"] is not None
    assert isinstance(is_158_cand["verification"]["flags"], list)

    # Run Node 05
    recs, abstained, _, _ = _merge_reasoning(
        candidates=verified_candidates,
        reasoning_items=[],
        req_summary="Steel bars and paint",
        warnings=[],
        req_text="High strength steel bars and ready mixed paint",
    )

    assert len(recs) >= 2
    r_1786 = next(r for r in recs if r["key"] == "IS 1786")
    r_158 = next(r for r in recs if r["key"] == "IS 158")

    for r in (r_1786, r_158):
        # Validate Pydantic schema validation
        item_schema = RecommendationItem(**r)
        dumped = item_schema.model_dump()

        # All four keys must exist and not be None
        assert dumped.get("certification") is not None
        assert dumped.get("version_check") is not None
        assert dumped.get("amendments") is not None
        assert dumped.get("verification") is not None

    # Assert specific fields for standard with data
    assert r_1786["certification"]["mandatory"] is True
    assert len(r_1786["certification"]["schemes"]) > 0
    assert r_1786["version_check"]["current_edition_year"] == 2008

    # Assert specific fields for standard without data
    assert r_158["certification"]["mandatory"] is None
    assert r_158["certification"]["schemes"] == []
    assert r_158["version_check"]["current_edition_year"] is None
    assert r_158["version_check"]["is_current"] is None
    assert r_158["amendments"]["status"] == "not_available_in_dataset"
    assert r_158["amendments"]["entries"] == []


def test_api_recommend_returns_all_four_blocks_for_all_items():
    """
    Test FastAPI /recommend endpoint directly to assert that every recommendation
    item returned in the API response contains all four blocks and none are omitted.
    """
    client = TestClient(app)
    response = client.post(
        "/recommend",
        json={"raw_query": "Supply of Fe 500D TMT reinforcement bars conforming to IS 1786"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0

    for item in data["recommendations"]:
        assert "certification" in item, f"Missing certification in API item {item.get('key')}"
        assert "version_check" in item, f"Missing version_check in API item {item.get('key')}"
        assert "amendments" in item, f"Missing amendments in API item {item.get('key')}"
        assert "verification" in item, f"Missing verification in API item {item.get('key')}"

        assert isinstance(item["certification"], dict)
        assert isinstance(item["version_check"], dict)
        assert isinstance(item["amendments"], dict)
        assert isinstance(item["verification"], dict)

        assert "mandatory" in item["certification"]
        assert "schemes" in item["certification"]
        assert "current_edition_year" in item["version_check"]
        assert "is_current" in item["version_check"]
        assert "status" in item["amendments"]
        assert "entries" in item["amendments"]
        assert "level" in item["verification"]
        assert "flags" in item["verification"]
