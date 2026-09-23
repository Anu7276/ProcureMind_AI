"""
Tests for Phase R1 — Audit Logging to PostgreSQL.
"""
import uuid
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from backend.main import app
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_05_recommend import node_05_recommend


@pytest.fixture(autouse=True)
def ensure_knowledge_loaded():
    kl.load_all()


@pytest.mark.asyncio
async def test_audit_log_failure_sets_warning_and_audit_saved_false():
    """When PostgreSQL fails during write_recommendation_log, audit_saved must be False and a warning added."""
    state = {
        "audit_id": "11111111-2222-3333-4444-555555555555",
        "raw_input": "TMT steel bars Fe 500D",
        "normalized_text": "TMT steel bars Fe 500D",
        "input_type": "text",
        "structured_requirement": {"product": "TMT steel bars"},
        "verified_candidates": [
            {
                "key": "IS 1786",
                "display_code": "IS 1786:2008",
                "title": "High strength deformed steel bars and wires for concrete reinforcement",
                "score": 0.95,
                "status": "ACTIVE",
                "whitelist_valid": True,
            }
        ],
        "stages_completed": ["ingest", "extract", "retrieve", "verify"],
        "pipeline_warnings": [],
    }

    with patch("backend.services.postgres_service.write_recommendation_log", new_callable=AsyncMock) as mock_write:
        mock_write.return_value = False
        res = await node_05_recommend(state)

        assert res.get("audit_saved") is False
        assert any("Audit log not saved" in w for w in res.get("pipeline_warnings", []))


@pytest.mark.asyncio
async def test_audit_log_exception_sets_warning_and_audit_saved_false():
    """When PostgreSQL raises an exception during write, audit_saved must be False."""
    state = {
        "audit_id": "11111111-2222-3333-4444-555555555555",
        "raw_input": "TMT steel bars Fe 500D",
        "normalized_text": "TMT steel bars Fe 500D",
        "input_type": "text",
        "structured_requirement": {"product": "TMT steel bars"},
        "verified_candidates": [],
        "stages_completed": [],
        "pipeline_warnings": [],
    }

    with patch("backend.services.postgres_service.write_recommendation_log", new_callable=AsyncMock) as mock_write:
        mock_write.side_effect = Exception("Connection refused to postgres")
        res = await node_05_recommend(state)

        assert res.get("audit_saved") is False
        assert any("Audit log not saved" in w for w in res.get("pipeline_warnings", []))


@pytest.mark.asyncio
async def test_audit_log_success_sets_audit_saved_true():
    """When PostgreSQL write succeeds, audit_saved must be True."""
    state = {
        "audit_id": "11111111-2222-3333-4444-555555555555",
        "raw_input": "TMT steel bars Fe 500D",
        "normalized_text": "TMT steel bars Fe 500D",
        "input_type": "text",
        "structured_requirement": {"product": "TMT steel bars"},
        "verified_candidates": [
            {
                "key": "IS 1786",
                "display_code": "IS 1786:2008",
                "title": "High strength deformed steel bars",
                "score": 0.95,
                "status": "ACTIVE",
                "whitelist_valid": True,
            }
        ],
        "stages_completed": [],
        "pipeline_warnings": [],
    }

    with patch("backend.services.postgres_service.write_recommendation_log", new_callable=AsyncMock) as mock_write:
        mock_write.return_value = True
        res = await node_05_recommend(state)

        assert res.get("audit_saved") is True


@pytest.mark.asyncio
async def test_multi_item_tender_writes_single_parent_audit_log_with_all_codes():
    """Posting a 3-item tender to /recommend must write exactly 1 audit log row for the parent audit_id."""
    three_item_tender = (
        "Item 1: 500 MT of High strength Fe 500D TMT reinforcement steel bars conforming to IS 1786.\n\n"
        "Item 2: 200 bags of 43 Grade Ordinary Portland Cement conforming to IS 8112.\n\n"
        "Item 3: 50 nos of Industrial safety helmets conforming to IS 2925."
    )

    db_log_store = {}

    async def fake_write_recommendation_log(
        audit_id: str,
        query_text: str,
        input_type: str,
        structured_requirement: dict | None,
        returned_codes: list[str],
        pipeline_warnings: list[str],
        top_confidence: float | None,
        abstained: bool = False,
        abstain_reason: str | None = None,
        top_match_strength: float | None = None,
        closest_matches_codes: list[str] | None = None,
    ) -> bool:
        db_log_store[audit_id] = {
            "audit_id": audit_id,
            "query_text": query_text,
            "returned_codes": returned_codes,
            "abstained": abstained,
            "top_match_strength": top_match_strength,
            "structured_requirement": structured_requirement,
        }
        return True

    parent_audit_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    with patch("backend.services.postgres_service.write_recommendation_log", side_effect=fake_write_recommendation_log) as mock_write:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/recommend",
                json={
                    "raw_query": three_item_tender,
                    "audit_id": parent_audit_id,
                },
                headers={"X-API-Key": "dev-insecure-key"},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["audit_id"] == parent_audit_id
            assert data["pipeline_meta"]["audit_saved"] is True

            # Assert write_recommendation_log was called exactly ONCE for the parent audit_id
            assert mock_write.call_count == 1
            assert len(db_log_store) == 1
            assert parent_audit_id in db_log_store

            # Assert all 3 recommended codes exist in returned_codes
            logged_codes = db_log_store[parent_audit_id]["returned_codes"]
            assert any("1786" in code for code in logged_codes)
            assert any("8112" in code for code in logged_codes)
            assert any("2925" in code for code in logged_codes)


@pytest.mark.asyncio
async def test_multi_item_tender_audit_saved_reflects_db_failure():
    """When PostgreSQL is unavailable during multi-item pipeline, audit_saved must be False."""
    three_item_tender = (
        "Item 1: 500 MT of High strength Fe 500D TMT reinforcement steel bars conforming to IS 1786.\n\n"
        "Item 2: 200 bags of 43 Grade Ordinary Portland Cement conforming to IS 8112.\n\n"
        "Item 3: 50 nos of Industrial safety helmets conforming to IS 2925."
    )

    with patch("backend.services.postgres_service.write_recommendation_log", new_callable=AsyncMock) as mock_write:
        mock_write.return_value = False
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/recommend",
                json={
                    "raw_query": three_item_tender,
                    "audit_id": "22222222-3333-4444-5555-666666666666",
                },
                headers={"X-API-Key": "dev-insecure-key"},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["pipeline_meta"]["audit_saved"] is False
            assert any("Audit log not saved" in w for w in data["pipeline_meta"]["warnings"])
