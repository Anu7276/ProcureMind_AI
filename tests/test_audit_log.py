"""
Tests for Phase R1 — Audit Logging to PostgreSQL.
"""
import pytest
from unittest.mock import AsyncMock, patch
from ai.pipeline.nodes.node_05_recommend import node_05_recommend


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
