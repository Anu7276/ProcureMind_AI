"""
Unit tests for BIS certification schemes and deterministic spec_line generation (Phase R6).
"""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_04_verify import _get_compliance
from ai.pipeline.nodes.node_05_recommend import _generate_spec_line


@pytest.fixture(autouse=True)
def ensure_knowledge_loaded():
    kl.load_all()


@pytest.mark.asyncio
async def test_meity_electronics_crs_scheme():
    # IS 13252 is an IT equipment safety standard under MeitY / CRS
    comp = await _get_compliance("IS 13252", postgres_ok=False)
    assert comp is not None
    assert "CRS" in (comp.get("scheme_name") or "") or "Scheme-II" in (comp.get("scheme_name") or "")
    assert comp["mandatory"] is True
    # Fields not in qco_orders.json / standards_clean.json must be None
    assert comp["marking_requirements"] is None
    assert comp["testing_frequency"] is None
    assert comp["lead_time_weeks"] is None


@pytest.mark.asyncio
async def test_steel_cement_isi_scheme():
    # IS 1786 (TMT bars) & IS 8112 (Cement) fall under Scheme-I / ISI Mark via QCO
    comp_steel = await _get_compliance("IS 1786", postgres_ok=False)
    assert comp_steel is not None
    assert "ISI" in (comp_steel.get("scheme_name") or "") or "Scheme-I" in (comp_steel.get("scheme_name") or "")
    assert comp_steel["mandatory"] is True

    comp_cement = await _get_compliance("IS 8112", postgres_ok=False)
    assert comp_cement is not None
    assert comp_cement["mandatory"] is True


@pytest.mark.asyncio
async def test_different_certification_blocks_and_no_scheme_null_fields():
    # IS 2365 and IS 1786 must return different certification blocks (not the same templated one)
    comp_2365 = await _get_compliance("IS 2365", postgres_ok=False)
    comp_1786 = await _get_compliance("IS 1786", postgres_ok=False)

    assert comp_2365 != comp_1786
    # IS 2365 has schemes = [] in standards_clean.json
    assert comp_2365["schemes"] == []
    assert comp_2365["marking_requirements"] is None
    assert comp_2365["testing_frequency"] is None
    assert comp_2365["lead_time_weeks"] is None

    # IS 1786 has QCO with schemes = ["Scheme-I / ISI Mark"]
    assert comp_1786["mandatory"] is True
    assert len(comp_1786["schemes"]) > 0

    # IS 158 has no QCO and no compliance flags (schemes = [], mandatory = None)
    comp_158 = await _get_compliance("IS 158", postgres_ok=False)
    assert comp_158["mandatory"] is None
    assert comp_158["schemes"] == []
    assert comp_158["marking_requirements"] is None
    assert comp_158["testing_frequency"] is None
    assert comp_158["lead_time_weeks"] is None


@pytest.mark.asyncio
async def test_no_placeholder_strings_in_api_response():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/recommend",
            json={"raw_query": "TMT steel bars 12mm Fe 500D and electronics laptop and wire ropes"},
            headers={"X-API-Key": "dev-insecure-key"},
        )
        assert resp.status_code == 200
        raw_text = resp.text
        assert "XXXXXXXX" not in raw_text
        assert "R-XXXX" not in raw_text
        assert "CM/L-XXXX" not in raw_text


def test_deterministic_spec_line_mandatory():
    cand = {
        "key": "IS 1786",
        "display_code": "IS 1786:2008",
        "title": "High Strength Deformed Steel Bars and Wires for Concrete Reinforcement",
        "certification": {
            "mandatory": True,
            "scheme_name": "Scheme-I / ISI Mark",
            "scheme_code": "Scheme-I",
            "gazette_reference": "S.O. 1234(E)",
        },
    }
    spec = _generate_spec_line(cand, req_summary="Fe 500D TMT Rebars")
    assert "shall conform to IS 1786:2008" in spec
    assert "valid BIS Scheme-I / ISI Mark License" in spec
    assert "valid BIS Certification Marks License (ISI Mark) prior to dispatch" in spec
    assert "mandatory for supply and non-compliant bids shall be rejected at technical stage" in spec


def test_deterministic_spec_line_voluntary():
    cand = {
        "key": "IS 456",
        "display_code": "IS 456:2000",
        "title": "Plain and Reinforced Concrete - Code of Practice",
        "certification": {
            "mandatory": False,
            "scheme_name": "Scheme-IV / Certificate of Conformity",
            "scheme_code": "Scheme-IV",
            "gazette_reference": None,
        },
    }
    spec = _generate_spec_line(cand, req_summary="Reinforced Concrete Structure")
    assert "shall conform to IS 456:2000" in spec
    assert "Certificate of Conformity" in spec
    assert "Compliance is recommended for quality assurance" in spec
