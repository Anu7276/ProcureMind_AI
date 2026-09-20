"""
Unit tests for BIS certification schemes and deterministic spec_line generation (Phase R6).
"""
import pytest
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_04_verify import _get_compliance, SCHEME_METADATA
from ai.pipeline.nodes.node_05_recommend import _generate_spec_line


@pytest.fixture(autouse=True)
def ensure_knowledge_loaded():
    kl.load_all()


@pytest.mark.asyncio
async def test_meity_electronics_crs_scheme():
    # IS 13252 is an IT equipment safety standard under MeitY / CRS
    comp = await _get_compliance("IS 13252", postgres_ok=False)
    assert comp is not None
    assert comp["scheme_code"] == "Scheme-II"
    assert "CRS" in comp["scheme_name"] or "Registration" in comp["scheme_name"]
    assert "R-XXXXXXXX" in comp["marking_requirements"]


@pytest.mark.asyncio
async def test_steel_cement_isi_scheme():
    # IS 1786 (TMT bars) & IS 8112 (Cement) fall under Scheme-I / ISI Mark
    comp_steel = await _get_compliance("IS 1786", postgres_ok=False)
    assert comp_steel is not None
    assert comp_steel["scheme_code"] == "Scheme-I"
    assert "ISI" in comp_steel["scheme_name"]
    assert "CM/L" in comp_steel["marking_requirements"]

    comp_cement = await _get_compliance("IS 8112", postgres_ok=False)
    assert comp_cement is not None
    assert comp_cement["scheme_code"] == "Scheme-I"


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
