"""
Unit tests for the Bid Check verification engine and API endpoint (Phase R6).
"""
from ai.pipeline.bid_check import verify_bid, VendorSubmission


def test_bid_check_compliant_cml_license():
    required = ["IS 1786", "IS 8112"]
    vendor = VendorSubmission(
        vendor_name="National Steel & Cement Infra Ltd",
        submitted_codes=["IS 1786:2008", "IS 8112:2013"],
        license_numbers=["CM/L-1234567890", "CM/L-9876543210"],
        test_certificate_dates=["2026-08-01", "2026-08-15"],
    )
    result = verify_bid(required, vendor)
    assert result.overall_verdict == "COMPLIANT"
    assert len(result.critical_gaps) == 0
    assert len(result.item_checks) == 2
    assert result.item_checks[0].status == "COMPLIANT"
    assert result.item_checks[0].license_valid is True


def test_bid_check_missing_mandatory_qco_standard():
    required = ["IS 1786", "IS 8112"]
    vendor = VendorSubmission(
        vendor_name="Partial Bidder Corp",
        submitted_codes=["IS 8112:2013"],
        license_numbers=["CM/L-9876543210"],
    )
    result = verify_bid(required, vendor)
    assert result.overall_verdict == "NON_COMPLIANT"
    assert len(result.critical_gaps) >= 1
    # IS 1786 was omitted and is mandatory
    missing_items = [ic for ic in result.item_checks if ic.status == "NON_COMPLIANT_MANDATORY"]
    assert len(missing_items) >= 1
    assert missing_items[0].required_code == "IS 1786"


def test_bid_check_superseded_standard_submitted():
    # IS 8828 is withdrawn and replaced by IS/IEC 60898 (Part 1)
    required = ["IS/IEC 60898 (Part 1)"]
    vendor = VendorSubmission(
        vendor_name="Legacy Electricals Ltd",
        submitted_codes=["IS 8828"],
        license_numbers=["CM/L-1122334455"],
        test_certificate_dates=["2026-07-01"],
    )
    result = verify_bid(required, vendor)
    assert result.overall_verdict == "CONDITIONAL"
    assert any("SUPERSEDED_CODE_SUBMITTED" in ic.status for ic in result.item_checks)
    assert any("IS/IEC 60898" in str(ic.flags) or "IS 8828" in str(ic.flags) for ic in result.item_checks)
