"""
Bid check verification engine.
Evaluates vendor technical bid submissions against tender mandatory and voluntary BIS standards.
Checks license validity, CM/L and CRS registration formats, certificate freshness (<180 days),
and active vs superseded standard versions.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from ai.knowledge import knowledge_loader as kl

# License format patterns
_CML_PATTERN = re.compile(r"^CM/L-?\d{7,10}$", re.IGNORECASE)
_CRS_PATTERN = re.compile(r"^R-\d{8}$", re.IGNORECASE)


class VendorSubmission(BaseModel):
    vendor_name: str
    submitted_codes: List[str] = []
    license_numbers: Optional[List[str]] = None
    test_certificate_dates: Optional[List[str]] = None
    remarks: Optional[str] = None


class BidItemCheck(BaseModel):
    required_code: str
    submitted_code: Optional[str] = None
    status: str  # "COMPLIANT" | "SUPERSEDED_CODE_SUBMITTED" | "NON_COMPLIANT_MANDATORY" | "MISSING_VOLUNTARY"
    mandatory: bool
    scheme_code: Optional[str] = None
    license_number: Optional[str] = None
    license_valid: Optional[bool] = None
    test_certificate_date: Optional[str] = None
    test_certificate_expired: Optional[bool] = None
    flags: List[str] = []
    remedy: Optional[str] = None


class BidCheckResponse(BaseModel):
    overall_verdict: str  # "COMPLIANT" | "NON_COMPLIANT" | "CONDITIONAL"
    item_checks: List[BidItemCheck]
    critical_gaps: List[str]
    warnings: List[str]


def _normalize_code(code: str) -> str:
    """Normalize IS code for comparison (e.g. 'IS 1786:2008' -> 'IS 1786')."""
    if not code:
        return ""
    c = code.strip()
    return c.split(":")[0].strip()


def _is_date_expired(date_str: str, max_days: int = 180) -> bool:
    """Check if ISO date string is older than max_days."""
    try:
        # Support YYYY-MM-DD or full ISO
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = (now - dt).days
        return age > max_days
    except Exception:
        return False


def verify_bid(
    required_standards: List[str],
    vendor_submission: VendorSubmission | Dict[str, Any],
) -> BidCheckResponse:
    """
    Verify vendor bid against required standards.
    """
    if isinstance(vendor_submission, dict):
        sub = VendorSubmission(**vendor_submission)
    else:
        sub = vendor_submission

    if not kl.STANDARDS_BY_KEY:
        kl.load_all()

    submitted_codes_norm = {_normalize_code(c): c for c in sub.submitted_codes}
    licenses = sub.license_numbers or []
    cert_dates = sub.test_certificate_dates or []

    item_checks: List[BidItemCheck] = []
    critical_gaps: List[str] = []
    warnings: List[str] = []

    has_non_compliant = False
    has_conditional = False

    for idx, req_code in enumerate(required_standards):
        req_norm = _normalize_code(req_code)
        std_rec = kl.get_standard(req_norm) or kl.get_standard(req_code)
        qco = kl.get_qco_for_key(req_norm)

        is_mandatory = False
        scheme_code = "Scheme-I"
        if qco:
            is_mandatory = qco.get("enforcement_status") == "MANDATORY_ENFORCED"
            cert_req = qco.get("certification_required", "")
            if "CRS" in cert_req or "Scheme-II" in cert_req:
                scheme_code = "Scheme-II"
            elif "Scheme-IV" in cert_req:
                scheme_code = "Scheme-IV"
            elif "FMCS" in cert_req or "Scheme-X" in cert_req:
                scheme_code = "FMCS"
        elif std_rec:
            comp = std_rec.get("compliance", {}) or {}
            is_mandatory = bool(comp.get("mandatory_certification"))
            cat = (std_rec.get("category") or "").lower()
            if "electronics" in cat or "it" in cat:
                scheme_code = "Scheme-II"

        # Check corresponding license and certificate date
        lic_num = licenses[idx] if idx < len(licenses) else (licenses[0] if len(licenses) == 1 else None)
        cert_date = cert_dates[idx] if idx < len(cert_dates) else (cert_dates[0] if len(cert_dates) == 1 else None)

        lic_valid = None
        cert_expired = None
        flags: List[str] = []
        remedy = None

        # Check license format if license provided
        if lic_num:
            if scheme_code == "Scheme-II":
                if _CRS_PATTERN.match(lic_num.strip()):
                    lic_valid = True
                else:
                    lic_valid = False
                    flags.append("INVALID_LICENSE_FORMAT")
                    warnings.append(f"License '{lic_num}' for {req_code} does not match BIS CRS format (R-XXXXXXXX).")
            else:
                if _CML_PATTERN.match(lic_num.strip()):
                    lic_valid = True
                else:
                    lic_valid = False
                    flags.append("INVALID_LICENSE_FORMAT")
                    warnings.append(f"License '{lic_num}' for {req_code} does not match BIS CM/L format (CM/L-XXXXXXXXXX).")

        # Check certificate date
        if cert_date:
            if _is_date_expired(cert_date, max_days=180):
                cert_expired = True
                flags.append("EXPIRED_TEST_CERTIFICATE")
                warnings.append(f"Test certificate dated '{cert_date}' for {req_code} is older than 180 days.")
                has_conditional = True
            else:
                cert_expired = False

        # Match submitted code against required standard
        matched_submitted = None
        if req_norm in submitted_codes_norm:
            matched_submitted = submitted_codes_norm[req_norm]
        else:
            # Check if any submitted code matches by key or substring
            for sc_norm, sc_raw in submitted_codes_norm.items():
                if sc_norm == req_norm or sc_raw == req_code:
                    matched_submitted = sc_raw
                    break

        # Check for superseded code
        std_status = std_rec.get("status", "ACTIVE") if std_rec else "ACTIVE"
        superseded_by = (std_rec.get("superseded_by") or []) if std_rec else []

        if matched_submitted:
            # Check if the submitted code itself is withdrawn/superseded
            submitted_rec = kl.get_standard(_normalize_code(matched_submitted))
            sub_status = submitted_rec.get("status", "ACTIVE") if submitted_rec else "ACTIVE"
            sub_succ = (submitted_rec.get("superseded_by") or []) if submitted_rec else []

            if sub_status in ("WITHDRAWN", "SUPERSEDED") or std_status in ("WITHDRAWN", "SUPERSEDED"):
                succ_name = ", ".join(sub_succ or superseded_by) or "the current active revision"
                status = "SUPERSEDED_CODE_SUBMITTED"
                flags.append(f"SUPERSEDED_CODE_SUBMITTED (must update to {succ_name})")
                remedy = f"Vendor must submit undertaking to comply with active successor: {succ_name}."
                has_conditional = True
            else:
                if is_mandatory and lic_valid is False:
                    status = "NON_COMPLIANT_MANDATORY"
                    flags.append("INVALID_MANDATORY_LICENSE")
                    critical_gaps.append(f"Mandatory standard {req_code} has invalid license number '{lic_num}'.")
                    has_non_compliant = True
                else:
                    status = "COMPLIANT"
        else:
            # Not submitted
            # Check if vendor submitted predecessor/successor
            found_alt = False
            for sc_norm, sc_raw in submitted_codes_norm.items():
                sc_rec = kl.get_standard(sc_norm)
                if sc_rec and (req_norm in (sc_rec.get("supersedes") or []) or req_norm in (sc_rec.get("superseded_by") or [])):
                    matched_submitted = sc_raw
                    status = "SUPERSEDED_CODE_SUBMITTED"
                    flags.append(f"SUPERSEDED_CODE_SUBMITTED (submitted {sc_raw} for {req_code})")
                    remedy = f"Verify compatibility between {sc_raw} and required {req_code}."
                    has_conditional = True
                    found_alt = True
                    break

            if not found_alt:
                if is_mandatory:
                    status = "NON_COMPLIANT_MANDATORY"
                    flags.append("MISSING_MANDATORY_STANDARD")
                    critical_gaps.append(f"Mandatory standard {req_code} was not provided in vendor bid submission.")
                    remedy = f"Vendor must provide proof of compliance/license for mandatory standard {req_code}."
                    has_non_compliant = True
                else:
                    status = "MISSING_VOLUNTARY"
                    flags.append("MISSING_VOLUNTARY_STANDARD")
                    remedy = f"Voluntary standard {req_code} missing. Confirm if bidder meets technical equivalent."
                    warnings.append(f"Voluntary standard {req_code} was omitted from vendor submission.")

        item_checks.append(
            BidItemCheck(
                required_code=req_code,
                submitted_code=matched_submitted,
                status=status,
                mandatory=is_mandatory,
                scheme_code=scheme_code,
                license_number=lic_num,
                license_valid=lic_valid,
                test_certificate_date=cert_date,
                test_certificate_expired=cert_expired,
                flags=flags,
                remedy=remedy,
            )
        )

    # Determine overall verdict
    if has_non_compliant or critical_gaps:
        overall_verdict = "NON_COMPLIANT"
    elif has_conditional or any(ic.status == "SUPERSEDED_CODE_SUBMITTED" for ic in item_checks):
        overall_verdict = "CONDITIONAL"
    else:
        overall_verdict = "COMPLIANT"

    return BidCheckResponse(
        overall_verdict=overall_verdict,
        item_checks=item_checks,
        critical_gaps=critical_gaps,
        warnings=warnings,
    )
