"""
POST /bid-check

Evaluates vendor technical bid submissions against tender requirements.
Verifies license formats (CM/L, CRS), QCO mandatory compliance, certificate age,
and superseded standard transitions.
"""
from __future__ import annotations

from fastapi import APIRouter
from ai.pipeline.bid_check import (
    BidCheckResponse,
    VendorSubmission,
    verify_bid,
)
from pydantic import BaseModel
from typing import List

router = APIRouter(tags=["Compliance & Bid Check"])


class BidCheckRequest(BaseModel):
    required_standards: List[str]
    vendor_submission: VendorSubmission


@router.post("/bid-check", response_model=BidCheckResponse)
async def check_bid(body: BidCheckRequest):
    """
    Verify vendor bid against required standards.
    """
    return verify_bid(
        required_standards=body.required_standards,
        vendor_submission=body.vendor_submission,
    )
