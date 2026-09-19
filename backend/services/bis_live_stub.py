"""
Stub for live BIS / GeM API integration.

v1 design decision: No live API is available from BIS Connect or GeM
for the Indian Standards dataset. For v1, every recommendation cites
the static data from the zip (qco_orders.json / BIS catalogue).

TODO: Wire in live integration when BIS Connect API becomes available.
      Replace the function body; the return type contract must stay the same.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def fetch_live_bis_update(code: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the latest version / withdrawal status of an IS code from BIS Connect.

    Args:
        code: Canonical IS code key, e.g. "IS 1786"

    Returns:
        None — live integration not yet implemented (v1 stub).
        When implemented, should return a dict with keys:
          - current_edition_year: int
          - status: str ("ACTIVE" | "WITHDRAWN" | "SUPERSEDED")
          - superseded_by: list[str]
          - source_url: str
    """
    # TODO: implement HTTP call to BIS Connect / BIS portal API
    # Example future implementation:
    #   resp = httpx.get(f"https://bis.gov.in/api/standards/{code}")
    #   return resp.json() if resp.status_code == 200 else None
    logger.debug("fetch_live_bis_update called for %s — returning None (stub)", code)
    return None


def fetch_live_gem_certification(code: str) -> Optional[Dict[str, Any]]:
    """
    Check GeM portal for active certification requirements for an IS code.

    Returns:
        None — live integration not yet implemented (v1 stub).
    """
    # TODO: implement GeM API integration
    logger.debug("fetch_live_gem_certification called for %s — returning None (stub)", code)
    return None
