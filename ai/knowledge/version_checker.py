"""
Version and amendment validity verification module.

Provides check_version() to:
- Compare cited tender publication years against catalog edition years
- Resolve predecessors and successors bidirectionally
- Detect WITHDRAWN / SUPERSEDED standards and edition mismatches
- Look up amendment data from amendments.json (or explicitly flag 'not_available_in_dataset')
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def parse_year_from_display_code(display_code: str) -> Optional[int]:
    """Extract 4-digit publication year from display codes like 'IS 1786:2008'."""
    if not display_code:
        return None
    match = re.search(r":\s*(\d{4})", str(display_code))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def check_version(
    candidate_record: Dict[str, Any],
    cited_year: Optional[int],
    all_records: Dict[str, Dict[str, Any]],
    successors_map: Optional[Dict[str, List[str]]] = None,
    predecessors_map: Optional[Dict[str, List[str]]] = None,
    amendments_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Verify version validity for a candidate BIS standard.

    Returns version_info dict:
      - cited_year: year cited in user's query/tender (or None)
      - current_edition_year: edition year known in catalog (or None)
      - is_current: True / False / None
      - status: ACTIVE / WITHDRAWN / SUPERSEDED / UNKNOWN
      - successors: list of successor standard keys
      - predecessors: list of predecessor standard keys
      - amendment_status: amendment details list or "not_available_in_dataset"
      - messages: list of human-readable explanation messages
    """
    key = candidate_record.get("key", "")
    display_code = candidate_record.get("display_code", key)
    status = candidate_record.get("status", "ACTIVE")

    # 1. Determine current edition year
    current_edition_year: Optional[int] = candidate_record.get("edition_year")
    if current_edition_year is not None:
        try:
            current_edition_year = int(current_edition_year)
        except (ValueError, TypeError):
            current_edition_year = None

    if current_edition_year is None:
        current_edition_year = parse_year_from_display_code(display_code)

    messages: List[str] = []
    is_current: Optional[bool] = None

    # 2. Successors and predecessors from bidirectional index
    successors: List[str] = []
    if successors_map and key in successors_map:
        successors = list(successors_map[key])
    else:
        raw_succ = candidate_record.get("superseded_by") or []
        if isinstance(raw_succ, str):
            raw_succ = [raw_succ]
        successors = list(raw_succ)

    predecessors: List[str] = []
    if predecessors_map and key in predecessors_map:
        predecessors = list(predecessors_map[key])
    else:
        raw_pred = candidate_record.get("supersedes") or []
        if isinstance(raw_pred, str):
            raw_pred = [raw_pred]
        predecessors = list(raw_pred)

    # 3. Check status
    if status in ("WITHDRAWN", "SUPERSEDED"):
        is_current = False
        if successors:
            messages.append(
                f"Standard is {status}; replaced by {', '.join(successors)}."
            )
        else:
            messages.append(f"Standard is {status} in dataset.")

    # 4. Check cited edition year against current edition year
    if current_edition_year is None:
        is_current = None
        messages.append("Edition year not available in dataset")
    elif cited_year is not None:
        if cited_year != current_edition_year:
            is_current = False
            messages.append(
                f"Tender cites {key}:{cited_year}; dataset lists current edition {current_edition_year}."
            )
        else:
            if status == "ACTIVE":
                is_current = True
                messages.append(
                    f"Tender cited year {cited_year} matches current active edition."
                )
    else:
        if is_current is None and status == "ACTIVE":
            is_current = True

    # 5. Amendment status (do not invent — only report if in amendments_map)
    amendment_status: Any = "not_available_in_dataset"
    if amendments_map and key in amendments_map and len(amendments_map[key]) > 0:
        amendment_status = amendments_map[key]

    return {
        "cited_year": cited_year,
        "current_edition_year": current_edition_year,
        "is_current": is_current,
        "status": status,
        "successors": successors,
        "predecessors": predecessors,
        "amendment_status": amendment_status,
        "messages": messages,
    }
