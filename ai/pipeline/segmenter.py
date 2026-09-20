"""
Tender item segmenter.
Splits multi-item procurement tenders (numbered lists, bullets, tables, semicolons)
into discrete LineItem objects for independent extraction, retrieval, and verification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ai.pipeline.nodes.node_01_ingest import extract_literal_codes


@dataclass
class LineItem:
    item_index: int
    raw_text: str
    cleaned_text: str
    cited_codes: List[str] = field(default_factory=list)
    estimated_quantity: Optional[str] = None


# Regex pattern to estimate quantities from text (e.g. 500 bags, 20 MT, 1000 nos)
_QTY_PATTERN = re.compile(
    r"\b(\d+(?:,\d+)*(?:\.\d+)?\s*(?:bags|mts?|metric\s+tonnes?|metric\s+tons?|tonnes?|tons?|nos?\.?|numbers?|pieces?|pcs?|units?|meters?|metres?|mtrs?|m|km|sqm|sqft|sq\.?\s*m|kg|kgs|kilograms?|litres?|liters?|ltrs?|l|sets?|boxes?|bundles?|coils?|drums?|runs?|lots?))\b",
    re.IGNORECASE,
)

# Numbered item prefix patterns
_NUMBERED_LINE_RE = re.compile(
    r"^\s*(?:item\s+)?(?:\d+|[a-zA-Z])[\.\)\:\-\/]\s*(.*)$",
    re.IGNORECASE,
)
_PAREN_NUMBERED_RE = re.compile(
    r"^\s*[\(\[](?:\d+|[a-zA-Z])[\)\]]\s*(.*)$",
    re.IGNORECASE,
)
_BULLET_LINE_RE = re.compile(
    r"^\s*[\*\-\•\▪\▫\–\—\+]\s+(.*)$",
)


def _extract_quantity(text: str) -> Optional[str]:
    """Find the first matching quantity expression in the text."""
    m = _QTY_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None


def _clean_text(text: str) -> str:
    """Strip bullet markers, numbering prefixes, and excessive whitespace."""
    t = text.strip()
    # Strip markdown table formatting if any
    if t.startswith("|") and t.endswith("|"):
        cells = [c.strip() for c in t.strip("|").split("|")]
        # Remove empty or pure index cells
        cells = [c for c in cells if c and not re.match(r"^\d+$", c)]
        t = " - ".join(cells)

    # Strip prefixes
    t = _NUMBERED_LINE_RE.sub(r"\1", t)
    t = _PAREN_NUMBERED_RE.sub(r"\1", t)
    t = re.sub(r"^[\*\-\•\▪\▫\–\—\+]\s*", "", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _is_table_divider(line: str) -> bool:
    """Check if a line is a markdown table header separator (e.g. |---|---|)."""
    cleaned = line.strip().strip("|")
    return bool(cleaned and all(c in "-: |" for c in cleaned) and "-" in cleaned)


def _is_table_header(line: str) -> bool:
    """Check if a table line is a header row."""
    lower = line.lower()
    header_keywords = ["s.no", "sl.no", "sl no", "item no", "description", "item description", "quantity", "qty", "unit", "specification"]
    return any(kw in lower for kw in header_keywords)


def _parse_table_rows(lines: List[str]) -> List[str]:
    """Parse markdown table rows into raw item strings."""
    rows: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        if _is_table_divider(stripped):
            continue
        if _is_table_header(stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells or all(not c for c in cells):
            continue
        # Check if first cell is just an index (e.g. "1")
        if len(cells) > 1 and re.match(r"^\d+$", cells[0]):
            content_cells = cells[1:]
        else:
            content_cells = cells
        row_str = " | ".join(c for c in content_cells if c)
        if row_str and len(row_str) > 3:
            rows.append(row_str)
    return rows


def _split_inline_numbered(text: str) -> List[str]:
    """Split text containing inline numbering like '1. Item one 2. Item two' or '(1) Item one (2) Item two'."""
    pattern = re.compile(
        r"(?:^|\s+)(?:(?:\d+|[a-zA-Z])[\.\)\:\-\/]|[\(\[](?:\d+|[a-zA-Z])[\)\]]|Item\s+\d+[\:\.\-])\s+",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) >= 2:
        segments = []
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seg = text[start:end].strip()
            if seg:
                segments.append(seg)
        return segments
    return []


def segment_tender(raw_text: str) -> List[LineItem]:
    """
    Split raw tender description into discrete LineItem instances.

    Strategies evaluated in order:
    1. Markdown table rows (lines starting and ending with '|')
    2. Multi-line numbered lists ('1.', '1)', '(1)', 'Item 1:') with continuation support
    3. Multi-line bullet points ('*', '-', '•', '▪', '▫', '–', '—', '+') with continuation support
    4. Inline numbered sequences
    5. Semicolon-separated item lists (when each segment contains distinct content)
    6. Fallback: single item returning the entire text.
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    raw_segments: List[str] = []

    # 1. Check for markdown table
    table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
    if len(table_lines) >= 2:
        parsed_rows = _parse_table_rows(table_lines)
        if len(parsed_rows) >= 1:
            raw_segments = parsed_rows

    # 2. Check for multi-line numbered or bullet list
    if not raw_segments and len(lines) >= 2:
        # Check if there are numbered lines
        numbered_indices = [
            i for i, l in enumerate(lines) if _NUMBERED_LINE_RE.match(l) or _PAREN_NUMBERED_RE.match(l)
        ]
        bullet_indices = [
            i for i, l in enumerate(lines) if _BULLET_LINE_RE.match(l)
        ]

        if len(numbered_indices) >= 2 or (len(numbered_indices) >= 1 and numbered_indices[0] == 0 and len(lines) == 1):
            # Group lines by numbered item, ignoring any preamble before first numbered item
            current_seg: List[str] = []
            for i, l in enumerate(lines):
                if i < numbered_indices[0]:
                    # Preamble header line before first item
                    continue
                if _NUMBERED_LINE_RE.match(l) or _PAREN_NUMBERED_RE.match(l):
                    if current_seg:
                        raw_segments.append(" ".join(current_seg))
                        current_seg = []
                    current_seg.append(l)
                else:
                    current_seg.append(l)
            if current_seg:
                raw_segments.append(" ".join(current_seg))

        elif len(bullet_indices) >= 2 or (len(bullet_indices) >= 1 and bullet_indices[0] == 0 and len(lines) == 1):
            current_seg = []
            for i, l in enumerate(lines):
                if i < bullet_indices[0]:
                    continue
                if _BULLET_LINE_RE.match(l):
                    if current_seg:
                        raw_segments.append(" ".join(current_seg))
                        current_seg = []
                    current_seg.append(l)
                else:
                    current_seg.append(l)
            if current_seg:
                raw_segments.append(" ".join(current_seg))

    # 3. Check for inline numbered sequences
    if not raw_segments:
        inline_segs = _split_inline_numbered(text)
        if len(inline_segs) >= 2:
            raw_segments = inline_segs

    # 4. Check for semicolon-separated items
    if not raw_segments and ";" in text:
        semi_parts = [p.strip() for p in text.split(";") if p.strip()]
        if len(semi_parts) >= 2 and all(len(p) >= 8 for p in semi_parts):
            raw_segments = semi_parts

    # 5. Fallback: single item
    if not raw_segments:
        raw_segments = [text]

    # Build LineItem list
    items: List[LineItem] = []
    for idx, seg in enumerate(raw_segments, start=1):
        cleaned = _clean_text(seg)
        if not cleaned:
            continue
        qty = _extract_quantity(seg)
        literal_matches = extract_literal_codes(cleaned)
        cited = [m.get("key") or m.get("raw") for m in literal_matches if m.get("key") or m.get("raw")]
        items.append(
            LineItem(
                item_index=idx,
                raw_text=seg,
                cleaned_text=cleaned,
                cited_codes=cited,
                estimated_quantity=qty,
            )
        )

    # If all segments were somehow empty, return single fallback
    if not items:
        cleaned = _clean_text(text)
        qty = _extract_quantity(text)
        literal_matches = extract_literal_codes(cleaned)
        cited = [m.get("key") or m.get("raw") for m in literal_matches if m.get("key") or m.get("raw")]
        items = [
            LineItem(
                item_index=1,
                raw_text=text,
                cleaned_text=cleaned,
                cited_codes=cited,
                estimated_quantity=qty,
            )
        ]

    return items
