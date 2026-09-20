"""
Unit tests for multi-item tender segmentation and pipeline integration (Phase R5).
"""
import pytest
from ai.pipeline.segmenter import segment_tender, LineItem
from ai.pipeline.graph import run_pipeline


def test_segmenter_numbered_list():
    raw_text = """
    Tender for Civil and Electrical Works:
    1. Supply of 500 bags of Ordinary Portland Cement 43 Grade conforming to IS 8112.
    2. Supply of 20 MT High strength deformed steel bars Fe 500D per IS 1786.
    3. Installation of Miniature Circuit Breakers 32A as per IS 8828.
    """
    items = segment_tender(raw_text)
    assert len(items) == 3
    assert items[0].item_index == 1
    assert "Ordinary Portland Cement" in items[0].cleaned_text
    assert items[0].estimated_quantity == "500 bags"
    assert "IS 8112" in items[0].cited_codes

    assert items[1].item_index == 2
    assert "Fe 500D" in items[1].cleaned_text
    assert items[1].estimated_quantity == "20 MT"
    assert "IS 1786" in items[1].cited_codes

    assert items[2].item_index == 3
    assert "Miniature Circuit Breakers" in items[2].cleaned_text
    assert "IS 8828" in items[2].cited_codes


def test_segmenter_semicolon_list():
    raw_text = (
        "Supply of 43 Grade OPC cement for foundation work 500 bags; "
        "Supply of TMT steel bars Fe 500D 20 MT; "
        "Supply of PVC insulated copper cables 1100V."
    )
    items = segment_tender(raw_text)
    assert len(items) == 3
    assert "43 Grade OPC" in items[0].cleaned_text
    assert "TMT steel bars" in items[1].cleaned_text
    assert "PVC insulated copper cables" in items[2].cleaned_text


def test_segmenter_table_format():
    raw_text = """
    | S.No | Item Description | Quantity |
    |---|---|---|
    | 1 | Ordinary Portland Cement 43 Grade | 500 bags |
    | 2 | High Strength Deformed Steel Bars Fe 500D | 20 MT |
    | 3 | UPVC pipes for potable water supply | 1000 m |
    """
    items = segment_tender(raw_text)
    assert len(items) == 3
    assert "Ordinary Portland Cement 43 Grade" in items[0].cleaned_text
    assert "500 bags" in items[0].estimated_quantity
    assert "High Strength Deformed Steel Bars" in items[1].cleaned_text
    assert "20 MT" in items[1].estimated_quantity
    assert "UPVC pipes" in items[2].cleaned_text
    assert "1000 m" in items[2].estimated_quantity


def test_segmenter_single_item_fallback():
    raw_text = "Procurement of 53 Grade Ordinary Portland Cement for dam construction conforming to IS 12269."
    items = segment_tender(raw_text)
    assert len(items) == 1
    assert items[0].item_index == 1
    assert "53 Grade Ordinary Portland Cement" in items[0].cleaned_text
    assert "IS 12269" in items[0].cited_codes


@pytest.mark.asyncio
async def test_pipeline_multi_item_execution():
    tender_text = """
    1. Supply of 500 bags of OPC 43 Grade Cement conforming to IS 8112:2013.
    2. Supply of 20 MT of Fe 500D TMT Rebars per IS 1786.
    """
    result = await run_pipeline(raw_input=tender_text, input_type="text")
    assert "per_item_results" in result
    per_item = result["per_item_results"]
    assert per_item is not None
    assert len(per_item) == 2

    # Item 1 should have IS 8112 recommendations
    item1_keys = [r["key"] for r in per_item[0]["recommendations"]]
    assert "IS 8112" in item1_keys

    # Item 2 should have IS 1786 recommendations
    item2_keys = [r["key"] for r in per_item[1]["recommendations"]]
    assert "IS 1786" in item2_keys

    # Top-level recommendations contains union
    top_keys = [r["key"] for r in result["recommendations"]]
    assert "IS 8112" in top_keys
    assert "IS 1786" in top_keys
