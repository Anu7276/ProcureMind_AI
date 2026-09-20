"""
Unit tests for text processing, tokenization, and BM25 search.
"""
import pytest
from ai.knowledge.text_utils import tokenize, STOPWORDS_EN, STOPWORDS_HI
from ai.knowledge import knowledge_loader as kl


def test_tokenize_basic():
    text = "Procurement of Portland Cement: Grade 53"
    tokens = tokenize(text)
    # 'procurement', 'of' are stopwords, 'grade' and 'cement' and '53' kept
    assert "portland" in tokens
    assert "cement" in tokens
    assert "53" in tokens
    assert "of" not in tokens
    assert "procurement" not in tokens


def test_tokenize_all_stopwords_returns_empty():
    text = "procurement of the for and with"
    tokens = tokenize(text)
    assert tokens == []


def test_tokenize_numeric_and_unit_preservation():
    text = "75 kW induction motor for 11 kV line with 500 mm clearance and Fe 500D bars"
    tokens = tokenize(text)
    # Unit and numeric tokens < 3 chars must be preserved
    assert "75" in tokens
    assert "kw" in tokens
    assert "11" in tokens
    assert "kv" in tokens
    assert "500" in tokens
    assert "mm" in tokens
    assert "fe" in tokens
    assert "500d" in tokens
    # Arbitrary short non-units should not be kept
    short_bogus = tokenize("ab cd ef zz 99")
    assert "99" in short_bogus
    assert "ab" not in short_bogus
    assert "cd" not in short_bogus


def test_tokenize_hindi_and_hinglish():
    # Devanagari
    text_hi = "पानी की सप्लाई के लिए स्टील पाइप"
    tokens_hi = tokenize(text_hi)
    assert "पानी" in tokens_hi
    assert "स्टील" in tokens_hi
    assert "पाइप" in tokens_hi
    assert "की" not in tokens_hi
    assert "लिए" not in tokens_hi

    # Hinglish
    text_hinglish = "pani ke liye steel pipe chahiye"
    tokens_hinglish = tokenize(text_hinglish)
    assert "pani" in tokens_hinglish
    assert "steel" in tokens_hinglish
    assert "pipe" in tokens_hinglish
    assert "ke" not in tokens_hinglish
    assert "liye" not in tokens_hinglish
    assert "chahiye" not in tokens_hinglish


def test_search_standards_in_memory_empty_query():
    kl.load_all()
    results = kl.search_standards_in_memory("procurement of the for and with")
    assert results == []


def test_search_standards_in_memory_bm25_normalized():
    kl.load_all()
    results = kl.search_standards_in_memory("Three phase induction motors", top_k=5)
    assert len(results) > 0
    # Top score must be 1.0 (normalized)
    assert results[0]["score"] == 1.0
    # All scores between 0 and 1
    for r in results:
        assert 0.0 <= r["score"] <= 1.0
