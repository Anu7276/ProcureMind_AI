"""
Text processing and tokenization utilities for BM25 retrieval.

Features:
- English stopwords (general + tender/procurement boilerplate)
- Hindi stopwords (both Devanagari and Romanized / Hinglish)
- tokenize(): lowercases, splits on non-alphanumerics, removes stopwords,
  preserves numeric tokens (e.g. '11', '500', '60898') and units ('kv', 'kw').
"""
from __future__ import annotations

import re
from typing import List, Set

# ── Stopwords: English (General + Tender Boilerplate) ────────────────────────
STOPWORDS_EN: Set[str] = {
    # Standard English grammatical stopwords
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves",
    # Procurement / tender boilerplate words (excluding content words like supply, type, work, general, standard)
    "supplies", "supplied", "supplier", "suppliers",
    "procure", "procurement", "procuring",
    "tender", "tenders", "tendered",
    "provision", "provisions",
    "item", "items",
    "require", "required", "requires", "requirement", "requirements",
    "shall",
    "spec", "specs", "specification", "specifications",
    "conforming", "conforms", "conform", "confirming",
    "strictly", "accordance", "relevant", "latest",
    "amendment", "amendments", "applicable", "per",
    "use", "used", "using", "etc", "also", "including", "includes",
    "purpose", "purposes", "various",
    "details", "detailed", "available", "good", "goods",
}

# ── Stopwords: Hindi (Devanagari & Transliterated / Hinglish) ────────────────
STOPWORDS_HI: Set[str] = {
    # Devanagari (kept complete)
    "का", "के", "की", "को", "में", "से", "पर", "लिए", "और", "या", "था", "थे", "थी",
    "है", "हैं", "हो", "होता", "होते", "होती", "किया", "किए", "गया", "गए", "गई",
    "ने", "एक", "यह", "वह", "जो", "तो", "भी", "तक", "साथ", "द्वारा", "इस", "उस",
    "इन", "उन", "पे", "रहे", "रहा", "रही", "सकते", "सकता", "सकती", "वाले", "वाला",
    "वाली", "चाहिए", "देना", "लेना", "सप्लाई", "सहित", "तथा", "एवं", "अथवा", "कर",
    "करें", "करना", "होगा", "होगी", "होंगे",
    # Transliterated / Romanized (excluding ka, me, to, the, ne, se, ki to prevent unit/English collisions)
    "ke", "ko", "mein", "par", "liye", "aur", "ya",
    "tha", "thi", "hai", "hain", "ho", "hota", "hote", "hoti",
    "kiya", "kiye", "gaya", "gaye", "gayi", "ek", "yeh", "voh", "woh",
    "jo", "bhi", "tak", "saath", "dwara", "us", "un", "pe",
    "rahe", "raha", "rahi", "sakte", "sakta", "sakti", "wale", "wala", "wali",
    "chahiye", "dena", "lena", "karna", "hoga", "hogi", "honge",
}

ALL_STOPWORDS: Set[str] = STOPWORDS_EN | STOPWORDS_HI

# Preserved short tokens (units, chemistry/engineering terms < 3 chars)
ALLOWED_SHORT_TOKENS: Set[str] = {
    # Electrical units & terms
    "kv", "kw", "hp", "va", "pa", "mw", "hz", "pf", "ac", "dc", "ka",
    # Physical units
    "mm", "cm", "kg", "kn", "gm", "ml",
    # Materials / Metallurgy
    "fe", "cu", "al", "gi", "ms", "ci", "di",
    # IP ratings / standards shorthand
    "ip",
}

# ── Boilerplate tokens excluded only from match strength denominator / coverage ─
PROCUREMENT_BOILERPLATE: Set[str] = {
    "government", "project", "projects", "tender", "tenders", "supply", "supplies",
    "installation", "install", "installed", "procure", "procurement", "procuring",
    "need", "needs", "needed", "require", "required", "requires", "requirement", "requirements",
    "quantity", "nos", "qty", "number", "numbers", "office", "building", "buildings",
    "residential", "commercial", "department", "clause", "conforming", "conforms",
    "conform", "confirming", "accordance", "strictly", "latest", "spec", "specs",
    "specification", "specifications", "etc", "details", "approx", "approximate",
    "rate", "rates", "cost", "estimate", "work", "works", "item", "items",
    "contractor", "site", "purpose", "purposes", "shall", "applicable",
    "including", "includes", "various", "general", "type", "types", "standard",
    "standards", "unit", "units", "complex", "centre", "center", "state", "national"
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize query or document text for BM25 retrieval.
    
    Rules:
    1. Lowercase
    2. Split on non-alphanumerics (retaining Unicode letters/numbers)
    3. Filter out English and Hindi stopwords
    4. Remove tokens shorter than 3 characters UNLESS:
       - Numeric (e.g. '11', '500', '60898')
       - Allowed short unit/material symbol (e.g. 'kv', 'kw', 'fe')
       - Devanagari non-stopword tokens
    """
    if not text:
        return []

    raw_tokens = re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())
    filtered_tokens: List[str] = []

    for t in raw_tokens:
        if t in ALL_STOPWORDS:
            continue

        # Check if token is purely ASCII alphanumeric
        is_ascii = t.isascii()

        if is_ascii:
            if len(t) < 3:
                # Keep if numeric (e.g. '11', '75') or allowed short token ('kv', 'fe')
                if t.isdigit() or t in ALLOWED_SHORT_TOKENS:
                    filtered_tokens.append(t)
            else:
                filtered_tokens.append(t)
        else:
            # Non-ASCII (e.g. Devanagari Hindi)
            # Retain non-stopword tokens
            if len(t) >= 2:
                filtered_tokens.append(t)

    return filtered_tokens
