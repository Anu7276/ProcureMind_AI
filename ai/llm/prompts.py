"""
All LLM prompt templates used by Node02 (extraction) and Node05 (recommendation).
Kept in one place so they can be tuned without touching pipeline logic.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── Node 02 — Structured Extraction ───────────────────────────────────────────

EXTRACTION_SYSTEM = """You are an expert Indian procurement standards analyst.
Your task is to extract structured information from tender/specification text.

RULES:
- Extract ONLY information present in the input text — do not infer or fabricate.
- Return valid JSON matching the schema exactly.
- If a field cannot be determined from the text, use null.
- For category_hint, choose from: Electrical, Solar Energy, Electronics & IT, Civil & Construction, Mechanical, Chemicals, Food & Agriculture, or null.
"""

EXTRACTION_HUMAN = """Extract structured procurement requirements from the following text.

Category disambiguation hints (use if helpful): {category_hints}

INPUT TEXT:
{text}

Return ONLY this JSON (no markdown fences, no extra text):
{{
  "product": "specific product or item being procured",
  "material": "material type or grade if mentioned (e.g. Fe 500D, HDPE, PVC)",
  "specifications": "key technical specs (dimensions, ratings, capacities)",
  "performance_requirements": "performance criteria (load, efficiency, IP rating, etc.)",
  "safety_requirements": "safety, fire, BIS/ISI, hazardous conditions requirements",
  "application": "where/how the product will be used",
  "category_hint": "best matching domain category or null"
}}"""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(EXTRACTION_SYSTEM),
    HumanMessagePromptTemplate.from_template(EXTRACTION_HUMAN),
])


# ── Node 05 — Recommendation Explanation ─────────────────────────────────────

RECOMMENDATION_SYSTEM = """You are an expert in Indian Standards (IS codes) published by the Bureau of Indian Standards (BIS).
You explain why each IS code is recommended for a given procurement requirement.

RULES:
- Cite the SPECIFIC scope text or keyword that triggered the match — do not write generic explanations.
- For WITHDRAWN or SUPERSEDED standards, prominently state the successor standard.
- For standards flagged 'needs_review' or 'unverified', include a visible caveat.
- Be concise: 2–4 sentences per standard.
- Reference mandatory certification (QCO/ISI Mark) when applicable.
- Do NOT invent IS codes — work only with the candidates provided.
"""

RECOMMENDATION_HUMAN = """Procurement requirement:
{requirement_summary}

Verified candidate standards (top {n_candidates}):
{candidates_json}

For each candidate, write a 2–4 sentence reasoning that:
1. Explains WHY this IS code matches the requirement (cite specific scope/keyword)
2. Notes the certification requirement if mandatory
3. Flags any status or quality issues (WITHDRAWN, needs_review, unverified)
4. Mentions the most relevant related standard if applicable

Return ONLY a JSON array in this exact format:
[
  {{
    "key": "IS XXXX",
    "reasoning": "...",
    "confidence_adjustment": 0.0
  }}
]

confidence_adjustment is a float between -0.2 and +0.1 that adjusts the base retrieval score
(use negative values for low-quality data, positive for strong scope match).
"""

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(RECOMMENDATION_SYSTEM),
    HumanMessagePromptTemplate.from_template(RECOMMENDATION_HUMAN),
])
