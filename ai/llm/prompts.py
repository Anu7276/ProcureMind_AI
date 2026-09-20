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


RECOMMENDATION_SYSTEM = """You are an expert in Indian Standards (IS codes) published by the Bureau of Indian Standards (BIS).
You explain why each IS code is recommended for a given procurement requirement.

RULES:
- Ground your explanation strictly in the candidate's title, scope, keywords, and evidence_clause.
- Quote up to 20 words directly from the standard's scope or clause where relevant.
- Explicitly name the matched terms that link the procurement requirement to the standard.
- NEVER invent IS codes, amendments, or technical clauses — work strictly with the provided candidates.
- For WITHDRAWN or SUPERSEDED standards, prominently state the successor standard.
- For standards flagged 'needs_review' or 'unverified', include a visible caveat.
- Reference mandatory certification (QCO/ISI Mark) when applicable.
- Be concise: 2–4 factual sentences per standard.
"""

RECOMMENDATION_HUMAN = """Procurement requirement:
{requirement_summary}

Verified candidate standards (top {n_candidates}):
{candidates_json}

For each candidate, write a 2–4 sentence reasoning that:
1. Explains WHY this IS code matches the requirement by naming matched terms and quoting <= 20 words from the scope or clause
2. Notes the certification requirement if mandatory (QCO / ISI mark)
3. Flags any status issues (WITHDRAWN, SUPERSEDED) and cites the successor standard
4. Notes any data quality flags if present ('needs_review' or 'unverified')

Return ONLY a JSON array in this exact format:
[
  {{
    "key": "IS XXXX",
    "reasoning": "..."
  }}
]
"""

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(RECOMMENDATION_SYSTEM),
    HumanMessagePromptTemplate.from_template(RECOMMENDATION_HUMAN),
])
