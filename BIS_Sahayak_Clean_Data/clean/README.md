# BIS Standards Recommendation Engine — Clean Data Package

This is the merged `file2` dataset, cleaned and flagged for build-readiness. Every one of the 1,380 standards now carries a `data_quality` block telling you exactly how trustworthy it is and what's missing — nothing was silently dropped.

## Folder structure

```
standards_clean.json          ← MAIN FILE. 1,380 standards, all original fields + data_quality + embedding_source_text
is_code_whitelist_clean.json  ← 1,380 canonical codes — use as an anti-hallucination guard on LLM output
circuit_breaker_test_queries.json / is_iec_60898_1_2002_fulltext.txt  ← unchanged from file2

standards/
  standards_master_clean.csv    flat Excel view, with verification_level + flags columns
  standards_relationships.json  836 edges (references / testing / safety / allied / superseded_by)
  conflicts_resolved.csv        51 standards with source disagreements, and what to do about each
  normalize_is_code.py          canonical-key helper, unchanged

regulatory/
  qco_orders.json               11 Quality Control Orders, full detail incl. gazette refs + penalties
  certification_schemes.json    ISI / CRS / Scheme-IV / FMCS
  product_rules.json            100 products → mandatory/voluntary IS codes

knowledge/
  procurement_thesaurus.json    CPWD DSR + GeM taxonomy + Hindi synonyms → IS codes
  category_keyword_maps.json    keyword seed lists per domain
  technical_clauses.json        clause-level tolerances (6 standards only — hand-written, demo flavor)

evaluation/
  queries_master.json           128 queries (75 eval / 33 training / 20 unlabeled), each resolved against standards_clean.json

samples/                        3 sample tender documents (one deliberately contains obsolete standards)
docs/
  DATA_CLEANING_LOG.md          exactly what was changed, and why
```

**Not included:** the old pre-merge Qdrant/SQLite vector store. It doesn't match this dataset — see the cleaning log for why. Build your embeddings fresh from `embedding_source_text`.

## Start here

1. Read `docs/DATA_CLEANING_LOG.md`.
2. Open `standards/conflicts_resolved.csv` — 51 rows, each needs a 30-second judgment call before you demo it.
3. Load `standards_clean.json`, filter by `data_quality.verification_level`, and pick your pilot category's records from `verified_multi_source` first.
4. Generate embeddings from `embedding_source_text` (all 1,380 records) + `full_text` (560 records) — nothing pre-built survives the merge, this is the one real "build" step left before search works.
5. Run the 75 `role: "eval"` queries in `evaluation/queries_master.json` against your retrieval to get a real recall number before building UI.
