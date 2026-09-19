# Data Cleaning Log

Applied to `file2` (the merged 1,380-standard dataset) to produce this package. Every change is **additive/non-destructive** — no original field was deleted or overwritten, so you can always trace a cleaned record back to its source data.

## What was done

1. **Added `data_quality` block to every one of the 1,380 records**, computed from existing provenance/text/compliance fields (nothing new was invented):
   - `verification_level`: one of
     - `verified_multi_source` (175 records) — confirmed by 2+ independent source repos, no conflicts
     - `needs_review` (51 records) — has an unresolved conflict (edition year, certification status, or a source disagreement) — see `standards/conflicts_resolved.csv`
     - `unverified` (11 records) — exists only in the SIH-108 demo dataset, not confirmed anywhere else
     - `single_source_unconfirmed` (remaining ~1,143) — only one source repo mentions it; plausible but not cross-checked
   - `flags`: machine-readable list, e.g. `no_full_text`, `no_scope_or_description`, `certification_status_unknown`, `withdrawn_standard_confirm_successor_before_use`, `amendment_or_verification_data_self_asserted_not_network_verified`
   - `has_usable_search_text`: whether the record has enough text to be meaningfully embedded

2. **Added `embedding_source_text` to every record** — a precomputed string (title + scope/description + keywords + category) ready to feed straight into an embedding model. For the 531 records with no scope/description, this falls back to title + category + keywords so they're still searchable, rather than being embedding-blind.

3. **Consolidated the conflict list**: the original `conflicts_to_review.csv` had 85 rows because some standards had more than one conflict logged separately. `standards/conflicts_resolved.csv` now has one row per affected standard (51 total) with all conflicts merged and a `resolution_applied` column stating exactly what was done automatically (e.g. "kept edition_year=2013 as current; full_text is from 1992 — flagged edition_mismatch") versus what still needs a human ("no automatic rule applied — manual review needed").

4. **Regenerated `standards_master_clean.csv`** — flat view with the new verification columns added, so you can filter in Excel/Sheets by `verification_level` or `flags` without touching JSON.

5. **Regenerated `is_code_whitelist_clean.json`** from the cleaned `display_code` field only (no change in content, just re-derived to guarantee it matches the current dataset 1:1).

6. **Removed `extra_raw/`** (the old BIS_AI SQLite DB + Qdrant vector store) from this package. It was built on BIS_AI's original 817-record set, before the merge — using it against the 1,380-record dataset would silently miss ~40% of standards and mis-align IDs. Rebuild your vector index from `standards_clean.json`'s `embedding_source_text` field instead (and `full_text` for the 560 records that have it).

## What was intentionally NOT changed

- No `edition_year`, `title`, or `full_text` values were altered — where sources disagreed, both the original merged fields and the conflict note are preserved as-is. **You still need to make the final call** on the 51 `needs_review` records; this package gives you the shortlist and the specific disagreement, not a guess.
- Amendment/withdrawal dates sourced from SIH-108 are still present but are explicitly flagged `amendment_or_verification_data_self_asserted_not_network_verified` — treat these as leads to verify against BIS Connect/the gazette, not as confirmed facts.
- The 820 records with no full text were kept (not deleted) — they're still valid for title/metadata search, just flagged `no_full_text` so your RAG layer knows not to expect a chunk to quote from.

## Quick filters you'll want

```python
import json
data = json.load(open("standards_clean.json"))

# Standards safe to use in a demo without caveats
safe = [r for r in data if r["data_quality"]["verification_level"] == "verified_multi_source"]

# Standards that need a manual decision before you rely on them
review = [r for r in data if r["data_quality"]["verification_level"] == "needs_review"]

# Everything with enough text to embed right now
embeddable = [r for r in data if r["data_quality"]["has_usable_search_text"]]
```
