#!/usr/bin/env python3
"""
Evaluation script — benchmark recall@1, @3, @5 and MRR (exact & family level)
across the 75 evaluation queries in queries_master.json.

Usage:
    # Direct in-process run with mock LLM:
    python ai/evaluation/eval_runner.py --out ai/evaluation/baseline.json

    # Compare against baseline:
    python ai/evaluation/eval_runner.py --compare ai/evaluation/baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from tabulate import tabulate
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval")

from backend.config.settings import settings  # noqa: E402

API_BASE = f"http://{settings.API_HOST}:{settings.API_PORT}"


# ── Key Normalization & Matching Helpers ─────────────────────────────────────

def normalize_exact(k: str) -> str:
    """
    Exact standard code normalization:
    - Strips publication year (":YYYY")
    - Normalizes internal slashes and whitespace
    - Uppercases
    Preserves (Part N) and (Part N/Sec M).
    """
    if not k:
        return ""
    k = str(k)
    k = re.sub(r":\s*\d{4}.*$", "", k)
    k = re.sub(r"\s*/\s*", "/", k)
    k = re.sub(r"\s+", " ", k)
    return k.strip().upper()


def family_key(k: str) -> str:
    """
    Family-level standard code normalization:
    - Strips publication year (":YYYY")
    - Strips (Part N) and (Part N/Sec M) suffixes
    - Normalizes internal slashes and whitespace
    - Uppercases
    Example: 'IS/IEC 60898 (Part 1) : 2002' -> 'IS/IEC 60898'
             'IS 1786:2008'                 -> 'IS 1786'
             'IS 1554 (Part 2/Sec 1):1988'  -> 'IS 1554'
    """
    if not k:
        return ""
    k = str(k)
    k = re.sub(r":\s*\d{4}.*$", "", k)
    k = re.sub(r"\s*/\s*", "/", k)
    k = re.sub(r"\s*\([Pp]art[^)]*\)", "", k)
    k = re.sub(r"\s+", " ", k)
    return k.strip().upper()


def compute_hits_and_rr(
    returned_keys: List[str], expected_standards: List[str]
) -> Dict[str, Any]:
    """Compute exact and family hits@1, @3, @5 and reciprocal rank."""
    exact_exp: Set[str] = {normalize_exact(e) for e in expected_standards if e}
    family_exp: Set[str] = {family_key(e) for e in expected_standards if e}

    exact_ret: List[str] = [normalize_exact(r) for r in returned_keys if r]
    family_ret: List[str] = [family_key(r) for r in returned_keys if r]

    # Exact hits
    exact_hit1 = any(r in exact_exp for r in exact_ret[:1])
    exact_hit3 = any(r in exact_exp for r in exact_ret[:3])
    exact_hit5 = any(r in exact_exp for r in exact_ret[:5])

    exact_rr = 0.0
    for idx, r in enumerate(exact_ret[:5]):
        if r in exact_exp:
            exact_rr = 1.0 / (idx + 1)
            break

    # Family hits
    family_hit1 = any(r in family_exp for r in family_ret[:1])
    family_hit3 = any(r in family_exp for r in family_ret[:3])
    family_hit5 = any(r in family_exp for r in family_ret[:5])

    family_rr = 0.0
    for idx, r in enumerate(family_ret[:5]):
        if r in family_exp:
            family_rr = 1.0 / (idx + 1)
            break

    return {
        "exact_hit@1": exact_hit1,
        "exact_hit@3": exact_hit3,
        "exact_hit@5": exact_hit5,
        "exact_rr": exact_rr,
        "family_hit@1": family_hit1,
        "family_hit@3": family_hit3,
        "family_hit@5": family_hit5,
        "family_rr": family_rr,
    }


# ── Evaluation Query Runner ──────────────────────────────────────────────────

async def run_http_query(
    client: httpx.AsyncClient, query: str, api_base: str
) -> Tuple[List[str], bool]:
    """POST /recommend, return (top 5 candidate keys, abstained)."""
    try:
        resp = await client.post(
            f"{api_base}/recommend",
            json={"raw_query": query},
            timeout=60.0,
        )
        if resp.status_code != 200:
            logger.warning("API error %d for query: %s", resp.status_code, query[:60])
            return [], True
        data = resp.json()
        recs = data.get("recommendations", [])
        abstained = bool(data.get("abstained", False))
        return [r.get("key") or r.get("is_code", "") for r in recs[:5]], abstained
    except Exception as exc:
        logger.error("Request failed: %s | query: %s", exc, query[:60])
        return [], True


def print_comparison_table(baseline: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Print a side-by-side delta comparison table against baseline."""
    print("\n" + "=" * 65)
    print("  DELTA COMPARISON AGAINST BASELINE")
    print("=" * 65)

    metrics = [
        ("Exact Recall@1", baseline.get("exact_recall_at_1", 0.0), current.get("exact_recall_at_1", 0.0)),
        ("Exact Recall@3", baseline.get("exact_recall_at_3", 0.0), current.get("exact_recall_at_3", 0.0)),
        ("Exact Recall@5", baseline.get("exact_recall_at_5", 0.0), current.get("exact_recall_at_5", 0.0)),
        ("Exact MRR", baseline.get("exact_mrr", 0.0), current.get("exact_mrr", 0.0)),
        ("Family Recall@1", baseline.get("family_recall_at_1", 0.0), current.get("family_recall_at_1", 0.0)),
        ("Family Recall@3", baseline.get("family_recall_at_3", 0.0), current.get("family_recall_at_3", 0.0)),
        ("Family Recall@5", baseline.get("family_recall_at_5", 0.0), current.get("family_recall_at_5", 0.0)),
        ("Family MRR", baseline.get("family_mrr", 0.0), current.get("family_mrr", 0.0)),
    ]

    rows = []
    for name, base_val, cur_val in metrics:
        delta = cur_val - base_val
        delta_str = f"{delta:+.1%}" if "Recall" in name else f"{delta:+.3f}"
        base_str = f"{base_val:.1%}" if "Recall" in name else f"{base_val:.3f}"
        cur_str = f"{cur_val:.1%}" if "Recall" in name else f"{cur_val:.3f}"
        rows.append([name, base_str, cur_str, delta_str])

    print(tabulate(rows, headers=["Metric", "Baseline", "Current", "Delta"], tablefmt="simple"))

    # Category Delta
    base_cats = baseline.get("per_category", {})
    cur_cats = current.get("per_category", {})
    all_cats = sorted(set(list(base_cats.keys()) + list(cur_cats.keys())))

    if all_cats:
        cat_rows = []
        for cat in all_cats:
            b_val = base_cats.get(cat, {}).get("family_recall@5", 0.0)
            c_val = cur_cats.get(cat, {}).get("family_recall@5", 0.0)
            c_tot = cur_cats.get(cat, {}).get("total", base_cats.get(cat, {}).get("total", 0))
            delta = c_val - b_val
            cat_rows.append([cat, c_tot, f"{b_val:.1%}", f"{c_val:.1%}", f"{delta:+.1%}"])

        print("\n-- Per-Category Delta (Family Recall@5) --")
        print(tabulate(cat_rows, headers=["Category", "Queries", "Baseline", "Current", "Delta"], tablefmt="simple"))
    print("=" * 65)


async def evaluate(
    api_base: str = API_BASE,
    concurrency: int = 3,
    out_path: Optional[str] = None,
    compare_path: Optional[str] = None,
    blind: Optional[str] = None,
    realistic: Optional[str] = None,
) -> Dict[str, Any]:
    if realistic:
        real_path = Path("BIS_Sahayak_Clean_Data/clean/evaluation/queries_realistic.json")
        with open(real_path, encoding="utf-8") as f:
            all_queries = json.load(f)
        if realistic in ("tune", "holdout"):
            eval_queries = [q for q in all_queries if q.get("split") == realistic]
        else:
            eval_queries = all_queries
        print(f"Loaded {len(eval_queries)} realistic evaluation queries (split={realistic}) from {real_path.name}")
    elif blind:
        blind_path = Path("BIS_Sahayak_Clean_Data/clean/evaluation/queries_blind.json")
        with open(blind_path, encoding="utf-8") as f:
            all_queries = json.load(f)
        if blind in ("tune", "holdout"):
            eval_queries = [q for q in all_queries if q.get("split") == blind]
        else:
            eval_queries = all_queries
        print(f"Loaded {len(eval_queries)} cited-code blind evaluation queries (split={blind}) from {blind_path.name}")
    else:
        queries_path = settings.eval_queries_json
        with open(queries_path, encoding="utf-8") as f:
            all_queries = json.load(f)
        eval_queries = [q for q in all_queries if q.get("role") == "eval"]
        print(f"Loaded {len(eval_queries)} evaluation queries from {queries_path.name}")

    results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)

    direct_mode = False
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{api_base}/health", timeout=2.0)
            health.raise_for_status()
            logger.warning("API health: %s", health.json().get("status"))
        except Exception:
            print(f"Notice: Backend server not reached at {api_base}.")
            print("Running in-process evaluation with in-memory pipeline...")
            direct_mode = True

        if not direct_mode:
            async def bounded_run(q: Dict) -> Dict[str, Any]:
                async with semaphore:
                    top5, abstained = await run_http_query(client, q["query"], api_base)
                    metrics = compute_hits_and_rr(top5, q.get("expected_standards", []))
                    return {
                        "id": q.get("id"),
                        "query": q["query"],
                        "category": q.get("category") or "Uncategorized",
                        "language": q.get("language") or "en",
                        "expected": q.get("expected_standards", []),
                        "top5": top5,
                        "abstained": abstained,
                        **metrics,
                    }

            tasks = [bounded_run(q) for q in eval_queries]
            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating (HTTP)")
        else:
            from ai.knowledge import knowledge_loader as kl
            from ai.pipeline.graph import run_pipeline

            kl.load_all()

            async def direct_run(q: Dict) -> Dict[str, Any]:
                try:
                    res = await run_pipeline(q["query"], input_type="text")
                    recs = res.get("recommendations", [])
                    abstained = bool(res.get("abstained", False))
                    top5 = [r.get("key") or r.get("is_code", "") for r in recs[:5]]
                except Exception as e:
                    logger.error("Pipeline error on %s: %s", q.get("id"), e)
                    top5 = []
                    abstained = True

                metrics = compute_hits_and_rr(top5, q.get("expected_standards", []))
                return {
                    "id": q.get("id"),
                    "query": q["query"],
                    "category": q.get("category") or "Uncategorized",
                    "language": q.get("language") or "en",
                    "expected": q.get("expected_standards", []),
                    "top5": top5,
                    "abstained": abstained,
                    **metrics,
                }

            tasks = [direct_run(q) for q in eval_queries]
            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating (In-Process)")

    # ── Aggregate Metrics ────────────────────────────────────────────────────
    total = len(results)
    if total == 0:
        print("No evaluation queries processed.")
        return {}

    if blind or realistic:
        mode_name = "REALISTIC" if realistic else "CITED-CODE BLIND"
        split_name = realistic if realistic else blind
        out_of_scope = [r for r in results if not r["expected"]]
        in_scope = [r for r in results if r["expected"]]

        tp = sum(1 for r in out_of_scope if r.get("abstained") or not r.get("top5"))
        fn = len(out_of_scope) - tp
        fp = sum(1 for r in in_scope if r.get("abstained"))
        abstention_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        abstention_rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        abstention_f1 = (2 * abstention_prec * abstention_rec) / (abstention_prec + abstention_rec) if (abstention_prec + abstention_rec) > 0 else 0.0
        false_abstention_rate = fp / len(in_scope) if in_scope else 0.0

        in_total = len(in_scope)
        in_fam1 = sum(1 for r in in_scope if r["family_hit@1"]) / in_total if in_total > 0 else 0.0
        in_fam5 = sum(1 for r in in_scope if r["family_hit@5"]) / in_total if in_total > 0 else 0.0
        in_fmrr = sum(r["family_rr"] for r in in_scope) / in_total if in_total > 0 else 0.0

        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        print("\n" + "=" * 65)
        print(f"  BIS STANDARDS ENGINE — {mode_name} EVALUATION REPORT")
        print("=" * 65)
        print(f"  Total Queries        : {total}")
        print(f"  Split                : {split_name}")
        print(f"  Out-of-Scope Queries : {len(out_of_scope)}")
        print(f"  In-Scope Queries     : {len(in_scope)}")
        print("-" * 65)
        print("  ABSTENTION METRIC        VALUE")
        print("-" * 65)
        print(f"  Abstention Precision     {abstention_prec:.1%} ({tp}/{tp + fp})")
        print(f"  Abstention Recall        {abstention_rec:.1%} ({tp}/{tp + fn})")
        print(f"  Abstention F1 Score      {abstention_f1:.3f}")
        print(f"  False Abstention Rate    {false_abstention_rate:.1%} ({fp}/{len(in_scope)})")
        print("-" * 65)
        print("  IN-SCOPE METRICS (Family Level)")
        print("-" * 65)
        print(f"  Family Recall@1          {in_fam1:.1%} ({sum(1 for r in in_scope if r['family_hit@1'])}/{in_total})")
        print(f"  Family Recall@5          {in_fam5:.1%} ({sum(1 for r in in_scope if r['family_hit@5'])}/{in_total})")
        print(f"  Family MRR               {in_fmrr:.3f}")
        print("=" * 65)

        summary = {
            "total": total,
            "eval_type": "realistic" if realistic else "blind",
            "split": split_name,
            "out_of_scope_count": len(out_of_scope),
            "in_scope_count": len(in_scope),
            "abstention_precision": round(abstention_prec, 4),
            "abstention_recall": round(abstention_rec, 4),
            "abstention_f1": round(abstention_f1, 4),
            "false_abstention_rate": round(false_abstention_rate, 4),
            "in_scope_family_recall@1": round(in_fam1, 4),
            "in_scope_family_recall@5": round(in_fam5, 4),
            "in_scope_family_mrr": round(in_fmrr, 4),
            "details": results,
        }
        prefix = "eval_realistic" if realistic else "eval_blind"
        save_file = Path(out_path) if out_path else Path(f"ai/evaluation/{prefix}_{split_name}.json")
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary metrics written to {save_file}")
        return summary

    exact_hits1 = sum(1 for r in results if r["exact_hit@1"])
    exact_hits3 = sum(1 for r in results if r["exact_hit@3"])
    exact_hits5 = sum(1 for r in results if r["exact_hit@5"])
    exact_mrr = sum(r["exact_rr"] for r in results) / total

    exact_recall1 = exact_hits1 / total
    exact_recall3 = exact_hits3 / total
    exact_recall5 = exact_hits5 / total

    family_hits1 = sum(1 for r in results if r["family_hit@1"])
    family_hits3 = sum(1 for r in results if r["family_hit@3"])
    family_hits5 = sum(1 for r in results if r["family_hit@5"])
    family_mrr = sum(r["family_rr"] for r in results) / total

    family_recall1 = family_hits1 / total
    family_recall3 = family_hits3 / total
    family_recall5 = family_hits5 / total

    # Per-Category Metrics
    cat_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        cat_groups[r["category"]].append(r)

    cat_table = []
    per_category_summary = {}
    for cat in sorted(cat_groups.keys()):
        group = cat_groups[cat]
        cnt = len(group)
        eh5 = sum(1 for r in group if r["exact_hit@5"])
        fh1 = sum(1 for r in group if r["family_hit@1"])
        fh5 = sum(1 for r in group if r["family_hit@5"])
        cat_table.append([
            cat,
            cnt,
            f"{eh5}/{cnt} ({eh5/cnt:.1%})",
            f"{fh1}/{cnt} ({fh1/cnt:.1%})",
            f"{fh5}/{cnt} ({fh5/cnt:.1%})",
        ])
        per_category_summary[cat] = {
            "total": cnt,
            "exact_hits@5": eh5,
            "exact_recall@5": round(eh5 / cnt, 4),
            "family_hits@1": fh1,
            "family_recall@1": round(fh1 / cnt, 4),
            "family_hits@5": fh5,
            "family_recall@5": round(fh5 / cnt, 4),
        }

    # Hindi / Hinglish Subset
    hindi_group = [r for r in results if r["language"] != "en"]
    h_total = len(hindi_group)
    hindi_metrics = {}
    if h_total > 0:
        h_eh1 = sum(1 for r in hindi_group if r["exact_hit@1"])
        h_eh5 = sum(1 for r in hindi_group if r["exact_hit@5"])
        h_fh1 = sum(1 for r in hindi_group if r["family_hit@1"])
        h_fh5 = sum(1 for r in hindi_group if r["family_hit@5"])
        h_fmrr = sum(r["family_rr"] for r in hindi_group) / h_total
        hindi_metrics = {
            "total": h_total,
            "exact_recall@1": round(h_eh1 / h_total, 4),
            "exact_recall@5": round(h_eh5 / h_total, 4),
            "family_recall@1": round(h_fh1 / h_total, 4),
            "family_recall@5": round(h_fh5 / h_total, 4),
            "family_mrr": round(h_fmrr, 4),
        }

    # Failures (family-level miss in top 5)
    failures = [r for r in results if not r["family_hit@5"]]

    # ── Console Output ───────────────────────────────────────────────────────
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("  BIS STANDARDS RECOMMENDATION ENGINE — EVALUATION REPORT")
    print("=" * 65)
    print(f"  Total Queries Evaluated : {total}")
    print(f"  LLM Mode Used           : {os.getenv('LLM_PROVIDER', settings.LLM_PROVIDER)}")
    print("-" * 65)
    print("  METRIC                   EXACT MATCH         FAMILY LEVEL")
    print("-" * 65)
    print(f"  Recall@1                 {exact_recall1:.1%} ({exact_hits1}/{total})        {family_recall1:.1%} ({family_hits1}/{total})")
    print(f"  Recall@3                 {exact_recall3:.1%} ({exact_hits3}/{total})        {family_recall3:.1%} ({family_hits3}/{total})")
    print(f"  Recall@5                 {exact_recall5:.1%} ({exact_hits5}/{total})        {family_recall5:.1%} ({family_hits5}/{total})")
    print(f"  MRR                      {exact_mrr:.3f}               {family_mrr:.3f}")
    print("-" * 65)

    print("\n-- Per-Category Breakdown (Categories from dataset) --")
    print(tabulate(cat_table, headers=["Category", "Total", "Exact Rec@5", "Family Rec@1", "Family Rec@5"], tablefmt="simple"))

    if h_total > 0:
        print("\n-- Hindi / Non-English Queries Subset --")
        print(f"  Queries: {h_total}")
        print(f"  Family Recall@1 : {hindi_metrics['family_recall@1']:.1%}")
        print(f"  Family Recall@5 : {hindi_metrics['family_recall@5']:.1%}")
        print(f"  Family MRR      : {hindi_metrics['family_mrr']:.3f}")

    if failures:
        print(f"\n-- Family-Level Misses ({len(failures)} queries where expected NOT in top 5) --")
        fail_table = [
            [r["id"], r["query"][:48], ", ".join(r["expected"]), ", ".join(r["top5"][:3])]
            for r in failures[:15]
        ]
        print(tabulate(fail_table, headers=["ID", "Query", "Expected", "Top-3 Returned"], tablefmt="simple"))
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")

    print("=" * 65)

    summary = {
        "total": total,
        "exact_recall_at_1": round(exact_recall1, 4),
        "exact_recall_at_3": round(exact_recall3, 4),
        "exact_recall_at_5": round(exact_recall5, 4),
        "exact_mrr": round(exact_mrr, 4),
        "family_recall_at_1": round(family_recall1, 4),
        "family_recall_at_3": round(family_recall3, 4),
        "family_recall_at_5": round(family_recall5, 4),
        "family_mrr": round(family_mrr, 4),
        "hindi_subset": hindi_metrics,
        "per_category": per_category_summary,
        "failures": [r["id"] for r in failures],
        "details": results,
    }

    # Save output if requested or to default
    save_file = Path(out_path) if out_path else Path("ai/evaluation/eval_results.json")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary metrics written to {save_file}")

    # Compare against baseline if requested
    if compare_path and Path(compare_path).exists():
        with open(compare_path, encoding="utf-8") as bf:
            baseline_data = json.load(bf)
        print_comparison_table(baseline_data, summary)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS Standards Evaluation Runner")
    parser.add_argument("--api-url", default=API_BASE, help="Backend API base URL")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent requests")
    parser.add_argument("--out", default=None, help="Output JSON path (e.g. ai/evaluation/baseline.json)")
    parser.add_argument("--compare", default=None, help="Baseline JSON path to compare against")
    parser.add_argument("--provider", default=None, help="Force LLM_PROVIDER (e.g. mock)")
    parser.add_argument("--blind", nargs="?", const="holdout", default=None, help="Run cited-code blind evaluation ('tune', 'holdout', or 'all')")
    parser.add_argument("--realistic", nargs="?", const="holdout", default=None, help="Run realistic evaluation ('tune', 'holdout', or 'all')")
    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        settings.LLM_PROVIDER = args.provider
        try:
            from ai.llm.llm_factory import get_llm
            get_llm.cache_clear()
        except Exception:
            pass

    asyncio.run(
        evaluate(
            api_base=args.api_url,
            concurrency=args.concurrency,
            out_path=args.out,
            compare_path=args.compare,
            blind=args.blind,
            realistic=args.realistic,
        )
    )
