#!/usr/bin/env python3
"""
Evaluation script — recall@5 on 75 eval queries.

Usage:
    # With the API running:
    python ai/evaluation/eval_runner.py

    # With a custom API URL:
    python ai/evaluation/eval_runner.py --api-url http://localhost:8000

Output:
    - recall@5 and recall@1 overall
    - Per-category breakdown
    - List of failed queries (expected standard not in top 5)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tabulate import tabulate
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval")

from backend.config.settings import settings

API_BASE = f"http://{settings.API_HOST}:{settings.API_PORT}"


async def run_query(client: httpx.AsyncClient, query: str, api_base: str) -> List[str]:
    """
    POST /recommend with raw_query.
    Returns list of IS code keys from the top-5 results.
    """
    try:
        resp = await client.post(
            f"{api_base}/recommend",
            json={"raw_query": query},
            timeout=60.0,
        )
        if resp.status_code != 200:
            logger.warning("API error %d for query: %s", resp.status_code, query[:60])
            return []
        data = resp.json()
        recs = data.get("recommendations", [])
        # Return keys (bare IS code without year) from top 5
        return [r.get("key", r.get("is_code", "")).split(":")[0].strip() for r in recs[:5]]
    except Exception as exc:
        logger.error("Request failed: %s | query: %s", exc, query[:60])
        return []


def _check_hit(top5_keys: List[str], expected_standards: List[str]) -> bool:
    """
    Return True if ANY expected standard key appears in top5_keys.
    Matching is done on bare key (no year suffix).
    """
    expected_bare = {s.split(":")[0].strip() for s in expected_standards}
    returned_bare = {k.split(":")[0].strip() for k in top5_keys}
    return bool(expected_bare & returned_bare)


def _check_hit_at_1(top5_keys: List[str], expected_standards: List[str]) -> bool:
    if not top5_keys:
        return False
    top1_bare = top5_keys[0].split(":")[0].strip()
    expected_bare = {s.split(":")[0].strip() for s in expected_standards}
    return top1_bare in expected_bare


async def evaluate(api_base: str, concurrency: int = 5) -> None:
    queries_path = settings.eval_queries_json
    with open(queries_path, encoding="utf-8") as f:
        all_queries = json.load(f)

    eval_queries = [q for q in all_queries if q.get("role") == "eval"]
    logger.warning("Running %d eval queries...", len(eval_queries))

    results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_run(q: Dict) -> Dict[str, Any]:
        async with semaphore:
            top5 = await run_query(client, q["query"], api_base)
            expected = q.get("expected_standards", [])
            hit5 = _check_hit(top5, expected)
            hit1 = _check_hit_at_1(top5, expected)
            return {
                "id": q.get("id"),
                "query": q["query"],
                "category": q.get("category", "Unknown"),
                "expected": expected,
                "top5": top5,
                "hit@5": hit5,
                "hit@1": hit1,
            }

    direct_mode = False
    async with httpx.AsyncClient() as client:
        # Check API is reachable
        try:
            health = await client.get(f"{api_base}/health", timeout=3.0)
            health.raise_for_status()
            logger.warning("API health: %s", health.json().get("status"))
        except Exception:
            print(f"Notice: Backend API server not running on {api_base}.")
            print("Running direct in-process evaluation with local pipeline...")
            direct_mode = True

        if not direct_mode:
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
                    top5 = [r.get("key", r.get("is_code", "")).split(":")[0].strip() for r in recs[:5]]
                except Exception as e:
                    top5 = []
                expected = q.get("expected_standards", [])
                return {
                    "id": q.get("id"),
                    "query": q["query"],
                    "category": q.get("category", "Unknown"),
                    "expected": expected,
                    "top5": top5,
                    "hit@5": _check_hit(top5, expected),
                    "hit@1": _check_hit_at_1(top5, expected),
                }

            tasks = [direct_run(q) for q in eval_queries]
            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating (Direct)")

    # ── Compute metrics ───────────────────────────────────────────────────────
    total = len(results)
    hits5 = sum(1 for r in results if r["hit@5"])
    hits1 = sum(1 for r in results if r["hit@1"])
    recall5 = hits5 / total if total else 0
    recall1 = hits1 / total if total else 0

    # Per-category
    from collections import defaultdict
    cat_hits: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        cat_name = str(r.get("category") or "General / Uncategorized")
        cat_hits[cat_name].append(r["hit@5"])

    cat_table = []
    for cat, hits in sorted(cat_hits.items(), key=lambda x: x[0]):
        cat_table.append([cat, len(hits), sum(hits), f"{sum(hits)/len(hits):.1%}"])

    # Failures
    failures = [r for r in results if not r["hit@5"]]

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BIS Standards Recommendation Engine — Eval Report")
    print("=" * 60)
    print(f"\n  Queries evaluated : {total}")
    print(f"  recall@5          : {recall5:.1%}  ({hits5}/{total})")
    print(f"  recall@1          : {recall1:.1%}  ({hits1}/{total})")

    print("\n-- Per-category breakdown --")
    print(tabulate(cat_table, headers=["Category", "Queries", "Hits@5", "Recall@5"]))

    if failures:
        print(f"\n-- Failures ({len(failures)} queries where expected NOT in top 5) --")
        fail_table = [
            [r["id"], r["query"][:60], r["expected"], r["top5"]]
            for r in failures[:20]  # show up to 20
        ]
        print(tabulate(fail_table, headers=["ID", "Query", "Expected", "Top5 returned"]))
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    print(f"\n[OK] recall@5 = {recall5:.1%}")
    print("=" * 60)

    # Write results to JSON
    out_path = Path("ai/evaluation/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "recall_at_5": recall5,
                "recall_at_1": recall1,
                "hits_at_5": hits5,
                "total": total,
                "per_category": {
                    cat: {"total": len(h), "hits": sum(h), "recall": sum(h) / len(h)}
                    for cat, h in cat_hits.items()
                },
                "failures": [r["id"] for r in failures],
                "details": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS eval runner")
    parser.add_argument("--api-url", default=API_BASE, help="Backend API base URL")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent requests")
    args = parser.parse_args()

    asyncio.run(evaluate(api_base=args.api_url, concurrency=args.concurrency))
