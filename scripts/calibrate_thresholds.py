#!/usr/bin/env python3
"""
Threshold Calibration Script for ProcureMind AI.

Tunes ABSTAIN_THRESHOLD, ABSTAIN_COVERAGE_FLOOR, and LOW_MATCH_THRESHOLD
using only labeled training data (training_pair, blind_tune, realistic_tune).
NEVER uses holdout or eval queries.

Usage:
    python scripts/calibrate_thresholds.py [--mode offline|full]
"""
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import numpy as np
from typing import Any, Dict, List, Tuple

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline

def load_training_queries() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load in-scope positive and out-of-scope negative queries from:
      1. queries_master.json (role == "training_pair")
      2. queries_blind.json (split == "tune")
      3. queries_realistic.json (split == "tune")
    """
    data_dir = Path("BIS_Sahayak_Clean_Data/clean/evaluation")
    in_scope = []
    out_of_scope = []

    # 1. Master training pairs
    master_p = data_dir / "queries_master.json"
    if master_p.exists():
        with open(master_p, "r", encoding="utf-8") as f:
            for q in json.load(f):
                if q.get("role") == "training_pair":
                    if q.get("expected_standards"):
                        in_scope.append(q)
                    else:
                        out_of_scope.append(q)

    # 2. Blind tune split
    blind_p = data_dir / "queries_blind.json"
    if blind_p.exists():
        with open(blind_p, "r", encoding="utf-8") as f:
            for q in json.load(f):
                if q.get("split") == "tune" or q.get("role") == "blind_tune":
                    if q.get("expected_standards"):
                        in_scope.append(q)
                    else:
                        out_of_scope.append(q)

    # 3. Realistic tune split
    real_p = data_dir / "queries_realistic.json"
    if real_p.exists():
        with open(real_p, "r", encoding="utf-8") as f:
            for q in json.load(f):
                if q.get("split") == "tune":
                    if q.get("expected_standards"):
                        in_scope.append(q)
                    else:
                        out_of_scope.append(q)

    return in_scope, out_of_scope


async def evaluate_query(query_text: str) -> Dict[str, Any]:
    res = await run_pipeline(raw_input=query_text, input_type="text")
    recs = res.get("recommendations", [])
    closest = res.get("closest_matches", [])
    top_cand = recs[0] if recs else (closest[0] if closest else {})
    return {
        "abstained": res.get("abstained", False),
        "top_strength": float(top_cand.get("match_strength", 0.0)),
        "top_coverage": float(top_cand.get("coverage", 0.0)),
        "has_literal": any(
            c.get("source") == "literal_mention"
            or c.get("is_literal_mention")
            or c.get("is_successor_promotion")
            for c in (recs + closest)
        ),
    }


async def main():
    parser = argparse.ArgumentParser(description="Calibrate pipeline thresholds")
    parser.add_argument("--mode", choices=["offline", "full"], default="offline")
    args = parser.parse_args()

    if args.mode == "offline":
        os.environ["LLM_PROVIDER"] = "mock"

    kl.load_all()

    in_scope, out_of_scope = load_training_queries()
    print(f"Loaded {len(in_scope)} in-scope and {len(out_of_scope)} out-of-scope training queries.")

    print("\nRunning pipeline on in-scope training queries...")
    in_scope_results = []
    for q in in_scope:
        res = await evaluate_query(q["query"])
        in_scope_results.append(res)

    print("Running pipeline on out-of-scope training queries...")
    out_of_scope_results = []
    for q in out_of_scope:
        res = await evaluate_query(q["query"])
        out_of_scope_results.append(res)

    # Grid search threshold space
    best_f1 = -1.0
    best_params = None

    # Abstain thresholds to test
    abstain_thresh_grid = [0.20, 0.25, 0.30, 0.35, 0.40]
    coverage_floor_grid = [0.10, 0.15, 0.20, 0.25]
    low_match_grid = [0.40, 0.45, 0.50, 0.55]

    print("\n--- Sweeping Threshold Grid ---")
    for a_th in abstain_thresh_grid:
        for c_fl in coverage_floor_grid:
            # Simulate decision rule:
            # abstain if not has_literal and top_strength < a_th and top_coverage < c_fl
            # TP = out_of_scope correctly abstained
            # FP = in_scope wrongly abstained (false abstention)
            # FN = out_of_scope wrongly NOT abstained
            # TN = in_scope correctly NOT abstained
            
            in_abstained = sum(
                1 for r in in_scope_results
                if not r["has_literal"] and r["top_strength"] < a_th and r["top_coverage"] < c_fl
            )
            false_abstention_rate = in_abstained / len(in_scope_results) if in_scope_results else 0.0

            out_abstained = sum(
                1 for r in out_of_scope_results
                if not r["has_literal"] and r["top_strength"] < a_th and r["top_coverage"] < c_fl
            )
            tp = out_abstained
            fp = in_abstained
            fn = len(out_of_scope_results) - out_abstained
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Constraint: False-abstention rate on in-scope <= 3% (0.03)
            if false_abstention_rate <= 0.03:
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        "ABSTAIN_THRESHOLD": a_th,
                        "ABSTAIN_COVERAGE_FLOOR": c_fl,
                        "F1": round(f1, 4),
                        "Precision": round(precision, 4),
                        "Recall": round(recall, 4),
                        "FalseAbstentionRate": round(false_abstention_rate, 4),
                    }

    print(f"\nOptimal Calibration Results for [{args.mode.upper()}] Mode:")
    print(json.dumps(best_params, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
