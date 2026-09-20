import os
os.environ["LLM_PROVIDER"] = "mock"

from backend.config.settings import settings
settings.LLM_PROVIDER = "mock"

import asyncio
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline

kl.load_all()

async def test():
    # In-scope queries that were wrongly abstaining
    in_scope = [
        ("Q017", "Complete internal electrical wiring installation for newly constructed government administrative building.", ["IS 732", "IS 3043"]),
        ("Q008", "Steel pipes for laying of raw water transmission main pipeline under municipal corporation jurisdiction.", ["IS 3589"]),
        ("Q070", "Moulded case circuit breaker MCCB 400A for main distribution panel.", ["IS/IEC 60947"]),
    ]
    # Out-of-scope queries that must still abstain
    out_scope = [
        ("OOS1", "purple elephant submarine banana"),
        ("OOS2", "blockchain based voting machine for municipal election"),
        ("OOS3", "CRISPR Cas9 gene editing reagents for crop enhancement"),
        ("OOS4", "spacecraft thermal protection ceramic tiles for atmospheric reentry"),
    ]

    print("=== IN-SCOPE (should NOT abstain) ===")
    for qid, q, expected in in_scope:
        res = await run_pipeline(q, input_type="text")
        recs = [r.get("key") for r in res.get("recommendations", [])[:5]]
        cands = res.get("candidates", [])
        ms = cands[0].get("match_strength") if cands else None
        cov = cands[0].get("coverage") if cands else None
        print(f"{qid}: abstained={res.get('abstained')}, recs={recs[:3]}, ms={ms}, cov={cov}")

    print("\n=== OUT-OF-SCOPE (should abstain) ===")
    for qid, q in out_scope:
        res = await run_pipeline(q, input_type="text")
        cands = res.get("candidates", [])
        ms = cands[0].get("match_strength") if cands else None
        cov = cands[0].get("coverage") if cands else None
        print(f"{qid}: abstained={res.get('abstained')}, ms={ms}, cov={cov}")

asyncio.run(test())
