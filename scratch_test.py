import os
os.environ["LLM_PROVIDER"] = "mock"

from backend.config.settings import settings
settings.LLM_PROVIDER = "mock"

import asyncio
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline

kl.load_all()

async def test():
    queries = [
        ("Q017", "Complete internal electrical wiring installation for newly constructed government administrative building."),
        ("Q008", "Steel pipes for laying of raw water transmission main pipeline under municipal corporation jurisdiction."),
        ("Q024", "uPVC pipes and fittings for overhead water tank plumbing and potable water distribution in residential quarters."),
        ("Q070", "Moulded case circuit breaker MCCB 400A for main distribution panel.")
    ]
    for qid, q in queries:
        res = await run_pipeline(q, input_type="text")
        recs = [r.get("key") for r in res.get("recommendations", [])[:5]]
        cands = res.get("candidates", [])
        ms = cands[0].get("match_strength") if cands else None
        print(qid, q[:35], "->", recs, "top_ms:", ms, "abstained:", res.get("abstained"))

asyncio.run(test())
