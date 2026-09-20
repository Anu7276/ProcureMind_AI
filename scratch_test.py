import os
os.environ["LLM_PROVIDER"] = "mock"
from backend.config.settings import settings
settings.LLM_PROVIDER = "mock"

import asyncio
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline

kl.load_all()

async def test():
    q = "spacecraft thermal protection ceramic tiles for atmospheric reentry"
    res = await run_pipeline(q, input_type="text")
    recs = res.get("recommendations", [])
    print("abstained:", res.get("abstained"))
    print("top recs:", [(r.get("key"), r.get("match_strength"), r.get("low_match")) for r in recs[:3]])
    print("low_match on first rec:", recs[0].get("low_match") if recs else None)

asyncio.run(test())
