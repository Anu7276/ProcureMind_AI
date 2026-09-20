import asyncio
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline_from_requirement

async def main():
    kl.load_all()
    structured_req = {
        "product": "Fire extinguisher dry powder 9 kg ABC",
        "material": "",
        "specifications": "",
        "performance_requirements": "",
        "safety_requirements": "",
        "application": "",
    }
    original_text = "Supply of TMT steel bars Fe 500D conforming to IS 1786"

    res = await run_pipeline_from_requirement(
        structured_requirement=structured_req,
        normalized_text=original_text,
        user_edited=True,
    )
    print("Stages completed:", res.get("stages_completed"))
    print("Literal codes:", res.get("literal_codes"))
    print("Thesaurus expansions:", res.get("thesaurus_expansions"))
    for r in res.get("recommendations", []):
        print("REC:", r.get("key"), r.get("score"), r.get("source"), r.get("retrieval_trace"))

if __name__ == "__main__":
    asyncio.run(main())
