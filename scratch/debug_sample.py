import asyncio
from pathlib import Path
from ai.knowledge import knowledge_loader as kl
from ai.pipeline.graph import run_pipeline

async def main():
    kl.load_all()
    sample_path = Path("BIS_Sahayak_Clean_Data/clean/samples/Sample_Tender_With_Obsolete_Standards.txt")
    content = sample_path.read_text(encoding="utf-8")
    result = await run_pipeline(raw_input=content, input_type="text")
    print("Stages completed:", result.get("stages_completed"))
    print("Abstained:", result.get("abstained"), "Reason:", result.get("abstain_reason"))
    print("Recommendations count:", len(result.get("recommendations", [])))
    for r in result.get("recommendations", []):
        print("REC:", r.get("key"), r.get("score"), r.get("status"), r.get("source"))
    for cm in result.get("closest_matches", []):
        print("CLOSEST:", cm.get("key"), cm.get("score"), cm.get("status"))

if __name__ == "__main__":
    asyncio.run(main())
