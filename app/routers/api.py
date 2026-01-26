from fastapi import APIRouter, BackgroundTasks 
from typing import Union
from app.ai.embedding import embed
from app.server.database import question_collection, bank_collection, label_collection
import traceback
from pathlib import Path

router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@router.get("/embed/{payload}")
def embed_api(payload: str):
    vec = embed(payload)
    return {
        "dim": vec.shape[0],
        "vector": vec.tolist()
    }


@router.get("/train-data")
async def embed_api():
    questions = []
    async for q in question_collection.find():
        q["_id"] = str(q["_id"])
        questions.append(q)
    return questions






MODELS_DIR = Path("models/data")

@router.get("/job")
async def job_api():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = await bank_collection.find().to_list()
    created = []
    for item in results:
        bank_id = str(item["_id"])
        item["_id"] = bank_id
        bank_dir = MODELS_DIR / bank_id
        bank_dir.mkdir(exist_ok=True)
        created.append(str(bank_dir))

    return {
        "status": "ok",
        "banks": len(results),
        "folders": created
    }

