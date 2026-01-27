from fastapi import APIRouter, BackgroundTasks 
from typing import Union
from app.ai.embedding import embed
from app.server.database import question_collection, bank_collection, label_collection
from config import Settings 
from app.ai.train import handle_label  
from pathlib import Path

settings = Settings()

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

@router.post("/train/{bank_id}")
async def train_bank(
    bank_id: str, 
    background_tasks: BackgroundTasks
):
    bank_dir = settings.models_dir / bank_id
    bank_dir.mkdir(parents=True, exist_ok=True)
    if not bank_dir.exists():
        return {
            "status": "error",
            "message": "Bank model directory not found"
        }

    background_tasks.add_task(
        handle_label,
        bank_id
    )

    return {
        "status": "queued",
        "bank_id": bank_id,
        "message": "Training job started in background"
    }
    
@router.post("/job/train-all")
async def train_all_banks(
    csv_base_dir: str,
    background_tasks: BackgroundTasks
):
    banks = [
        p.name for p in settings.models_dir.iterdir()
        if p.is_dir()
    ]

    for bank_id in banks:
        csv_path = Path(csv_base_dir) / f"{bank_id}.csv"
        if csv_path.exists():
            background_tasks.add_task(
                handle_label,
                bank_id,
                str(csv_path)
            )

    return {
        "status": "queued",
        "banks": len(banks),
        "message": "All training jobs queued"
    }
