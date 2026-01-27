import torch
from torch import nn
from fastapi import APIRouter, BackgroundTasks 
from typing import Union
from app.ai.embedding import embed
from app.server.database import question_collection, bank_collection, label_collection
from config import Settings 
from app.ai.train import handle_label  
from app.ai.model_loader import load_model  
from pathlib import Path
from pydantic import BaseModel

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
            )

    return {
        "status": "queued",
        "banks": len(banks),
        "message": "All training jobs queued"
    }



class BankPredict(BaseModel):
    bank_id: str
    text: str
@router.post("/predict-label")
async def predict_label(req: BankPredict):
    if not req.text.strip() or not req.bank_id.strip():
        raise HTTPException(status_code=400, detail="Text or bank_id is empty")

    try:
        model, label2id, id2label = load_model(req.bank_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    x = embed(req.text).unsqueeze(0)  
    print(x)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    top_id = probs.argmax(dim=1).item()
    confidence = probs[0, top_id].item()

    return {
        "bank_id": req.bank_id,
        "text": req.text,
        "predicted_label": id2label[top_id],
        "confidence": round(confidence, 4),
        "scores": {
            id2label[i]: round(probs[0, i].item(), 4)
            for i in range(len(id2label))
        },
    }