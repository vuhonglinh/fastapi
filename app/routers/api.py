from fastapi import APIRouter
from typing import Union
from app.ai.embedding import embed

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
