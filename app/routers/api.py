from fastapi import APIRouter
from typing import Union
from app.ai.embedding import embed
from app.server.database import question_collection
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
