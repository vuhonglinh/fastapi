from fastapi import FastAPI
from app.routers.api import router

app = FastAPI(title="AI Service")

app.include_router(router)
