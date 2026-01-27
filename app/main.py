from fastapi import FastAPI
from app.routers.api import router
from config import Settings
import uvicorn

settings = Settings()

app = FastAPI(title=settings.app_name)

app.include_router(router) 

 