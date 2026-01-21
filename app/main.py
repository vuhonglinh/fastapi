from fastapi import FastAPI
from app.routers.api import router
import os
 

app = FastAPI(title=os.getenv("APP_NAME", "FastAPI"))

app.include_router(router)
