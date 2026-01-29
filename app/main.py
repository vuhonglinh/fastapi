from fastapi import FastAPI
from app.routers.api import router
from config import Settings
import uvicorn
from contextlib import asynccontextmanager
from app.core.scheduler import start_scheduler, shutdown_scheduler
from fastapi.middleware.cors import CORSMiddleware

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    shutdown_scheduler()

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan  
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
app.include_router(router) 





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)