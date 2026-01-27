from fastapi import FastAPI
from app.routers.api import router
from config import Settings
import uvicorn
from contextlib import asynccontextmanager
from app.core.scheduler import start_scheduler, shutdown_scheduler

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

app.include_router(router) 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)