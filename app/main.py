from fastapi import FastAPI
from app.routers.api import router
import os
import uvicorn 
print("CWD:", os.getcwd())

app = FastAPI(title=os.getenv("APP_NAME", "FastAPI"))

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)