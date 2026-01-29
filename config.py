from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Settings(BaseSettings):
    models_dir: Path = Path("models/edupen")    
    models_dir_edly: Path = Path("models/edly")
    vncorenlp_dir: str = os.getenv('VNCORENLP_DIR')
    csv_path: str = os.getenv('CSV_PATH')
    app_name: str = os.getenv("APP_NAME")
    
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    embed_dim: int = 1024
    confidence_threshold: float = 0.9
    

settings = Settings()