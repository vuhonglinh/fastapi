import json
import torch
from pathlib import Path
from torch import nn
from config import settings


class TorchClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_model(bank_id: str):
    model_dir = settings.models_dir / bank_id / "latest"
    model_path = model_dir / "model.pt"
    label_map_path = model_dir / "label_map.json"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError("Model checkpoint not found")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = TorchClassifier(
        input_dim=config["embed_dim"],
        num_classes=config["num_classes"],
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()

    id2label = {v: k for k, v in label2id.items()}

    return model, label2id, id2label
