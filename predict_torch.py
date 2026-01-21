import torch
from torch import nn
from app.ai.embedding import embed

# =====================
# LOAD MODEL
# =====================
CKPT_PATH = "models/torch_classifier.pt"

ckpt = torch.load(CKPT_PATH, map_location="cpu")

label2id = ckpt["label2id"]
id2label = {v: k for k, v in label2id.items()}
EMBED_DIM = ckpt["embed_dim"]
NUM_CLASSES = len(label2id)

# =====================
# DEFINE MODEL (PHẢI GIỐNG LÚC TRAIN)
# =====================
class TorchClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

model = TorchClassifier(EMBED_DIM, NUM_CLASSES)
model.load_state_dict(ckpt["model_state"])
model.eval()

# =====================
# PREDICT FUNCTION
# =====================
def predict(text: str) -> str:
    with torch.no_grad():
        vec = embed(text)              # (1024,)
        vec = vec.unsqueeze(0)         # (1, 1024)

        logits = model(vec)            # (1, num_classes)
        pred_id = torch.argmax(logits, dim=1).item()

    return id2label[pred_id]

# =====================
# CLI TEST
# =====================
if __name__ == "__main__":
    while True:
        text = input("Nhập câu (q để thoát): ")
        if text.lower() == "q":
            break
        print("👉 Dự đoán:", predict(text))
