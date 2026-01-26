import os
import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime

from app.ai.embedding import embed

# =====================
# CONFIG
# =====================
CSV_PATH = "/var/www/html/fastapi/text.csv"
BASE_MODEL_DIR = "models/skill_classifier"
TARGET_COL = "skill_id"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 1024

MODEL_VERSION = "v1"
MODEL_DIR = os.path.join(BASE_MODEL_DIR, MODEL_VERSION)
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(CSV_PATH)

texts = df["text"].tolist()
labels = df[TARGET_COL].tolist()

# label mapping
unique_labels = sorted(set(labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

y = torch.tensor([label2id[l] for l in labels], dtype=torch.long)
num_classes = len(label2id)

print("Classes:", label2id)

# =====================
# EMBEDDING
# =====================
X = []
print("🔄 Embedding texts...")
for text in tqdm(texts):
    X.append(embed(text))

X = torch.stack(X)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# MODEL
# =====================
class TorchClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

model = TorchClassifier(EMBED_DIM, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================
# TRAIN
# =====================
print("🧠 Training classifier...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

# =====================
# EVAL (TRAIN SET – baseline)
# =====================
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    conf, y_pred = probs.max(dim=1)

acc = (y_pred == y).float().mean().item()
avg_conf = conf.mean().item()

print(f"\n📊 Train accuracy: {acc:.4f}")
print(f"📈 Avg confidence: {avg_conf:.4f}")

# =====================
# SAVE MODEL (PRODUCTION WAY)
# =====================

# 1️⃣ model weights
torch.save(model.state_dict(), f"{MODEL_DIR}/model.pt")

# 2️⃣ label map
with open(f"{MODEL_DIR}/label_map.json", "w") as f:
    json.dump(label2id, f, indent=2)

# 3️⃣ config
config = {
    "embed_dim": EMBED_DIM,
    "num_classes": num_classes,
    "architecture": "linear",
    "loss": "cross_entropy"
}
with open(f"{MODEL_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# 4️⃣ meta (rất quan trọng cho self-training)
meta = {
    "version": MODEL_VERSION,
    "trained_at": datetime.utcnow().isoformat(),
    "dataset": os.path.basename(CSV_PATH),
    "num_samples": len(dataset),
    "train_accuracy": acc,
    "avg_confidence": avg_conf,
    "confidence_threshold_recommended": 0.9,
    "notes": "initial supervised model"
}
with open(f"{MODEL_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ TRAINING DONE")
print(f"Saved model to: {MODEL_DIR}")
