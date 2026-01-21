import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from app.ai.embedding import embed 
 
CSV_PATH = r"D:\CT\fastapi\text.csv"
MODEL_DIR = "models"
TARGET_COL = "skill_id"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 1024

os.makedirs(MODEL_DIR, exist_ok=True)
 
df = pd.read_csv(CSV_PATH)
texts = df["text"].tolist()
labels = df[TARGET_COL].tolist()

unique_labels = sorted(set(labels))
# label2id = {label: i for i, label in enumerate(unique_labels)}
label2id = {}
for i, label in enumerate(unique_labels):
    label2id[label] = i

id2label = {i: label for label, i in label2id.items()}

y = torch.tensor([label2id[l] for l in labels], dtype=torch.long)
num_classes = len(label2id)
print(num_classes)

# print("Classes:", label2id)
 
# X = []

# print("🔄 Embedding texts...")
# for text in tqdm(texts):
#     vec = embed(text) 
#     X.append(vec)

# X = torch.stack(X)     
# print(X)
# dataset = TensorDataset(X, y)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

 
# class TorchClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)

# model = TorchClassifier(EMBED_DIM, num_classes)

# # =====================
# # TRAIN SETUP
# # =====================
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # =====================
# # TRAIN LOOP
# # =====================
# print("🧠 Training classifier...")
# for epoch in range(EPOCHS):
#     total_loss = 0.0
#     for xb, yb in loader:
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

# # =====================
# # SAVE MODEL
# # =====================
# torch.save(
#     {
#         "model_state": model.state_dict(),
#         "label2id": label2id,
#         "embed_dim": EMBED_DIM
#     },
#     f"{MODEL_DIR}/torch_classifier.pt"
# )

# print("✅ TRAINING DONE")
# print("Saved:")
# print(" - models/torch_classifier.pt")

# # =====================
# # EVALUATION (train set)
# # =====================
# with torch.no_grad():
#     logits = model(X)
#     y_pred = torch.argmax(logits, dim=1)

# acc = (y_pred == y).float().mean().item()
# print(f"\n📊 Accuracy (train): {acc:.4f}")
