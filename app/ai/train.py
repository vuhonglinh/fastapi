from pathlib import Path 
import json
import torch
import re 
from html import unescape
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
import shutil
from config import settings
from app.ai.embedding import embed
from app.server.database import question_collection, label_collection


def get_next_version(bank_dir: Path) -> str:
    versions = []
    for p in bank_dir.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            try:
                versions.append(int(p.name.replace("v", "")))
            except ValueError:
                pass
    return f"v{max(versions) + 1}" if versions else "v1"


def log(bank_dir: Path, msg: str):
    log_file = bank_dir / "training.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] {msg}\n")


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_question_text(q: dict) -> str:
    parts = []

    parts.append(clean_html(q.get("content", "")))

    for opt in q.get("options", []):
        parts.append(clean_html(opt.get("content", "")))

    if q.get("explanation"):
        parts.append(clean_html(q["explanation"]))

    return " ".join(parts)


async def handle_label(bank_id: str):
    bank_dir = settings.models_dir / bank_id
    bank_dir.mkdir(parents=True, exist_ok=True)

    version = get_next_version(bank_dir)
    model_dir = bank_dir / version
    model_dir.mkdir(parents=True, exist_ok=True)

 
    labels = await label_collection.find(
        {"bank_id": bank_id}
    ).to_list(None) 
    
    if not labels:
        raise ValueError(f"No labels found for bank {bank_id}")

    label_ids = [str(l["_id"]) for l in labels]
    label_id_to_name = {
        str(l["_id"]): l.get("name", str(l["_id"]))
        for l in labels
    }

 
    questions = await question_collection.find(
        {"labels": {"$in": label_ids}}
    ).to_list(None) 
    
    if not questions:
        raise ValueError("No questions found for training")

    texts = []
    y_labels = []

    for q in questions:
        q_text = build_question_text(q)
        for lid in q.get("labels", []):
            if lid in label_id_to_name:
                texts.append(q_text)
                y_labels.append(label_id_to_name[lid])

    if not texts:
        raise ValueError("No valid training samples built")

    unique_labels = sorted(set(y_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    y = torch.tensor([label2id[l] for l in y_labels], dtype=torch.long)
    num_classes = len(label2id)

 
    X = []
    for text in tqdm(texts, desc=f"[{bank_id}] Embedding"):
        X.append(embed(text))

    X = torch.stack(X)
    dataset = TensorDataset(X, y)
    log(bank_dir, X)
    log(bank_dir, y)

    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True  # giúp tốt cho học sâu
    )

 
    class TorchClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    model = TorchClassifier(settings.embed_dim, num_classes)  # Khởi tạo model


    latest_dir = bank_dir / "latest"
    latest_model = latest_dir / "model.pt"
    latest_label_map = latest_dir / "label_map.json"

    finetune = False

    if latest_model.exists() and latest_label_map.exists():
        with open(latest_label_map, "r", encoding="utf-8") as f:
            old_label2id = json.load(f)

        if old_label2id == label2id:
            model.load_state_dict(
                torch.load(latest_model, map_location="cpu")
            )
            finetune = True
            log(bank_dir, "♻️ Fine-tune from latest model")
        else:
            log(bank_dir, "⚠️ Label map changed → train from scratch")
    else:
        log(bank_dir, "🆕 No previous model → train from scratch")

 
    criterion = nn.CrossEntropyLoss()  # hàm phạt

    # nếu fine-tune → giảm learning rate
    lr = settings.lr * 0.1 if finetune else settings.lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr  # tự sửa mình
    )

 
    for epoch in range(settings.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        log(bank_dir, f"Epoch {epoch+1}/{settings.epochs} loss={total_loss:.4f}")

 
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        conf, y_pred = probs.max(dim=1)

    acc = (y_pred == y).float().mean().item()
    avg_conf = conf.mean().item()

    # Accuracy : % câu đúng
    # AvgConf :% tự tin đoán đúng
    log(bank_dir, f"📊 Accuracy={acc:.4f} | AvgConf={avg_conf:.4f}")

 
    torch.save(model.state_dict(), model_dir / "model.pt")

    with open(model_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "embed_dim": settings.embed_dim,
            "num_classes": num_classes,
            "architecture": "linear",
            "optimizer": "adam",
            "loss": "cross_entropy",
            "finetune": finetune
        }, f, indent=2)

    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "bank_id": bank_id,
            "version": version,
            "trained_at": datetime.utcnow().isoformat(),
            "num_samples": len(dataset),
            "accuracy": acc,
            "avg_confidence": avg_conf,
            "finetune": finetune
        }, f, indent=2)

    update_latest(bank_dir, model_dir)

    log(bank_dir, f"✅ Training {version} DONE")

    return {
        "bank_id": bank_id,
        "version": version,
        "accuracy": acc,
        "finetune": finetune
    }


def update_latest(bank_dir: Path, model_dir: Path):
    latest = bank_dir / "latest"

    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)

    try:
        latest.symlink_to(model_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(model_dir, latest)

