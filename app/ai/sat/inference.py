import re
import json
import torch
from pathlib import Path
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from typing import Optional, Tuple, Dict

# =====================
# DEV FIX CỨNG (TEST)
# =====================
BANK_ID = "A1234"
MODEL_FAMILY = "roberta-large"
MODEL_VERSION: Optional[str] = "v2"   # None = latest

MODEL_BASE_DIR = Path("models")
LABEL_PATH = "data/subskill_label2id.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# LOAD GLOBAL TAXONOMY
# =====================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    name2global = json.load(f)
GLOBAL2NAME = {v: k for k, v in name2global.items()}

# Cache: (bank_id, family, version/latest) -> (model, tokenizer, dense2global, version_name)
_MODEL_CACHE: Dict[Tuple[str, str, str], tuple] = {}

def get_latest_version_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"❌ Model root không tồn tại: {root}")

    versions = []
    for p in root.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"v(\d+)", p.name)
            if m:
                versions.append((int(m.group(1)), p))

    if not versions:
        raise FileNotFoundError(f"❌ Không tìm thấy model version nào trong {root}")

    return sorted(versions, key=lambda x: x[0])[-1][1]

def resolve_model_dir(bank_id: str, model_family: str, model_version: Optional[str]) -> Path:
    root = MODEL_BASE_DIR / bank_id / model_family
    if not root.exists():
        raise FileNotFoundError(f"❌ Không thấy model root: {root}")

    if model_version is None:
        return get_latest_version_dir(root)

    p = root / model_version
    if not p.exists():
        raise FileNotFoundError(f"❌ Không thấy version: {p}")
    return p

def load_model(bank_id: str, model_family: str, model_version: Optional[str] = None):
    key = (bank_id, model_family, model_version or "latest")
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_dir = resolve_model_dir(bank_id, model_family, model_version)
    version_name = model_dir.name

    model_path = model_dir / "model"
    tok_path = model_dir / "tokenizer"
    remap_path = model_dir / "label_remap.json"

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Thiếu thư mục model: {model_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"❌ Thiếu thư mục tokenizer: {tok_path}")
    if not remap_path.exists():
        raise FileNotFoundError(f"❌ Thiếu file remap: {remap_path}")

    print(f"🚀 Loading: {bank_id}/{model_family}/{version_name}")

    model = RobertaForSequenceClassification.from_pretrained(str(model_path)).to(DEVICE)
    tokenizer = RobertaTokenizer.from_pretrained(str(tok_path))
    model.eval()

    with open(remap_path, "r", encoding="utf-8") as f:
        remap = json.load(f)
    dense2global = {int(k): int(v) for k, v in remap["dense2orig"].items()}

    _MODEL_CACHE[key] = (model, tokenizer, dense2global, version_name)
    return _MODEL_CACHE[key]

@torch.no_grad()
def predict(question: str, bank_id: str, model_family: str, model_version: Optional[str] = None) -> dict:
    model, tokenizer, dense2global, version_name = load_model(bank_id, model_family, model_version)

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(DEVICE)

    logits = model(**inputs).logits  # [1, num_labels]
    probs = torch.softmax(logits, dim=-1)
    dense_id = int(torch.argmax(probs, dim=-1).item())
    confidence = float(probs[0, dense_id].item())

    global_id = dense2global[dense_id]
    label = GLOBAL2NAME.get(global_id, f"UNKNOWN_{global_id}")

    return {
        "bank_id": bank_id,
        "model_family": model_family,
        "model_version": version_name,
        "dense_id": dense_id,
        "global_id": global_id,
        "label": label,
        "confidence": round(confidence, 6),
    }

if __name__ == "__main__":
    # pre-load 1 lần cho chắc (optional)
    load_model(BANK_ID, MODEL_FAMILY, MODEL_VERSION)

    while True:
        q = input("\nNhập câu hỏi (enter để thoát): ").strip()
        if not q:
            break
        print(predict(q, BANK_ID, MODEL_FAMILY, MODEL_VERSION))
