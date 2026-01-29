import os, re, json, time, logging
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments

# =====================
# CONFIG (FIX CỨNG TẠM)
# =====================
BANK_ID = "A1234"
MODEL_FAMILY = "roberta-large"

DATA_PATH = "data/sat_questions_new.json"       
TAXONOMY_PATH = "data/subskill_label2id.json"   

MODEL_ROOT = Path("models") / BANK_ID / MODEL_FAMILY  
# BASE_MODEL_DIR = Path("models/hf/roberta-large")  
BASE_MODEL_DIR = "roberta-base"
    

logging.basicConfig(level=logging.CRITICAL)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_version_dir(root: Path) -> Path:
    if not root.exists():
        return None
    versions = []
    for p in root.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"v(\d+)", p.name)
            if m:
                versions.append((int(m.group(1)), p))
    if not versions:
        return None
    return sorted(versions, key=lambda x: x[0])[-1][1]

def next_version_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in root.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"v(\d+)", p.name)
            if m:
                versions.append(int(m.group(1)))
    n = (max(versions) + 1) if versions else 1
    v = f"v{n}"
    return v, root / v

# =====================
# LOAD TAXONOMY
# =====================
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    name2global = json.load(f)
global2name = {v: k for k, v in name2global.items()}

# =====================
# LOAD NEW DATA (JSON)
# =====================
ds = load_dataset("json", data_files=DATA_PATH)["train"]
ds = ds.train_test_split(test_size=0.1)

all_global = list(ds["train"]["label"]) + list(ds["test"]["label"])
all_global = [int(x) for x in all_global]

unknown = sorted({x for x in all_global if x not in global2name})
if unknown:
    raise ValueError(f"❌ File data có label chưa có trong taxonomy: {unknown}")

unique_global = sorted(set(all_global))
num_labels = len(unique_global)

# remap dense theo label xuất hiện trong file mới
orig2dense = {g: i for i, g in enumerate(unique_global)}
dense2orig = {i: g for g, i in orig2dense.items()}

id2label = {i: global2name[dense2orig[i]] for i in range(num_labels)}
label2id = {v: k for k, v in id2label.items()}

print(f"✅ labels_in_new_file={num_labels} dense=0..{num_labels-1}")

# =====================
# CHỌN MODEL ĐỂ TRAIN TIẾP
# =====================
latest = get_latest_version_dir(MODEL_ROOT)
if latest is None:
    # chưa có model nào -> train từ base
    load_model_dir = BASE_MODEL_DIR
    load_tok_dir = BASE_MODEL_DIR
    print("⚠️ Chưa có version nào -> train từ base roberta-large")
else:
    load_model_dir = latest / "model"
    load_tok_dir = latest / "tokenizer"
    print(f"🔁 Continue training from: {latest.name}")

# NOTE: chỉ continue được nếu label-set KHÔNG đổi so với model cũ
# (vì classifier head shape phải khớp)
if latest is not None:
    remap_path = latest / "label_remap.json"
    if not remap_path.exists():
        raise FileNotFoundError(f"❌ Missing label_remap.json in {latest}")
    with open(remap_path, "r", encoding="utf-8") as f:
        old_remap = json.load(f)
    old_labels = set(int(v) for v in old_remap["dense2orig"].values())
    new_labels = set(unique_global)
    if new_labels != old_labels:
        raise RuntimeError(
            "❌ Label-set đã thay đổi so với model latest.\n"
            f"latest={sorted(old_labels)}\nnew={sorted(new_labels)}\n"
            "=> Không thể train tiếp. Hãy chạy fine_tune.py (rebuild head) để publish version mới."
        )

# =====================
# LOAD MODEL + TOKENIZER
# =====================
model = RobertaForSequenceClassification.from_pretrained(
    str(load_model_dir),
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
).to(DEVICE)

tokenizer = RobertaTokenizer.from_pretrained(str(load_tok_dir))

def tokenize(batch):
    tok = tokenizer(
        batch["question"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    tok["labels"] = [orig2dense[int(x)] for x in batch["label"]]
    return tok

tok_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =====================
# TRAIN
# =====================
VERSION, SAVE_DIR = next_version_dir(MODEL_ROOT)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=str(SAVE_DIR / "logs"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["test"],
)

trainer.train()

# =====================
# SAVE VERSION
# =====================
(model_dir := SAVE_DIR / "model").mkdir(parents=True, exist_ok=True)
(tok_dir := SAVE_DIR / "tokenizer").mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(model_dir))
tokenizer.save_pretrained(str(tok_dir))

with open(SAVE_DIR / "label_remap.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "global_ids_in_this_version": unique_global,
            "orig2dense": {str(k): v for k, v in orig2dense.items()},
            "dense2orig": {str(k): v for k, v in dense2orig.items()},
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(SAVE_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "bank_id": BANK_ID,
            "model_family": MODEL_FAMILY,
            "version": VERSION,
            "continued_from": latest.name if latest else None,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": DATA_PATH,
            "num_labels": num_labels,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"✅ Continue training completed → saved to {SAVE_DIR}")
