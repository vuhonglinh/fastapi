# scripts/fine_tune.py
import os, re, json, time, logging
import torch
from pathlib import Path
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# =====================
# CONFIG (FIX CỨNG TẠM)
# =====================
BANK_ID = "A1234"
MODEL_FAMILY = "roberta-large"

# BASE_MODEL_DIR = Path("models/hf/roberta-large")
BASE_MODEL_DIR = "roberta-large"

DATA_PATH = "data/sat_questions.json"
LABEL_PATH = "data/subskill_label2id.json"

SAVE_ROOT = Path("models") / BANK_ID / MODEL_FAMILY

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

VERSION, SAVE_DIR = next_version_dir(SAVE_ROOT)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.CRITICAL)

# =====================
# LOAD TAXONOMY (NAME <-> GLOBAL ID)
# =====================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    name2global = json.load(f)

global2name = {v: k for k, v in name2global.items()}

# =====================
# LOAD DATASET
# =====================
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset["train"].train_test_split(test_size=0.1)

all_global_labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"])
all_global_labels = [int(x) for x in all_global_labels]

# validate label tồn tại trong taxonomy
unknown = sorted({x for x in all_global_labels if x not in global2name})
if unknown:
    raise ValueError(
        f"❌ Dataset có label global-id chưa có trong {LABEL_PATH}: {unknown}\n"
        f"-> Thêm label vào subskill_label2id.json hoặc lọc data trước khi train."
    )

# =====================
# BUILD DENSE REMAP for TRAINING
# =====================
unique_global = sorted(set(all_global_labels))
orig2dense = {g: i for i, g in enumerate(unique_global)}
dense2orig = {i: g for g, i in orig2dense.items()}

num_labels = len(unique_global)

# model id2label/label2id: dùng tên label cho dễ đọc
id2label = {i: global2name[dense2orig[i]] for i in range(num_labels)}
label2id = {v: k for k, v in id2label.items()}

print(f"✅ Loaded taxonomy labels: {len(name2global)}")
print(f"✅ Found labels in dataset: {num_labels} (dense 0..{num_labels-1})")
print(f"🚀 Bank={BANK_ID} publish={VERSION} -> {SAVE_DIR}")

# =====================
# LOAD MODEL + TOKENIZER
# =====================
model = RobertaForSequenceClassification.from_pretrained(
    str(BASE_MODEL_DIR),
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

tokenizer = RobertaTokenizer.from_pretrained(str(BASE_MODEL_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =====================
# TOKENIZE + LABEL REMAP
# =====================
def tokenize_and_add_labels(examples):
    tokenized = tokenizer(
        examples["question"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    tokenized["labels"] = [orig2dense[int(x)] for x in examples["label"]]
    return tokenized

tokenized = dataset.map(
    tokenize_and_add_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =====================
# TRAINING CONFIG
# =====================
training_args = TrainingArguments(
    output_dir=str(SAVE_DIR / "logs"),
    num_train_epochs=5,
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
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()

# =====================
# SAVE MODEL + TOKENIZER + META + REMAP
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
            "base_model": str(BASE_MODEL_DIR),
            "num_labels": num_labels,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"✅ Training completed → saved to {SAVE_DIR}")
