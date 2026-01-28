import torch
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
import os
from config import Settings

settings = Settings()

VNCORENLP_DIR = settings.vncorenlp_dir 
MODEL_NAME = "vinai/phobert-large"
 
rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=VNCORENLP_DIR
)
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def embed(text: str) -> torch.Tensor:
    segmented = rdrsegmenter.word_segment(text)
    seg_text = " ".join(segmented)

    inputs = tokenizer(
        seg_text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs) 
    embedding = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
    return embedding.squeeze(0)


