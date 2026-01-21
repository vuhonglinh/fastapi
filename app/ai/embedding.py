import torch
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel

VNCORENLP_DIR = r"D:\vncorenlp"
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

if __name__ == "__main__":
    text = (
        "Xét hàm số $f(x) = x^3 - 3x^2 + 2x - 1$. Tính đạo hàm $f'(x) = 3x^2 - 6x + 2$ và giải phương trình $f'(x) = 0$. Xét dấu của $f'(x)$ trên các khoảng xác định, từ đó suy ra các điểm cực trị của hàm số. Cuối cùng, tính tích phân xác định $\int_0^1 (x^3 - 3x^2 + 2x - 1)\,dx$.")

    vec = embed(text)
    print(vec)
    print("Vector shape:", vec.shape)
