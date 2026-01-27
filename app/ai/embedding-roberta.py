import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from config import Settings

settings = Settings()

MODEL_NAME = "FacebookAI/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    add_prefix_space=True
)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed(text: str) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(
        outputs.last_hidden_state,
        inputs["attention_mask"]
    )

    # Normalize để cosine similarity / retrieval ổn định
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0)

if __name__ == "__main__":
    text = (
        "Xét hàm số f(x) = x^3 - 3x^2 + 2x - 1. "
        "Tính đạo hàm và xét cực trị của hàm số."
    )

    vec = embed(text)
    print(vec)
    print("Vector shape:", vec.shape)
