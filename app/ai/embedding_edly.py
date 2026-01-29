# app/ai/embedding_edly.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "roberta-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load 1 lần
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
_model = AutoModel.from_pretrained(MODEL_NAME,add_pooling_layer=False).to(DEVICE)
_model.eval()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed(text: str, max_length: int = 256, normalize: bool = True) -> torch.Tensor:
    """
    Return: 1D tensor [1024] for roberta-large
    """
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = _model(**inputs)
    emb = _mean_pool(out.last_hidden_state, inputs["attention_mask"])

    if normalize:
        emb = F.normalize(emb, p=2, dim=1)

    return emb.squeeze(0).cpu() 
