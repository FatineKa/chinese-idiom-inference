import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_tok = AutoTokenizer.from_pretrained(MODEL)
_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
_model.eval()

@torch.no_grad()
def log_prob_total(texte: str) -> float:
    """Somme des log-probabilités que Qwen attribue au texte."""
    ids = _tok(texte, return_tensors="pt").input_ids
    logp = torch.log_softmax(_model(ids).logits[0], dim=-1)
    total = 0.0
    for k in range(1, ids.shape[1]):
        total += logp[k - 1, ids[0, k]].item()
    return total