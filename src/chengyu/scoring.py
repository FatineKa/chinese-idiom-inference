import os

import torch
from transformers import (  # two Hugging Face tools: AutoTokenizer (turns text into numbers) and AutoModelForCausalLM (loads a causal language model, i.e. one that predicts the next word)
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL = os.environ.get("CHENGYU_MODEL", "Qwen/Qwen2.5-7B-Instruct")  # override with e.g. CHENGYU_MODEL="Qwen/Qwen2.5-0.5B-Instruct" for a quick CPU test
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype = torch.bfloat16 if _device.type == "cuda" else torch.float32  # bf16: half the memory, near-lossless precision on a recent GPU; pointless on CPU
_4bit = os.environ.get("CHENGYU_4BIT") == "1"  # load in 4-bit (bitsandbytes NF4) to fit
                        # a large checkpoint on a smaller GPU (e.g. a free-tier T4).
                        # Needs CUDA. Changes the model's numerics slightly versus full
                        # precision -- fine for a compute-constrained test, worth a note
                        # in the chapter if used for results that get reported.
_tok = AutoTokenizer.from_pretrained(MODEL)  # loads the model's tokenizer
if _4bit:
    from transformers import BitsAndBytesConfig
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
else:
    _model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=_dtype).to(_device)  # loads the model into memory, on GPU if available
_model.eval()  # sets the model to evaluation mode

@torch.no_grad()  # gradients are for training; since we're only evaluating, disabling them makes the code faster and less memory-hungry
def log_prob_total(text: str) -> float:  # takes a text and returns its score
    """Sum of the log-probabilities Qwen assigns to the text."""
    ids = _tok(text, return_tensors="pt").input_ids.to(_device)  # turns the text into numbers (ids) and puts them in a PyTorch tensor, on the same device as the model
    logp = torch.log_softmax(_model(ids).logits[0], dim=-1)  # Qwen predicts the log-probability of each word: _model(ids) runs the numbers through the model, producing a (text length x vocabulary) matrix; log_softmax normalizes the scores into log-probabilities, dim=-1 means we normalize over the vocabulary dimension
    total = 0.0
    for k in range(1, ids.shape[1]):  # walk through the text's tokens (from 1 to the text length); we start at 1 because the first token has no preceding prediction
        total += logp[k - 1, ids[0, k]].item()  # the log-probability of token k sits at row k-1 (the preceding prediction) and the column matching token k's id; add it to the running total, item() converts the PyTorch tensor to a Python float, total += : summed because the probability of a sequence is the product of each token's probability, and the log of a product is the sum of the logs
    return total  # returns the sum of log-probabilities: the text's score under Qwen

@torch.no_grad()
def score_summary(text, idiom):
    """Score of: does this idiom summarize this text? One forward pass on
    prompt+text, summing the log-probabilities of the text's own tokens
    only -- equivalent to log p(prompt+text) - log p(prompt) via the chain
    rule, but half as expensive. Same tokenization scheme as
    argmax.text_scores, so the two stay numerically comparable."""
    prompt_ids = _tok(f"成语「{idiom}」概括了这句话：").input_ids      # "The idiom X summarizes this sentence:"
    text_ids = _tok(text, add_special_tokens=False).input_ids
    ids = torch.tensor([prompt_ids + text_ids], device=_device)
    logp = torch.log_softmax(_model(ids).logits[0], dim=-1)
    total = 0.0
    for k in range(len(prompt_ids), len(prompt_ids) + len(text_ids)):  # text tokens only
        total += logp[k - 1, ids[0, k]].item()
    return total


"""Let's trace a text all the way through log_prob_total

Take a simple example (in English, to illustrate the mechanics):
text = "the cat sleeps"


1. Text -> tokens (numbers). The tokenizer splits and numbers it:
   "the cat sleeps" -> [ 12, 88, 305 ]


2. Numbers -> model -> logits. The model reads these numbers and produces,
   at every position, a raw score for every possible word. This is a large
   matrix: (3 positions) x (vocabulary size, ~150,000)


3. Logits -> log-probabilities. log_softmax turns each row into
   probabilities (summing to 1), then takes the logarithm


4. We read off the log-probability of the actual next word. At position 0
   ("the"), the model predicts word 1 ("cat"): we read log p(cat | the),
   say -0.7. At position 1 ("the cat"), we read log p(sleeps | the cat),
   say -1.2.


5. We sum them. Score = -0.7 + (-1.2) = -1.9."""
