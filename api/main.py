from fastapi import FastAPI
from pydantic import BaseModel

from chengyu.argmax import rank
from chengyu.evaluation import load_dictionary

app = FastAPI(title="Chinese Idiom Inference", version="0.1.0")

# Full dictionary (~31k idioms), loaded once at startup.
_dictionary, _ = load_dictionary()
IDIOMS = list(_dictionary)


class Query(BaseModel):
    text: str
    top_k: int = 3
    candidates: list[str] | None = None  # optional: restrict instead of using the full dictionary


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(q: Query):
    idioms = q.candidates or IDIOMS
    top = rank(q.text, idioms, k=q.top_k)
    return {
        "query": q.text,
        "predictions": [{"idiom": i, "score": s} for i, s in top],
    }
