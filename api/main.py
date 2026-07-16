from fastapi import FastAPI
from pydantic import BaseModel

from chengyu.argmax import classer
from chengyu.evaluation import charger_dico

app = FastAPI(title="Chinese Idiom Inference", version="0.1.0")

# Dictionnaire complet (~31k idiomes), chargé une seule fois au démarrage.
_dico, _ = charger_dico()
IDIOMS = list(_dico)


class Query(BaseModel):
    text: str
    top_k: int = 3
    candidates: list[str] | None = None  # optionnel : restreindre au lieu du dictionnaire complet


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(q: Query):
    idioms = q.candidates or IDIOMS
    top = classer(q.text, idioms, k=q.top_k)
    return {
        "query": q.text,
        "predictions": [{"idiom": i, "score": s} for i, s in top],
    }
