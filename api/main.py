from fastapi import FastAPI
from pydantic import BaseModel

from chengyu.scoring import score_resume  # ta fonction réelle

app = FastAPI(title="Chinese Idiom Inference", version="0.1.0")

# Petite liste par défaut pour que l'API tourne tout de suite.
# À remplacer par ton vrai vocabulaire d'idiomes (ex. chargé depuis data/ChID).
DEFAULT_IDIOMS = [
    "画蛇添足", "亡羊补牢", "守株待兔", "对牛弹琴", "井底之蛙",
    "掩耳盗铃", "塞翁失马", "一箭双雕", "熟能生巧", "掉以轻心",
]


class Query(BaseModel):
    text: str
    top_k: int = 3
    candidates: list[str] | None = None  # optionnel : fournir sa propre liste


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(q: Query):
    idioms = q.candidates or DEFAULT_IDIOMS
    scored = [(idiome, score_resume(q.text, idiome)) for idiome in idioms]
    scored.sort(key=lambda x: x[1], reverse=True)  # score haut = meilleur
    top = scored[: q.top_k]
    return {
        "query": q.text,
        "predictions": [{"idiom": i, "score": s} for i, s in top],
    }
