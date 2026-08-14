from fastapi import FastAPI
from pydantic import BaseModel

from chengyu.argmax import exact_posterior, text_scores
from chengyu.evaluation import idioms_of_length, load_dictionary
from chengyu.prior import log_prior

app = FastAPI(title="Chinese Idiom Inference", version="0.1.0")

# Full dictionary (~31k idioms), loaded once at startup.
_dictionary, _ = load_dictionary()
IDIOMS = list(_dictionary)

# {(text, idiom): score}, shared across requests so a repeated (text, idiom)
# pair -- e.g. the same text queried again with a different top_k -- is
# scored once.
_score_cache: dict = {}


class Query(BaseModel):
    text: str
    top_k: int = 3
    length: int | None = None  # restrict candidates to I_n (idioms with exactly n characters)
    candidates: list[str] | None = None  # optional: restrict instead of using the full dictionary


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(q: Query):
    idioms = q.candidates or (idioms_of_length(IDIOMS, q.length) if q.length else IDIOMS)
    likelihood = text_scores(q.text, idioms, cache=_score_cache)
    h = {i: likelihood[i] + log_prior(i) for i in idioms}
    # posterior is normalized over the whole candidate set, not just the top-k
    posterior = exact_posterior(h)
    ranked = sorted(h.items(), key=lambda kv: kv[1], reverse=True)[:q.top_k]
    return {
        "query": q.text,
        "predictions": [
            {
                "idiom": i,
                "log_likelihood": likelihood[i],
                "log_prior": log_prior(i),
                "score": h[i],
                "posterior": posterior[i],
            }
            for i, _ in ranked
        ],
    }
