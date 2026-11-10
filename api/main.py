import json
import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
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

# Demo only, loaded if the files exist:
# - shortlist index (A) <- scripts/30_build_shortlist_index.py
# - example index (B)   <- scripts/33_build_examples_index.py
# - English meanings    <- scripts/32_build_translations.py (CC-CEDICT)
_index = None
if os.path.exists("data/shortlist_index.npz"):
    from chengyu.shortlist import load_index, shortlist
    _index = load_index()
_examples = None
if os.path.exists("data/examples_index.npz"):
    from chengyu.neighbours import load_examples, neighbour_candidates
    _examples = load_examples()
_english = {}
if os.path.exists("data/idiom_en.json"):
    with open("data/idiom_en.json", encoding="utf-8") as f:
        _english = json.load(f)


class Query(BaseModel):
    text: str
    top_k: int = 3
    length: int | None = None  # restrict candidates to I_n (idioms with exactly n characters)
    candidates: list[str] | None = None  # optional: restrict instead of using the full dictionary
    shortlist_k: int | None = None  # demo shortcut: score only the k most promising idioms (None = full dictionary)


@app.get("/")
def demo_page():
    return FileResponse("web/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "shortlist_available": _index is not None,
            "examples_available": _examples is not None}


@app.post("/predict")
def predict(q: Query):
    t0 = time.time()
    # script 31, recall@200: B 63% > A+B 55% > A 13% -> B, A only as fallback
    if q.shortlist_k and not q.candidates and _examples is not None:
        idioms = neighbour_candidates(q.text, *_examples, k=q.shortlist_k)   # B
        mode = f"shortlist of {len(idioms)}"
    elif q.shortlist_k and not q.candidates and _index is not None:
        idioms = shortlist(q.text, *_index, k=q.shortlist_k)                 # A
        mode = f"shortlist of {len(idioms)}"
    else:
        idioms = q.candidates or (idioms_of_length(IDIOMS, q.length) if q.length else IDIOMS)
        mode = "candidates" if q.candidates else "full dictionary"
    if len(_score_cache) > 200_000:   # cap memory on a public server
        _score_cache.clear()
    likelihood = text_scores(q.text, idioms, cache=_score_cache)
    h = {i: likelihood[i] + log_prior(i) for i in idioms}
    # posterior is normalized over the whole candidate set, not just the top-k
    posterior = exact_posterior(h)
    ranked = sorted(h.items(), key=lambda kv: kv[1], reverse=True)[:q.top_k]
    return {
        "query": q.text,
        "mode": mode,                      # posterior is normalized over these candidates
        "n_candidates": len(idioms),
        "seconds": round(time.time() - t0, 2),
        "predictions": [
            {
                "idiom": i,
                "log_likelihood": likelihood[i],
                "log_prior": log_prior(i),
                "score": h[i],
                "posterior": posterior[i],
                "pinyin": _english.get(i, {}).get("pinyin"),   # None if not in CC-CEDICT
                "english": _english.get(i, {}).get("en"),
            }
            for i, _ in ranked
        ],
    }
