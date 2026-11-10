"""31_shortlist_recall.py — accuracy cost of the demo shortlist.

- recall@K = share of test texts whose correct idiom is in the top K.
- Computed for every layer (1 forward pass per text gives all layers).
- Not in the top K -> the demo cannot find it.
- Timing: shortlist + scoring K idioms, vs full dictionary (estimated).
- Compares, at the best layer:
  - A   = idiom vectors (shortlist.py)
  - B   = example-sentence neighbours (neighbours.py)
  - A+B = half from each (combine)
- Needs script 30 first, same CHENGYU_MODEL. B needs script 33 (skipped if missing).
- Output:
  - results/outputs/31_shortlist_recall.csv     (A, per layer)
  - results/outputs/31_shortlist_compare.csv    (A vs B vs A+B)

Run: CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct python scripts/31_shortlist_recall.py"""
import os
import time

import numpy as np
import pandas as pd

from chengyu.argmax import text_scores
from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.neighbours import (
    EXAMPLES_PATH,
    combine,
    encode,
    idioms_from_neighbours,
    load_examples,
)
from chengyu.representation import text_state_by_layer
from chengyu.shortlist import cosine_to_all, load_index, top_k

N_TEXTS = int(os.environ.get("N_TEXTS", "200"))
KS = [50, 100, 200, 500, 1000]
N_TIMED = 5          # texts used for timing
K_TIMED = 200

dictionary, lengths = load_dictionary()
idioms, vectors = load_index()
df = pd.read_csv("data/raw/cip/in_domain/test.in.csv").drop_duplicates(subset=["src", "dst"])

texts, targets = [], []
for src, dst in zip(df["src"], df["dst"]):
    target = find_idiom(src, dst, dictionary, lengths)
    if target is not None:
        texts.append(normalize(dst)); targets.append(target)
    if len(texts) == N_TEXTS:
        break
print(f"{len(texts)} test texts")
train = pd.read_csv("data/raw/cip/train.csv")
leaked = len(set(texts) & {normalize(d) for d in train["dst"]})
print(f"{leaked} test texts also appear word for word in train.csv (would make B look too good)")

# rank of the correct idiom (1 = best), per layer
target_pos = {i: j for j, i in enumerate(idioms)}
ranks = []   # (n_texts, n_layers)
for text, target in zip(texts, targets):
    states = text_state_by_layer(text).cpu().numpy()          # 1 pass, all layers
    row = []
    for v in states:
        sim = cosine_to_all(vectors, v)
        row.append(int((sim > sim[target_pos[target]]).sum()) + 1)
    ranks.append(row)
ranks = np.array(ranks)

table = pd.DataFrame({f"recall@{k}": (ranks <= k).mean(axis=0) for k in KS})
table.index.name = "layer"
os.makedirs("results/outputs", exist_ok=True)
table.to_csv("results/outputs/31_shortlist_recall.csv")
print(table.round(3).to_string())
best_layer = int(table[f"recall@{K_TIMED}"].idxmax())
print(f"\nbest layer for K={K_TIMED}: {best_layer} -> set CHENGYU_SHORTLIST_LAYER={best_layer}")

# A vs B vs A+B, at the best layer
max_k = max(KS)
a_lists = [top_k(cosine_to_all(vectors, text_state_by_layer(t)[best_layer].cpu().numpy()), idioms, max_k)
           for t in texts]
methods = {"A": lambda i, k: a_lists[i][:k]}
if os.path.exists(EXAMPLES_PATH):
    ex_idioms, ex_vectors = load_examples()
    text_vecs = encode(texts)
    b_lists = [idioms_from_neighbours(ex_vectors @ v, ex_idioms, max_k) for v in text_vecs]
    methods["B"] = lambda i, k: b_lists[i][:k]
    methods["A+B"] = lambda i, k: combine(a_lists[i], b_lists[i], k)
else:
    print(f"\n{EXAMPLES_PATH} missing -> only A (run script 33 for B)")
compare = pd.DataFrame({
    name: {f"recall@{k}": np.mean([targets[i] in get(i, k) for i in range(len(texts))]) for k in KS}
    for name, get in methods.items()
})
compare.to_csv("results/outputs/31_shortlist_compare.csv")
print(f"\nlayer {best_layer}:\n{compare.round(3).to_string()}")

# timing
t_short, t_score = [], []
for text in texts[:N_TIMED]:
    t0 = time.time()
    cand = top_k(cosine_to_all(vectors, text_state_by_layer(text)[best_layer].cpu().numpy()), idioms, K_TIMED)
    t1 = time.time()
    text_scores(text, cand)
    t2 = time.time()
    t_short.append(t1 - t0); t_score.append(t2 - t1)
per_idiom = np.mean(t_score) / K_TIMED
print(f"\nshortlist: {np.mean(t_short):.2f} s | scoring {K_TIMED} idioms: {np.mean(t_score):.1f} s"
      f" | full dictionary (estimated): {per_idiom * len(idioms):.0f} s"
      f" | speed-up ~{len(idioms) / K_TIMED:.0f}x")
if "B" in methods:
    t0 = time.time()
    for text in texts[:N_TIMED]:
        idioms_from_neighbours(ex_vectors @ encode([text])[0], ex_idioms, K_TIMED)
    print(f"option B search: {(time.time() - t0) / N_TIMED:.2f} s per text")
