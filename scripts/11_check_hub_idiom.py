"""11_check_hub_idiom.py — is a candidate idiom's dominance in the informed
proposer (script 10) a real per-text signal, or a hub artifact?

Two independent checks, for two different hub mechanisms:

Part 1 (embedding space): cosine similarity is scale-invariant (mcmc.py's
text_proposer divides by both norms), so an unusually large embedding norm
cannot by itself explain why an idiom is disproportionately favored -- that
was the wrong diagnostic. The actual rogue-dimension mechanism (Timkey &
van Schijndel, 2021) is directional: a few high-variance dimensions that
most embeddings partially share, inflating cosine similarity between
otherwise-unrelated vectors. A real hub therefore has unusually high average
cosine similarity to a broad, random sample of other idioms, and to the
centroid of the whole embedding cloud -- not necessarily a large norm.
Cheap: static embeddings are embedding-table lookups, no forward pass.

Part 2 (likelihood space): a separate hypothesis for why the SAME idioms
keep winning the exact posterior across unrelated texts in script 10 --
some idioms may be generic/broadly-applicable enough that Qwen assigns them
decent score_summary(text, idiom) across many different texts, independent
of actual fit, rather than sharply distinguishing correct from incorrect.
Checked by comparing mean and spread of score_summary across a random text
sample -- unlike Part 1, this needs a real forward pass per (text, idiom)
pair, since score_summary depends on inserting the candidate into the prompt.
"""
import random

import numpy as np
import pandas as pd

from chengyu.evaluation import load_dictionary, normalize
from chengyu.geometry import embeddings
from chengyu.scoring import score_summary

SEED = 0
N_SAMPLE = 1000        # random idioms to compare against, per candidate
CANDIDATES = ["肆无忌惮", "无动于衷"]   # the two idioms that kept dominating
                        # in script 10's runs -- checked together so the
                        # "most favored idiom" isn't judged against nothing


def cos(u, v):
    return (u @ v.T) / (np.linalg.norm(u, axis=-1, keepdims=True)
                         * np.linalg.norm(v, axis=-1))


rng = random.Random(SEED)
dictionary, _ = load_dictionary()
idiom_list = sorted(dictionary)
print(f"dictionary size: {len(idiom_list)}")

print("computing static embeddings for the full dictionary "
      "(embedding-table lookups, no forward pass)...")
X = embeddings(idiom_list)                     # (N, dim)
norms = np.linalg.norm(X, axis=1)
centroid = X.mean(axis=0)

index = {idiom: k for k, idiom in enumerate(idiom_list)}
sample_idx = rng.sample(range(len(idiom_list)), N_SAMPLE)
X_sample = X[sample_idx]

print(f"\nnorm percentiles over the full dictionary: "
      f"p50={np.percentile(norms, 50):.3f} "
      f"p90={np.percentile(norms, 90):.3f} "
      f"p99={np.percentile(norms, 99):.3f} "
      f"max={norms.max():.3f}")

# baseline: a few random idioms, for comparison -- is CANDIDATES actually
# unusual, or does every idiom look like this?
baseline = rng.sample(idiom_list, 5)

print(f"\n{'idiom':<10} {'norm':>8} {'norm pct':>9} "
      f"{'mean cos to sample':>20} {'cos to centroid':>17}")
for idiom in CANDIDATES + baseline:
    k = index[idiom]
    v = X[k]
    norm_pct = (norms < norms[k]).mean() * 100
    mean_cos_sample = cos(v[None, :], X_sample).mean()
    cos_centroid = cos(v[None, :], centroid[None, :]).item()
    tag = " <-- candidate" if idiom in CANDIDATES else ""
    print(f"{idiom:<10} {norms[k]:>8.3f} {norm_pct:>8.1f}% "
          f"{mean_cos_sample:>20.4f} {cos_centroid:>17.4f}{tag}")

print(f"""
Reading this:
- 'norm' / 'norm pct': included for completeness, but norm does NOT explain
  cosine-similarity dominance (cosine is scale-invariant) -- don't over-read this column.
- 'mean cos to sample': average cosine similarity to {N_SAMPLE} random idioms.
  If a candidate's value is far above the baseline idioms', it is unusually
  similar to *everything* -- a hub, not a text-specific match.
- 'cos to centroid': cosine similarity to the mean of all {len(idiom_list)}
  embeddings. A hub aligns strongly with this shared direction; an idiom with
  a distinctive, specific embedding should not.
""")

# --- Part 2: likelihood-space hub check ------------------------------------
N_TEXTS_LIK = 30        # random texts scored per idiom -- each is one real
                         # forward pass (unlike Part 1), so kept modest

print(f"--- likelihood-space hub check ({N_TEXTS_LIK} random texts) ---")
df = pd.read_csv("data/raw/cip/train.csv")
sample_rows = rng.sample(range(len(df)), N_TEXTS_LIK)
texts = [normalize(df["dst"].iloc[k]) for k in sample_rows]
print("scoring each candidate against every sampled text "
      f"({len(CANDIDATES + baseline)} idioms x {N_TEXTS_LIK} texts = "
      f"{len(CANDIDATES + baseline) * N_TEXTS_LIK} forward passes)...")

print(f"\n{'idiom':<10} {'mean loglik':>12} {'std':>8} {'min':>8} {'max':>8}")
for idiom in CANDIDATES + baseline:
    scores = np.array([score_summary(t, idiom) for t in texts])
    tag = " <-- candidate" if idiom in CANDIDATES else ""
    print(f"{idiom:<10} {scores.mean():>12.2f} {scores.std():>8.2f} "
          f"{scores.min():>8.2f} {scores.max():>8.2f}{tag}")

print(f"""
Reading this:
- These {N_TEXTS_LIK} texts are random, not chosen for fit with any
  candidate -- this measures baseline likelihood, independent of whether an
  idiom is ever actually correct for a given text.
- A hub-like idiom should have a HIGHER mean loglik than the baseline
  idioms, AND a SMALLER std -- consistently decent across unrelated content,
  rather than being a strong match for some texts and a poor one for others.
  If a candidate's mean is unremarkable next to the baseline, this
  hypothesis is not what's driving script 10's results -- look elsewhere.
""")
