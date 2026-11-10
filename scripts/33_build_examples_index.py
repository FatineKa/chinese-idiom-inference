"""33_build_examples_index.py — example-sentence index for option B.

- Input: data/raw/cip/train.csv (src, dst)
- Output: data/examples_index.npz (~100 MB)
- GPU (Colab): ~1-2 min. CPU: much slower.
- Run: python scripts/33_build_examples_index.py"""
import time

import pandas as pd

from chengyu.evaluation import find_idiom, load_dictionary, normalize
from chengyu.neighbours import EXAMPLES_PATH, build_examples

dictionary, lengths = load_dictionary()
df = pd.read_csv("data/raw/cip/train.csv").drop_duplicates(subset=["src", "dst"])

texts, idioms = [], []
for src, dst in zip(df["src"], df["dst"]):
    target = find_idiom(src, dst, dictionary, lengths)
    if target is not None:
        texts.append(normalize(dst)); idioms.append(target)
print(f"{len(texts)} example sentences, {len(set(idioms))} different idioms")

t0 = time.time()
build_examples(texts, idioms)
print(f"saved to {EXAMPLES_PATH} in {time.time() - t0:.0f} s")
