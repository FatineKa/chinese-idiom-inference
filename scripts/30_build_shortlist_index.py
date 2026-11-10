"""30_build_shortlist_index.py — idiom vectors for the demo shortlist.

- Output: data/shortlist_index.npz
- CPU is fine (table lookups, no forward pass).
- Use the SAME model as the demo:
  CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct python scripts/30_build_shortlist_index.py"""
import time

from chengyu.evaluation import load_dictionary
from chengyu.shortlist import INDEX_PATH, build_index

dictionary, _ = load_dictionary()
idioms = sorted(dictionary)   # fixed order
t0 = time.time()
build_index(idioms)
print(f"saved {len(idioms)} idiom vectors to {INDEX_PATH} in {time.time() - t0:.0f} s")
