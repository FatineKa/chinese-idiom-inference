"""00_build_freq.py — counts idiom frequency in CIP (train.csv) and
   writes data/idiom_freq.json. Run from the ~/projet-memoire root."""

import json
from collections import Counter
import pandas as pd
from chengyu.evaluation import load_dictionary, normalize, idioms_present

dictionary, lengths = load_dictionary()
df = pd.read_csv("data/raw/cip/train.csv")

# Prior = usage frequency: count every idiom seen in the src column.
# (more data than counting only the target idiom, so a more robust prior)
# Note: idioms_present returns a SET per row, so an idiom appearing twice
# in the same src is counted once for that row -- this is document
# frequency (how many rows an idiom appears in), not raw occurrence count.
count = Counter()
for src in df["src"]:
    for idiom in idioms_present(normalize(src), dictionary, lengths):
        count[idiom] += 1

assert set(count).issubset(dictionary), "counted an idiom outside the dictionary"

with open("data/idiom_freq.json", "w", encoding="utf-8") as f:
    json.dump(dict(count), f, ensure_ascii=False)

print("idioms counted:", len(count), "| top:", count.most_common(5))
