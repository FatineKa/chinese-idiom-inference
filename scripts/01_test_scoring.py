import pandas as pd
from chengyu.evaluation import load_dictionary, normalize, find_idiom
from chengyu.scoring import score_summary

df = pd.read_csv("data/raw/cip/train.csv")
dictionary, lengths = load_dictionary()

# first "clean" row of the corpus
for src, dst in zip(df["src"], df["dst"]):
    target = find_idiom(src, dst, dictionary, lengths)
    if target:
        break
text = normalize(dst)

print("text  :", text)
print("target:", target)
print("score(target)      :", score_summary(text, target))
print("score(off-topic)   :", score_summary(text, "无论如何"))
