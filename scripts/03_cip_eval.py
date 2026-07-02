import csv, os, random
from chengyu.scoring import score_resume

random.seed(0)
CIP = "data/raw/cip"
N, K = 50, 7            # 50 exemples, 7 candidats

with open(os.path.join(CIP, "idioms.txt"), encoding="utf-8") as f:
    idiomes = [l.strip() for l in f if l.strip()]
idiom_set = set(idiomes)

def trouver_idiome(phrase):
    s = phrase.replace(" ", "")
    for i in range(len(s) - 3):
        if s[i:i+4] in idiom_set:
            return s[i:i+4]
    return None

bons, total = 0, 0
with open(os.path.join(CIP, "train.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if total >= N:
            break
        bon = trouver_idiome(row["src"])
        if bon is None:
            continue
        texte = row["dst"].replace(" ", "")
        candidats = list({bon, *random.sample(idiomes, K - 1)})
        scores = {c: score_resume(texte, c) for c in candidats}
        pred = max(scores, key=scores.get)
        total += 1; bons += (pred == bon)
        print(f"{total}: {pred} vs {bon} {'OK' if pred==bon else 'X'} ({bons}/{total})")

print(f"\nJustesse texte->idiome : {bons}/{total} = {bons/total:.1%}")
