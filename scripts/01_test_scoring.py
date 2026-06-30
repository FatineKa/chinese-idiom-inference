import json
from huggingface_hub import hf_hub_download
from chengyu.scoring import log_prob_total

path = hf_hub_download("thu-coai/chid", "original/train_data.txt", repo_type="dataset")

def remplir(texte, gt, k, c):
    """Remplit le trou k par c, les autres trous par leur bonne réponse."""
    t = texte
    for j in range(len(gt)):
        t = t.replace(f"#idiom{j:06d}#", c if j == k else gt[j])
    return t

def evaluer(rec):
    texte, gt, cands = " ".join(rec["content"]), rec["groundTruth"], rec["candidates"]
    bons = 0
    for k in range(len(gt)):
        scores = {c: log_prob_total(remplir(texte, gt, k, c)) for c in cands[k]}
        pred = max(scores, key=scores.get)
        ok = (pred == gt[k]); bons += ok
        print(f"trou {k} : prédit = {pred}  | correct = {gt[k]}  {'✓' if ok else '✗'}")
    return bons, len(gt)

with open(path, encoding="utf-8") as f:
    rec = json.loads(f.readline())     # premier document

bons, total = evaluer(rec)
print(f"\nJustesse : {bons}/{total}")