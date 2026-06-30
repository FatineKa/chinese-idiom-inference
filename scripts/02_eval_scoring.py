import json
from huggingface_hub import hf_hub_download
from chengyu.scoring import log_prob_total

N = 20   # nombre de documents à évaluer (augmente plus tard si tu as le temps)

path = hf_hub_download("thu-coai/chid", "original/train_data.txt",
                       repo_type="dataset")

def remplir(texte, gt, k, c):
    t = texte
    for j in range(len(gt)):
        t = t.replace(f"#idiom{j:06d}#", c if j == k else gt[j])
    return t

def evaluer(rec):
    texte, gt, cands = " ".join(rec["content"]), rec["groundTruth"], rec["candidates"]
    bons = 0
    for k in range(len(gt)):
        scores = {c: log_prob_total(remplir(texte, gt, k, c)) for c in cands[k]}
        if max(scores, key=scores.get) == gt[k]:
            bons += 1
    return bons, len(gt)

bons_tot, total_tot = 0, 0
with open(path, encoding="utf-8") as f:
    for i in range(N):
        rec = json.loads(f.readline())
        b, t = evaluer(rec)
        bons_tot += b; total_tot += t
        print(f"doc {i+1}/{N} : {b}/{t}   "
              f"(cumul {bons_tot}/{total_tot} = {bons_tot/total_tot:.1%})")

print(f"\nJustesse globale : {bons_tot}/{total_tot} = {bons_tot/total_tot:.1%}")