import json
from huggingface_hub import hf_hub_download

# le fichier original contient les réponses (groundTruth) 
path = hf_hub_download("thu-coai/chid", "original/train_data.txt", repo_type="dataset")
print("fichier :", path)

with open(path, encoding="utf-8") as f:
    rec = json.loads(f.readline())     # premier document

print("clés :", list(rec.keys()))
print("groundTruth :", rec.get("groundTruth"))
print("candidates  :", rec.get("candidates"))
print("realCount   :", rec.get("realCount"))
print("1er paragraphe :", rec["content"][0][:200], "…")