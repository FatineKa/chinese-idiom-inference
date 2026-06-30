import json
from datasets import load_dataset

ds = load_dataset("thu-coai/chid")
print("Splits :", {k: len(v) for k, v in ds.items()})

rec = json.loads(ds["train"][0]["text"])     # parser la chaîne JSON

print("\nClés disponibles :", list(rec.keys()))
print("Nombre de candidats :", len(rec["candidates"]))
print("Candidats :", rec["candidates"])
print("Nombre de paragraphes :", len(rec["content"]))
print("\nPremier paragraphe :\n", rec["content"][0][:300], "…")