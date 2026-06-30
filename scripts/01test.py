from datasets import load_dataset
ds = load_dataset("thu-coai/chid")
print(ds)              # les splits disponibles
print(ds["train"][0])  # UN exemple : regarde les clés (passage, candidats, réponse)