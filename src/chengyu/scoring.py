import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # deux outils de Hugging Face : AutoTokenizer (qui transforme le texte en nombres) et AutoModelForCausalLM (qui charge un modèle de langue « causal », c.-à-d. qui prédit le mot suivant

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_tok = AutoTokenizer.from_pretrained(MODEL) # Charge le tokenizer du modèle
_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32) # charge le modèle en mémoire
_model.eval() # met le modèle en mode evaluation

@torch.no_grad() #  les gradients servent à apprendre ; comme on ne fait qu’évaluer, les désactiver rend le code plus rapide et moins gourmand en mémoire
def log_prob_total(texte: str) -> float: # elle prend un texte et renvoie son score 
    """Somme des log-probabilités que Qwen attribue au texte."""
    ids = _tok(texte, return_tensors="pt").input_ids # transforme le texte en nombres (ids) et les met dans un tenseur PyTorch
    logp = torch.log_softmax(_model(ids).logits[0], dim=-1) # Qwen prédit le log-probabilité de chaque mot : _model(ids) fait passer les nombres dans le modèle, c'est une matrice (longueur du texte x vocabulaire) ; log_softmax normalise les scores pour qu’ils soient des log-probabilités, dim =-1 signifie qu’on normalise sur la dimension du vocabulaire
    total = 0.0
    for k in range(1, ids.shape[1]): # on parcourt les tokens du texte (de 1 à la longueur du texte) ; on commence à 1 car le premier token n’a pas de prédiction précédente
        total += logp[k - 1, ids[0, k]].item() # la log probabilité du token k est dans la ligne k-1 (la prédiction précédente) et la colonne correspondant à l’id du token k ; on ajoute cette log-probabilité au total, item convertit le tenseur PyTorch en float Python, total += : additionne car la probabilité d’une séquence est le produit des probabilités de chaque token, et le log d’un produit est la somme des logs
    return total # renvoie la somme des log-probabilités, qui est le score du texte selon Qwen

def score_resume(texte, idiome):
    """Score de : cet idiome résume-t-il ce texte ?"""
    prompt = f"成语「{idiome}」概括了这句话："      # "L'idiome X résume cette phrase :"
    return log_prob_total(prompt + texte) - log_prob_total(prompt)


""" Suivons un texte tout au long de log_prob_total

Prenons un exemple simple (en français, pour illustrerla mécanique) : texte = "le chat dort"


1. Texte → tokens (nombres). Le tokenizer découpe et numérote : "le chat dort" −→ [ 12, 88, 305 ] 


2. Nombres → modèle → logits. Le modèle lit ces nombres et produit, à chaque position, un score brut pour chaque mot possible. C’est une grande matrice : (3 positions) × (taille du vocabulaire, ∼150 000)


3. Logits → log-probabilités. log_softmax transforme chaque ligne en probabilités (qui somment à 1), puis en logarithme


4. On lit la log-probabilité du vrai mot suivant. À la position 0 (« le »), le modèle prédit le mot 1 (« chat ») : on lit log p(chat | le), disons −0,7. À la position 1 (« le chat »), on lit log p(dort | le chat), disons −1,2.


5. On additionne. Score = −0,7 + (−1,2) = −1,9."""


