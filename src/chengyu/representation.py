"""representation.py — signal de modification (statique vs contextualisée),
couche par couche. Un seul forward pass calcule toutes les couches à la
fois (output_hidden_states ne coûte rien de plus)."""
import torch

from chengyu.geometry import plongement_idiome
from chengyu.scoring import _device, _model, _tok


@torch.no_grad()
def etats_dernier_token(prompt: str) -> torch.Tensor:
    """État à la dernière position de `prompt`, à chaque couche
    (0 = embeddings, n_couches = dernière). Forme : (n_couches+1, dim)."""
    ids = _tok(prompt, return_tensors="pt").input_ids.to(_device)
    sorties = _model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in sorties.hidden_states]).float()


def modification_par_couche(idiome: str, texte: str) -> list:
    """Δ_l(i,t) pour chaque couche l : distance statique/contextualisée.
    Prompt inversé (texte puis idiome) pour respecter le masquage causal :
    l'idiome doit venir après le texte pour que sa représentation en soit
    influencée (modèle causal — voir le chapitre, section "The causal
    constraint")."""
    prompt = f"这句话「{texte}」，可以概括为成语「{idiome}"
    e_stat = torch.from_numpy(plongement_idiome(idiome))
    etats = etats_dernier_token(prompt)
    return [1 - torch.nn.functional.cosine_similarity(e_stat, e, dim=0).item()
            for e in etats]


def etat_texte_par_couche(texte: str) -> torch.Tensor:
    """État du texte seul (sans idiome), à chaque couche, à la position
    juste avant l'idiome. Un seul forward pass, indépendant du nombre
    d'idiomes — utilisé par le proposeur efficace (mcmc.proposeur_par_texte)."""
    prompt = f"这句话「{texte}」，可以概括为成语「"
    return etats_dernier_token(prompt)
