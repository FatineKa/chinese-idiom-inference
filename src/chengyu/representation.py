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


@torch.no_grad()
def etats_dernier_token_lot(prompts: list) -> torch.Tensor:
    """Comme etats_dernier_token, mais pour tout un lot de prompts en un seul
    forward pass -- essentiel sur GPU (beaucoup de petits appels successifs
    sous-utilise le parallelisme du GPU ; un seul appel sur un lot l'exploite).

    Padding a GAUCHE (pas a droite comme argmax.py) : il nous faut la DERNIERE
    position (-1) de chaque sequence, qui ne correspond au dernier token reel
    que si les tokens de remplissage sont ajoutes avant, pas apres -- contrairement
    a argmax.py, qui ne lit que les positions du texte (jamais la derniere
    position absolue) et peut donc se permettre un padding a droite.

    position_ids est calcule explicitement a partir du masque d'attention
    (plutot que laisse au defaut du modele) : sous RoPE, une position mal
    decalee a cause du padding changerait silencieusement le sens des
    positions causales pour les tokens reels -- ne pas se fier a un defaut
    implicite ici."""
    pad = _tok.pad_token_id or _tok.eos_token_id
    ids_liste = [_tok(p).input_ids for p in prompts]
    L = max(len(ids) for ids in ids_liste)
    lot = len(prompts)
    batch = torch.full((lot, L), pad, dtype=torch.long, device=_device)
    masque = torch.zeros((lot, L), dtype=torch.long, device=_device)
    for j, ids in enumerate(ids_liste):
        batch[j, L - len(ids):] = torch.tensor(ids, device=_device)
        masque[j, L - len(ids):] = 1
    position_ids = (masque.cumsum(-1) - 1).clamp(min=0)
    sorties = _model(batch, attention_mask=masque, position_ids=position_ids,
                      output_hidden_states=True)
    return torch.stack([h[:, -1] for h in sorties.hidden_states], dim=1).float()


def modification_par_couche_lot(paires: list) -> list:
    """Comme modification_par_couche, mais pour une liste de (idiome, texte) a
    la fois -- un seul forward pass pour tout le lot plutot qu'un par paire.
    Renvoie une liste (une entree par paire) de listes de Delta_l."""
    prompts = [f"这句话「{texte}」，可以概括为成语「{idiome}" for idiome, texte in paires]
    etats = etats_dernier_token_lot(prompts)   # (lot, n_couches, dim)
    resultats = []
    for k, (idiome, _texte) in enumerate(paires):
        e_stat = torch.from_numpy(plongement_idiome(idiome))
        resultats.append([
            1 - torch.nn.functional.cosine_similarity(e_stat, etats[k, l], dim=0).item()
            for l in range(etats.shape[1])
        ])
    return resultats
