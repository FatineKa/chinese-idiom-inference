"""argmax.py — classement exact des idiomes par force brute.
Remplace le MCMC : pour l'argmax, la constante Z est inutile
(elle est la même pour tous les idiomes), on score tout le
dictionnaire et on prend le max."""
import torch
from chengyu.scoring import _tok, _model
from chengyu.prior import log_prior

@torch.no_grad()
def scores_texte(texte: str, idiomes: list, batch_size: int = 16) -> dict:
    """log p(texte | idiome) pour chaque idiome, par lots.
    Un seul forward par prompt+texte : on somme les log-probas des
    tokens du texte seulement — équivalent à la soustraction
    log p(prompt+texte) - log p(prompt) par la règle de chaînage,
    mais deux fois moins cher."""
    ids_texte = _tok(texte, add_special_tokens=False).input_ids
    pad = _tok.pad_token_id or _tok.eos_token_id
    resultats = {}
    for debut in range(0, len(idiomes), batch_size):
        lot = idiomes[debut:debut + batch_size]
        seqs, lp = [], []          # séquences d'ids, longueurs de prompt
        for idiome in lot:
            ids_prompt = _tok(f"成语「{idiome}」概括了这句话：").input_ids
            lp.append(len(ids_prompt))
            seqs.append(ids_prompt + ids_texte)
        # padding à droite : sans effet sur les positions utiles (modèle causal)
        L = max(len(s) for s in seqs)
        batch  = torch.full((len(lot), L), pad, dtype=torch.long)
        masque = torch.zeros((len(lot), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            batch[j, :len(s)] = torch.tensor(s)
            masque[j, :len(s)] = 1
        logp = torch.log_softmax(_model(batch, attention_mask=masque).logits, dim=-1)
        for j, idiome in enumerate(lot):
            total = 0.0
            for k in range(lp[j], len(seqs[j])):   # tokens du texte seulement
                total += logp[j, k - 1, seqs[j][k]].item()
            resultats[idiome] = total
    return resultats

def classer(texte, idiomes, k=10, avec_prior=True, batch_size=16):
    """Top-k exact : argmax de log p(t|i) (+ log p(i))."""
    s = scores_texte(texte, idiomes, batch_size)
    if avec_prior:
        s = {i: v + log_prior(i) for i, v in s.items()}
    return sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:k]