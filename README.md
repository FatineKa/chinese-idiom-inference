# Mémoire project: Chinese idiom (chengyu) inference

Infer the chengyu (成语) that summarizes a text, framed as Bayesian
inference `p(idiom | text) = p(text | idiom) · p(idiom) / Z`, where Qwen
provides the likelihood and the prior comes from idiom usage frequencies.

**Two pillars:** the geometric study of embeddings, and model selection
(learning theory).

## Installation

```bash
conda activate projet-memoire        # or: source .venv/bin/activate
pip install -e .
```

## Structure

- `src/chengyu/`: the code (one file per task):
  `scoring.py`, `prior.py`, `geometry.py`, `mcmc.py`, `evaluation.py`,
  `representation.py`, `argmax.py`
- `scripts/`: the commands to run (numbered)
- `data/`: raw and cleaned data (not versioned)
- `results/`: outputs (figures, tables)
- `config.yaml`: settings (model, K, sigma, seed)

## Data

ChID (text → correct idiom):
`from datasets import load_dataset; load_dataset("thu-coai/chid")`

## Getting started

```bash
python scripts/01_test_scoring.py    # checks that the correct idiom wins
```

## Status

Bayesian scoring pipeline, exact argmax ranking, and the MCMC sampler are
working. Current focus: the classification study (script 09) validating
the per-layer Delta_l signal used to build an informed MCMC proposer.
