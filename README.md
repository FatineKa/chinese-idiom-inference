# Mémoire project: Chinese idiom (chengyu) inference

Infer the chengyu (成语) that summarizes a text: `p(idiom | text) =
p(text | idiom) · p(idiom) / Z`. Qwen gives the likelihood, a frequency
count gives the prior.

**Two pillars:** the geometric study of embeddings, and model selection
(learning theory).

## Installation

```bash
conda activate projet-memoire        # or: source .venv/bin/activate
pip install -e .
```

## Structure

- `src/chengyu/`: the code (scoring, prior, MCMC, embeddings, evaluation)
- `scripts/`: the commands to run (numbered — see below)
- `api/main.py`: small FastAPI demo endpoint
- `data/`: raw and cleaned data (not versioned)
- `results/`: figures, tables, output CSVs
- `tests/`: run with `CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest`

## Data

CIP dataset, in `data/raw/cip/`: `idioms.txt` (31,114-idiom dictionary),
`train.csv` (95,560 pairs), plus held-out `in_domain`/`out_domain` splits.

## Running the pipeline

```bash
python scripts/00_build_freq.py              # 1. frequency prior

python scripts/01_test_scoring.py             # 2. sanity checks
python scripts/02_verify_mcmc.py

python scripts/03_cip_eval.py                 # 3. small diagnostics
python scripts/04_argmax_eval.py

python scripts/17_delta_controlled_test.py    # 4. pick metric/layer/direction
CHENGYU_FINAL_METRIC=euc CHENGYU_FINAL_LAYER=23 CHENGYU_FINAL_DIRECTION=smaller \
    python scripts/18_delta_final_test.py     #    confirm it, frozen

python scripts/20_delta_proposal_comparison.py  # 5. informed vs. uniform MCMC
python scripts/21_llm_judge_verification.py     # 6. LLM-judge check on top-1
```

Scripts 09–13 and 15 are archived exploratory work, not part of the
current pipeline.

## Status

**Working and validated:** Bayesian scoring, exact full-dictionary
ranking, and the MCMC sampler. The representation study: Euclidean
distance at layer 23 (smaller = better fit) predicts idiom-text fit,
confirmed on a held-out split.

**Implemented, awaiting real GPU results:** a representation-informed MCMC
proposal (script 20) and an LLM-as-judge check on top-1 predictions
(script 21).
