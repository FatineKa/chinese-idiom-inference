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

## Code architecture

How a text turns into a predicted idiom, file by file:

```
data/raw/cip/idioms.txt, train.csv
            |
            v
scripts/00_build_freq.py
            |
            v
   data/idiom_freq.json
            |
            v
        prior.py                      scoring.py
   (idiom usage counts                (loads Qwen,
    -> prior probability)              scores text given idiom)
            |                                |
            +----------------+---------------+
                             |
                             v
                        argmax.py
             (scores every idiom in the dictionary,
              likelihood x prior, picks the best one)
                             |
            +----------------------------------+
            |                                  |
  geometry.py, representation.py               |
  (idiom embedding, how it shifts               |
   once Qwen reads the text)                   |
            |                                  |
            v                                  |
         mcmc.py  <------------------------------
   (samples idioms instead of scoring
    all 31,114, guided by the shift signal)
            |
            v
      api/main.py
  (FastAPI endpoint: text in, ranked idioms out)
```

`src/chengyu/` holds the code, split into small single-purpose files:

- `scoring.py`: loads Qwen (the AI model) and the tokenizer. Every other
  file that needs the model imports it from here, so it is loaded only
  once.
- `prior.py`: turns idiom usage counts into the prior probability
  p(idiom). Reads `data/idiom_freq.json`, built by
  `scripts/00_build_freq.py`.
- `evaluation.py`: loads the idiom dictionary and cleans up text so it
  matches what Qwen expects.
- `argmax.py`: scores every idiom in the dictionary for a given text and
  picks the best one directly, no sampling. Used as the exact reference
  answer.
- `mcmc.py`: the sampling loop (MCMC) that guesses its way to good idioms
  instead of scoring all 31,114 of them.
- `geometry.py` and `representation.py`: turn an idiom into a vector of
  numbers (an embedding) and measure how that vector shifts once Qwen
  reads it right after the text. This shift is the signal used to guide
  the sampler in `mcmc.py`.

Other folders:
- `scripts/`: the commands to run, numbered in the order they are meant
  to run (see below).
- `api/main.py`: a small FastAPI web endpoint that wraps `argmax.py` and
  `prior.py` so a text can be sent in and a ranked list of idioms comes
  back.
- `data/`: raw and cleaned data, not stored in git.
- `results/`: figures, tables, output CSVs produced by the scripts.
- `tests/`: run with `CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest`.

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

## Results

Goal: given a plain-language sentence, find which Chinese idiom (out of
31,114 candidates) it is describing. Two scores are combined: how well
an AI model (Qwen) thinks the idiom explains the sentence, and how
common that idiom is in real usage.

Some terms used below:
- Top-1: the single best guess is correct.
- Top-10: the correct answer is somewhere in the 10 best guesses.
- Pairwise rate: shown two idioms, how often the method correctly picks
  the better match.
- TVD (total variation distance): a score from 0 to 1 for how close a
  fast, approximate method's answer is to the slow, exact answer. 0
  means identical, 1 means completely different.
- Mode-MAP: the fast method's single most-visited answer versus the
  slow method's single best answer. This checks whether the shortcut
  and the exact calculation agree on the top pick.
- MCMC (Markov Chain Monte Carlo): a sampling method that guesses its
  way toward good answers instead of checking all 31,114 candidates
  one by one, which would be slow.

Full-dictionary ranking, 50 test texts:
- Score from the AI model alone: top-1 correct 6%, top-10 correct 14%
- Score from the AI model plus how common the idiom is: top-1 correct
  8%, top-10 correct 24%
- Adding "how common the idiom is" moves the correct answer closer to
  the top.

Representation study: testing whether the AI model's internal state
changes in a useful way when it reads an idiom right after the matching
text, across different layers and distance measures.
- Best signal: Euclidean distance (a standard way to measure how far
  apart two points are) at layer 23 of the model. A smaller distance
  means a better match.
- Picks the correct idiom over a wrong one 72% of the time (pairwise
  rate)
- Picks the single best idiom correctly 21.5% of the time out of 21
  candidates (random guessing gives about 4.8%)

Using that signal to guide the MCMC sampler, tested on 50 texts the
signal was not trained on, 10,000 guesses each, averaged over 5 runs:
- Plain sampler (no guidance): TVD 0.8243, top pick matches the exact
  answer 17.2% of the time (mode-MAP)
- Guided sampler: TVD 0.7795, top pick matches the exact answer 22.8%
  of the time
- The guided sampler does better but is still far from a perfect match.

LLM judge check: asking the AI model whether a predicted idiom fits the
sentence, even if it is not the exact idiom recorded in the dataset.
- Exact match with the dataset answer: 8% of 50 texts
- Judge says the predicted idiom still fits the sentence: 78% of 50
  texts
- Many "wrong" predictions are actually reasonable alternatives, not
  real mistakes.
- Judge approval of the MCMC sampler's top pick, on the same 50 texts:
  61.6% for the plain sampler, 66.4% for the guided sampler.
