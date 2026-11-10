# Deploying the live demo

> **Free option (used):** Hugging Face Docker Spaces now need a PRO subscription.
> The free demo runs on Streamlit Community Cloud: `demo/streamlit_app.py`
> (main file), `demo/requirements.txt`, Python 3.11. Its data files are in a
> Hugging Face dataset repo (`Fatinx48/chengyu-demo-data`), downloaded at start.
> The Docker steps below still work on any server that runs Docker.

The demo is a FastAPI web service (`api/main.py`) with a small web page
(`web/index.html`), packaged with Docker (`Dockerfile`) and hosted for free on
Hugging Face Spaces. The research code in `src/chengyu/` is unchanged; the demo
only adds a fast shortlist step (`src/chengyu/shortlist.py` = option A,
`src/chengyu/neighbours.py` = option B).

## How a demo request works

```
sentence ──► B: similar training sentences (bge-small-zh) -> their idioms
         ──► 200 candidates  (A = idiom vectors: only used if B's file is missing)
         ──► Qwen likelihood + frequency prior on those 200
         ──► exact posterior over the 200 ──► top 8 shown on the page
```

## 0. Before publishing: data licence

The demo needs two data files that are not in this repository:
`data/raw/cip/idioms.txt` (the CIP idiom dictionary) and `data/idiom_freq.json`
(built by `scripts/00_build_freq.py`). Check that the CIP dataset licence allows
redistributing the idiom list before putting it in a public Space.

## 1. Measure the shortcut (Colab, GPU not required but faster)

```bash
pip install -e . && pip install -r requirements.txt
export CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python scripts/30_build_shortlist_index.py     # idiom vectors -> data/shortlist_index.npz
python scripts/31_shortlist_recall.py          # recall@K per layer + timing
python scripts/32_build_translations.py        # English meanings + pinyin -> data/idiom_en.json
python scripts/33_build_examples_index.py      # option B: training sentences -> data/examples_index.npz
python scripts/31_shortlist_recall.py          # run again after 33: compares A, B, A+B
```

Script 31 also prints how many test sentences appear word for word in
`train.csv`. If that number is large, B's recall is too optimistic.

Option B can only suggest idioms that appear in the training sentences. If the
comparison shows B does not help, delete `data/examples_index.npz`: the demo then
uses A only (and change the "Demo shortcut" sentence in `web/index.html`).

Script 32 prints how many idioms CC-CEDICT covers. Idioms it does not cover are
shown on the demo page without a translation. CC-CEDICT is CC BY-SA 4.0: the demo
page credits it.

Read the table: pick the layer with the best `recall@200`, and put it in the
Dockerfile (`CHENGYU_SHORTLIST_LAYER`). Report the recall and the speed-up in the
README: they tell visitors how much accuracy the demo trades for speed.

## 2. Try it locally (optional, needs Docker and ~4 GB of free RAM)

```bash
docker build -t chengyu-demo .
docker run -p 7860:7860 chengyu-demo
# open http://localhost:7860
```

## 3. Publish on Hugging Face Spaces

1. Create a free account on https://huggingface.co and a new Space:
   **SDK = Docker**, hardware = **CPU basic (free)**.
2. Clone the Space, copy this repository's files into it, and use
   `deploy/space-README.md` as the Space's `README.md` (Spaces read their
   settings from it).
3. Add the data files (they are git-ignored here, so force-add them):
   ```bash
   git lfs install && git lfs track "*.npz"      # the example index is ~100 MB
   git add -f data/raw/cip/idioms.txt data/idiom_freq.json data/examples_index.npz .gitattributes
   git add . && git commit -m "Live demo" && git push
   ```
4. The Space builds the image (about 10 minutes the first time: it downloads the
   model and builds the index), then the demo is live at
   `https://huggingface.co/spaces/<user>/<space-name>`.

Free Spaces go to sleep after a period without visitors; the first visit after
that waits for the Space to restart.

## 4. Show it in the portfolio

Embed the Space page (the `https://<user>-<space-name>.hf.space` address) in an
iframe, like the other live demos.

## Measured (Colab T4, Qwen2.5-0.5B, 200 in-domain test sentences)

| correct idiom in the 200 candidates | A | B | A+B |
|---|---|---|---|
| recall@200 | 13% | 63% | 55% |

- 0 test sentences appear word for word in train.csv.
- Search: B 0.02 s. Scoring 200 idioms: 4.5 s on GPU (CPU is slower).
- Full dictionary, estimated: 701 s -> ~156x faster.
- B can only suggest the 7,234 idioms seen in train.csv.
- English meanings: 6,979 / 31,114 idioms (22%) in CC-CEDICT.
