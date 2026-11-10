"""streamlit_app.py — free live demo (Streamlit Community Cloud).

- Same steps as api/main.py:
  - shortlist B: 200 idioms of the most similar training sentences
  - Qwen likelihood + frequency prior on those 200
  - exact posterior over the 200
- Data files are not in git -> downloaded from a Hugging Face dataset repo.
- Research code unchanged."""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                  # research code reads data/... from here
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CHENGYU_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_REPO = "Fatinx48/chengyu-demo-data"
K = 200       # candidates scored
TOP = 8       # results shown

EXAMPLES = {
    "画已经很好了，他又多加了几笔，反而把画弄坏了。": "The painting was already good, but he added a few more strokes and ruined it.",
    "他在专家面前卖弄自己的一点知识。": "He shows off his little bit of knowledge in front of experts.",
    "他一直等着好运自己送上门，却什么也不做。": "He keeps waiting for good luck to come to him, without doing anything.",
    "他每天早起晚睡，非常努力地工作。": "He gets up early and goes to bed late every day, working very hard.",
    "他们俩说的话完全一样，好像事先商量好了。": "The two of them said exactly the same thing, as if they had agreed beforehand.",
    "他讲得很认真，可是听的人一点也不懂。": "He explained it carefully, but the listeners did not understand a thing.",
}


@st.cache_resource(show_spinner="Loading the models (first visit after a pause: 1–3 min)…")
def load():
    """Once per server start: data files + Qwen + example index."""
    import json

    from huggingface_hub import snapshot_download
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir=ROOT, allow_patterns="data/*")
    from chengyu.argmax import exact_posterior, text_scores  # loads Qwen
    from chengyu.neighbours import load_examples, neighbour_candidates
    from chengyu.prior import log_prior
    with open("data/idiom_en.json", encoding="utf-8") as f:
        english = json.load(f)
    return exact_posterior, text_scores, log_prior, neighbour_candidates, load_examples(), english


st.set_page_config(page_title="Chinese Idiom Inference", page_icon="📜")
st.title("Which idiom fits this sentence?")
st.write("Write a sentence in Chinese. A language model (Qwen2.5-0.5B) scores how well each "
         "candidate idiom explains it, a frequency prior is added, and Bayes' rule gives a posterior.")

choice = st.radio("Try an example (English translation shown):", list(EXAMPLES),
                  format_func=lambda s: f"{s}  —  {EXAMPLES[s]}", index=None)
text = st.text_area("Sentence", value=choice or "", height=80)

if st.button("Find the idiom", type="primary", disabled=not text.strip()):
    exact_posterior, text_scores, log_prior, neighbour_candidates, examples, english = load()
    with st.spinner(f"Scoring {K} candidate idioms on a free CPU…"):
        import time
        t0 = time.time()
        idioms = neighbour_candidates(text.strip(), *examples, k=K)
        likelihood = text_scores(text.strip(), idioms)
        h = {i: likelihood[i] + log_prior(i) for i in idioms}
        posterior = exact_posterior(h)
        ranked = sorted(h, key=h.get, reverse=True)[:TOP]
        seconds = time.time() - t0
    for n, i in enumerate(ranked, 1):
        info = english.get(i)
        meaning = f"*{info['pinyin']}* — {info['en']}" if info else "_no English entry in CC-CEDICT_"
        st.markdown(f"**{n:02d}. {i}** · {posterior[i] * 100:.1f}%  \n{meaning}")
        st.progress(min(1.0, posterior[i] / posterior[ranked[0]]))
        st.caption(f"log-likelihood {likelihood[i]:.1f} · log-prior {log_prior(i):.1f}")
    st.caption(f"{seconds:.0f} s · posterior normalized over these {len(idioms)} candidates")

st.divider()
st.caption("Demo shortcut: the 200 candidates are the idioms of the most similar training sentences "
           "(the correct idiom is among them for 63% of test sentences). The thesis results use the "
           "full dictionary of 31,114 idioms and a larger model. English meanings and pinyin: "
           "CC-CEDICT (CC BY-SA 4.0). "
           "[Code on GitHub](https://github.com/FatineKa/chinese-idiom-inference)")
