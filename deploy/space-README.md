---
title: Chinese Idiom Inference
emoji: 📜
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Bayesian Chinese idiom retrieval with an LLM likelihood
---

Live demo of [chinese-idiom-inference](https://github.com/FatineKa/chinese-idiom-inference):
p(idiom | text) ∝ p(text | idiom) · p(idiom), with Qwen2.5-0.5B as the likelihood
and idiom frequency as the prior. The demo scores a shortlist of 200 idioms;
see the repository for the full-dictionary research results.
