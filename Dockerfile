# Live demo image: Hugging Face Spaces (Docker, free CPU).
# - local test: docker build -t chengyu-demo . && docker run -p 7860:7860 chengyu-demo
# - needs data/raw/cip/idioms.txt + data/idiom_freq.json (see DEPLOY.md)
# - optional: data/examples_index.npz (option B, built on Colab by script 33)
FROM python:3.11-slim

# Spaces run as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# torch: CPU-only build (smaller)
COPY requirements-demo.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-demo.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --no-deps -e .

USER user
ENV CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    CHENGYU_SHORTLIST_LAYER=24 \
    HF_HOME=/app/.cache/huggingface

# done at build time -> fast start
# - download the model
# - build the shortlist index
# - build English meanings (CC-CEDICT)
# - download the sentence model (option B)
RUN python -c "import chengyu.scoring; from chengyu.neighbours import encode; encode(['热'])" \
 && python scripts/30_build_shortlist_index.py \
 && python scripts/32_build_translations.py

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
