# syntax=docker/dockerfile:1

# =============================================================================
# COEUR-Score — Artifact Evaluation image (GPU/CUDA enabled)
# -----------------------------------------------------------------------------
# Base: CUDA 12.8 + cuDNN runtime on Ubuntu 24.04 (ships Python 3.12, the same
# interpreter requirements-linux.txt was frozen against).
# The code automatically falls back to CPU when no GPU is visible, so this
# image also runs on CPU-only hosts (just slower). On a GPU host run it with
# `--gpus all` and the NVIDIA Container Toolkit installed.
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# --- Environment ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Caches for pre-downloaded models / corpora so the container runs offline.
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache/sentence-transformers \
    # Disable the Xet CAS transfer backend: it is unreliable in build/CI
    # environments (CAS "Request failed after N retries") and forces a clean
    # fallback to standard HTTPS downloads from the Hugging Face Hub.
    HF_HUB_DISABLE_XET=1 \
    NLTK_DATA=/usr/share/nltk_data \
    # COEUR is imported as a top-level package (e.g. `from coeur.score import Coeur`).
    PYTHONPATH=/app \
    # Use an isolated virtualenv so pip is not blocked by Ubuntu 24.04's
    # PEP 668 "externally-managed-environment" guard, and so `python`/`pip`
    # resolve to Python 3.12.
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- System dependencies ----------------------------------------------------
# build-essential / python3-dev: needed to build wheels that lack manylinux
#   binaries (e.g. some scientific deps).
# git, curl: convenience for reviewers and a few pip installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        build-essential \
        git \
        curl \
        ca-certificates \
    && python3 -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

# --- Python dependencies ----------------------------------------------------
# Copy only the requirements first to maximise Docker layer caching.
COPY requirements-linux.txt ./requirements-linux.txt

# Use uv (a fast, drop-in pip replacement) to resolve and install the pinned
# dependencies. It installs into the active virtualenv ($VIRTUAL_ENV) and is
# considerably faster than pip for the large torch/tensorflow/CUDA wheels.
RUN python -m pip install --upgrade pip uv \
    && uv pip install -r requirements-linux.txt

# --- Language model / NLP assets --------------------------------------------
# spaCy model used by the USQA baseline.
RUN python -m spacy download en_core_web_sm

# NLTK corpora used by COEUR cohesion and the AQUSA/USQA baselines.
RUN python -m nltk.downloader -d "$NLTK_DATA" \
        averaged_perceptron_tagger_eng \
        averaged_perceptron_tagger \
        wordnet \
        omw-1.4 \
        stopwords \
        punkt_tab \
        punkt

# Pre-fetch the Hugging Face / SentenceTransformer models so the demo and
# experiments run without network access.
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Encoder used by coeur.score.Coeur (default model_name="bert-base-uncased").
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")

# Sentence embedders used by cohesion scoring and the spec retriever.
SentenceTransformer("all-mpnet-base-v2")
SentenceTransformer("all-MiniLM-L6-v2")

# Base models fine-tuned by the SFT experiment (experiments/llm_based/sft.ipynb).
# The other LLM experiments use remote services (Azure OpenAI / Ollama) and so
# cannot be pre-baked — they are configured via .env (see .env.example).
for _model_id in ("HuggingFaceTB/SmolLM2-135M", "distilgpt2"):
    AutoTokenizer.from_pretrained(_model_id)
    AutoModelForCausalLM.from_pretrained(_model_id)
print("Pre-downloaded Hugging Face models OK")
PY

# --- Project source ---------------------------------------------------------
COPY . /app

# --- Jupyter configuration --------------------------------------------------
# Install a server config so that root_dir is always /app (the repo root)
# regardless of how Jupyter is launched (docker compose, VS Code attach, etc.).
RUN mkdir -p /root/.jupyter /root/.ipython/profile_default/startup
COPY jupyter_server_config.py /root/.jupyter/jupyter_server_config.py
# Ensure every notebook kernel starts with cwd=/app (the repo root) so that
# relative paths like "datasets/..." resolve correctly in any notebook.
COPY ipython_startup_chdir.py /root/.ipython/profile_default/startup/00-chdir.py

# --- Jupyter ----------------------------------------------------------------
EXPOSE 8888

# Default: launch Jupyter Lab. Reviewers open the printed token URL.
# The repository root is the working directory, so example/coeur_demo.ipynb
# and the experiments/ notebooks resolve their relative dataset paths.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app"]
