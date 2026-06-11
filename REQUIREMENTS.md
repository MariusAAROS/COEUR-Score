# REQUIREMENTS

This document describes the architecture, hardware, and software required to
build and run the **COEUR-Score** artifact. The artifact is packaged as a
self-contained Docker image; reviewers do not need to install any Python
dependencies manually.

The machine-readable dependency descriptors are committed in the repository:

| File | Purpose |
| --- | --- |
| [`Dockerfile`](Dockerfile) | Builds the complete, reproducible runtime image (OS packages, Python deps, NLP corpora, pre-baked models). |
| [`docker-compose.yml`](docker-compose.yml) | One-command build/run, port mapping, optional GPU reservation, and volume mounts. |
| [`requirements-linux.txt`](requirements-linux.txt) | Fully pinned (`pip freeze`) Python dependency list installed inside the image. **This is the authoritative list for Linux/Docker.** |
| [`.env.example`](.env.example) | Template for the optional credentials used only by the LLM-based experiments. |

---

## 1. Architecture

- **CPU architecture:** `x86-64` (`amd64`).
  - The base image (`nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`) and the
    pinned Linux wheels in `requirements-linux.txt` are `x86-64`.
  - ARM (e.g. Apple Silicon, AWS Graviton) is **not** supported as-is: the
    CUDA base image and several pinned wheels have no `arm64` build. On ARM
    hosts Docker would fall back to slow `x86-64` emulation and the GPU stack
    would be unavailable.

## 2. Operating system

- The **container** runs **Ubuntu 24.04** with **Python 3.12** (provided by the
  base image — no host Python is required).
- The **host** can be any OS that runs Docker:
  - **Linux** (recommended for GPU; native NVIDIA Container Toolkit support).
  - **Windows 10/11** via Docker Desktop with the **WSL2** backend (GPU
    pass-through supported with an up-to-date NVIDIA driver).
  - **macOS** via Docker Desktop (CPU-only; no NVIDIA GPU pass-through).

## 3. Software requirements (host)

| Software | Minimum version | Notes |
| --- | --- | --- |
| **Docker Engine** | 20.10+ | Docker Desktop 4.x on Windows/macOS. Tested with Docker 29.x. |
| **Docker Compose** | v2 (`docker compose`) | Optional; only needed for the `docker-compose.yml` workflow. |
| **NVIDIA driver** | recent, CUDA 12.8-capable | **GPU only.** Required on the host (not inside the container). |
| **NVIDIA Container Toolkit** | latest | **GPU only.** Enables `--gpus all` / the Compose `deploy.resources` block. |

> The artifact runs **without a GPU**. The scoring code detects the absence of a
> CUDA device and automatically falls back to CPU (slower, but functionally
> identical results). On a CPU-only host, omit `--gpus all` or delete the
> `deploy.resources` block in `docker-compose.yml`.

## 4. Hardware requirements

| Resource | CPU-only | With GPU |
| --- | --- | --- |
| **CPU** | x86-64, 4+ cores recommended | x86-64, 4+ cores |
| **RAM** | 8 GB minimum, 16 GB recommended | 16 GB recommended |
| **GPU** | not required | NVIDIA GPU, CUDA 12.8 compatible. 8 GB+ VRAM recommended for the optional SFT fine-tuning experiment. |
| **Disk** | ~20 GB free | ~20 GB free |

**Disk usage breakdown (approximate):** the CUDA/cuDNN base image plus the
PyTorch/TensorFlow/CUDA Python wheels dominate the footprint; the pre-baked
Hugging Face models, SentenceTransformers, NLTK corpora, and the spaCy
`en_core_web_sm` model add roughly 2–3 GB. Plan for **~20 GB** of free disk for
the build (intermediate layers + final image).

- **Non-commodity peripherals:** none required.
- **Network:** required **at build time** (to pull the base image, Python
  packages, and the pre-baked models/corpora). After a successful build, the
  core scoring pipeline and the demo run **fully offline**. The only features
  that need network at run time are the optional LLM-based experiments that
  call external services (see §6).

## 5. What is pre-installed in the image

Built once during `docker build`, so reviewers can run offline:

- **Python 3.12** in an isolated virtualenv (`/opt/venv`).
- All Python dependencies from `requirements-linux.txt` (installed with `uv`).
- **NLTK** corpora: `punkt`, `punkt_tab`, `wordnet`, `omw-1.4`, `stopwords`,
  `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`.
- **spaCy** model: `en_core_web_sm`.
- **Hugging Face / SentenceTransformer** models:
  - `bert-base-uncased` (default COEUR encoder),
  - `all-mpnet-base-v2`, `all-MiniLM-L6-v2` (cohesion / retrieval embedders),
  - `HuggingFaceTB/SmolLM2-135M`, `distilgpt2` (base models for the SFT
    experiment).
- **Jupyter Lab**, exposed on port **8888**.

## 6. Optional external services (LLM-based experiments only)

The notebooks under `experiments/llm_based/` (`icl*.ipynb`, `sft.ipynb`) can
exercise large language models. These are **not** bundled in the image because
they are remote services or very large models:

- **Azure OpenAI** (`gpt-4.1`) — set `AZURE_OPENAI_API_KEY` and
  `AZURE_OPENAI_ENDPOINT` in a `.env` file.
- **Ollama** (`mistral-small:22b`) — run an Ollama server on the host and set
  `OLLAMA_HOST` (default `http://host.docker.internal:11434`).

Copy `.env.example` to `.env` and fill in the values only if you intend to run
these experiments. The pre-computed outputs of these experiments are committed
under `experiments/` so the analysis notebooks (e.g. `icl3.ipynb`) reproduce
the figures without re-running generation.

## 7. Build & run

```bash
# Build the image
docker build -t coeur-score:latest .

# Run Jupyter Lab (open the printed token URL in a browser)
docker run --rm -p 8888:8888 coeur-score:latest            # CPU
docker run --rm --gpus all -p 8888:8888 coeur-score:latest # GPU

# Or, with Docker Compose
docker compose up --build

# Headless smoke test
docker run --rm coeur-score:latest \
  python -c "from coeur.score import Coeur; print('COEUR import OK')"
```
