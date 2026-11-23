---
title: 'Reproducibility & Running the Code'
short_title: 'Reproducibility'
---

# Reproducibility & Running the Code

Every figure, simulation, and model in **TAIHLS** is produced by live, repeatable Python code. Everything is organized into two tiers so students and teachers never get stuck fighting software installations.

## Tier 1 — In-page interactive labs (zero install)

All **28 chapters** rely exclusively on standard, Pyodide-compatible Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`) and built-in or self-contained datasets. No downloads, logins, or API keys are needed, and every code cell runs directly in your browser via **JupyterLite**.

## Tier 2 — Companion Google Colab notebooks (free GPU)

Six chapters link to optional deeper experiments in `notebooks/` that open in **Google Colab** with one click:

| Notebook | Paired Lesson | What You'll Run |
|---|---|---|
| `notebooks/01-pytorch-neural-net.ipynb` | [](../content/3_3-inside-a-neural-network.md) | Train a drug-window neural net in PyTorch |
| `notebooks/02-medical-images-cnn.ipynb` | [](../content/3_4-how-computers-see-scans.md) | Train a real CNN on pediatric chest X-rays (MedMNIST) |
| `notebooks/03-clinical-llm-probabilities.ipynb` | [](../content/4_3-predicting-the-next-word.md) | Inspect live next-token probabilities in a real LLM |
| `notebooks/04-protein-foundation-models-esm2.ipynb` | [](../content/4_6-foundation-models-of-life.md) | Compute real ESM-2 protein embeddings & score the sickle-cell mutation |
| `notebooks/05-fairness-and-abstention-audit.ipynb` | [](../content/5_4-humans-in-the-loop.md) | Run a full subgroup fairness, calibration & abstention audit (CPU) |
| `notebooks/06-capstone-starter.ipynb` | [](../content/5_6-capstone-project.md) | Complete starter scaffold for all 3 biomedical capstone tracks (CPU) |

## Running locally with `uv`

Exact package versions are locked in `pyproject.toml` and `uv.lock` (with `requirements.txt` kept in sync for Binder):

```bash
git clone https://github.com/5x5x5x5/taihls.git
cd taihls
uv sync
uv run jupyter book start --execute
```

To build the static HTML site with strict validation:

```bash
uv run jupyter book build --html --execute --strict
```
