---
title: Reproducibility & running the code
---

# Reproducibility & running the code

Everything in this book is meant to *run*. There are two tiers of code, and you need
nothing installed for either.

## Tier 1 — Light examples (in the page / your browser)

Most chapters carry small, self-contained code cells using only the standard
scientific-Python stack (`numpy`, `pandas`, `matplotlib`, `scikit-learn`). They use
small, built-in datasets — no downloads — so they run during the book build and,
best-effort, right in your browser via **JupyterLite** (look for the run button).

```{note}
JupyterLite runs Python in your browser via WebAssembly (Pyodide). It covers the
light examples; the heavy ones below need a real GPU, which is what Colab is for.
```

## Tier 2 — Heavy examples (Google Colab, free GPU)

Training a real neural network, running a language model, or computing protein
embeddings needs a GPU. Those live in **companion notebooks** under `notebooks/`,
linked from the relevant chapters by an **Open in Colab** badge. Click the badge,
then *Runtime → Change runtime type → GPU*. Each notebook installs its own
dependencies in the first cell — nothing to set up in advance.

## Running locally with `uv`

```bash
git clone git@github.com:5x5x5x5/taihls.git
cd taihls
uv sync                              # exact versions from uv.lock
uv run jupyter book start --execute  # serve + run the light code
```

The heavy companion notebooks are best run on Colab (for the free GPU), but you can
also open them locally with `uv run jupyter lab` if you have a CUDA GPU.

<!-- PLACEHOLDER -->
Add any dataset notes or version caveats here as the book grows.
