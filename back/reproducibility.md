---
title: Reproducibility & running the code
---

# Reproducibility & running the code

Everything in this book is meant to *run*. There are three ways to run it, from
zero-effort to full control.

## 1. In your browser, no install (recommended for students)

This book ships with **JupyterLite** (`project.jupyter.lite: true`), so every code
cell can run directly in the page — Python executes in your browser via
WebAssembly. Look for the launch / run button on any chapter with code.

```{note}
JupyterLite runs a slimmed-down Python (Pyodide). Most teaching libraries
(`matplotlib`, `numpy`, `pandas`) work; very large or compiled-only packages may
not. The examples in this book are chosen to run there.
```

## 2. Binder — a full cloud environment

The **Binder** launch button (configured in `myst.yml` under `project.binder`)
spins up a complete environment from `requirements.txt`. Slower to start, but it's
real CPython with no browser limits.

## 3. Locally with `uv`

```bash
git clone git@github.com:5x5x5x5/taihls.git
cd taihls
uv sync                              # exact versions from uv.lock
uv run jupyter book start --execute  # serve + run the code
```

<!-- PLACEHOLDER -->
Add any dataset download steps, version notes, or hardware/runtime caveats here.
