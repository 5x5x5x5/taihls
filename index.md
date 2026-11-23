---
title: 'TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences'
short_title: 'Home'
---

# Artificial Intelligence for Health & Life Sciences

*An interactive, open-source textbook where every idea starts with a real biomedical puzzle — and every line of code runs right in your browser.*

Artificial intelligence is reshaping medicine and biology: predicting the 3D structures of proteins and drug complexes, reading radiology scans alongside clinicians, decoding the non-coding genome, and assisting doctors during diagnostic conversations. Yet most introductions either bury beginners under graduate calculus or hand out buzzwords without ever opening the hood.

**TAIHLS** takes a different path. With only **high-school algebra**, curiosity, and a willingness to tinker with short Python labs, you will build small working versions of the real algorithms behind modern health and life-science AI — and develop the scientific skepticism needed to tell genuine breakthroughs from dangerous hype.

## Explore the five parts

::::{grid} 1 1 2 2

:::{card} Part 1 · Data, Uncertainty & Causality
:link: content/1_1-models-and-outbreaks.md
:header: **Foundations of Scientific Reasoning**

Simulate outbreaks (exponential, logistic, SIR), inspect messy clinical tables, untangle **Simpson's paradox** with causal diagrams, and master base rates and bootstrap confidence intervals.
:::

:::{card} Part 2 · Classical ML in the Clinic
:link: content/2_1-nearest-neighbors-and-features.md
:header: **Decisions, Metrics & Honest Validation**

Build a breast-tumor classifier from scratch, catch subtle **data leakage**, trade sensitivity against specificity with ROC/PR curves and calibration plots, and compare trees, forests, and linear rules.
:::

:::{card} Part 3 · Deep Learning & Medical Imaging
:link: content/3_1-loss-and-gradient-descent.md
:header: **Opening the Black Box (No Calculus)**

Roll downhill on loss landscapes, train a neural network on a curved drug-synergy zone, write a 2D image convolution filter by hand, and expose **shortcut learning** in a simulated hospital scan experiment.
:::

:::{card} Part 4 · Sequences, Language & Foundation Models
:link: content/4_1-dna-proteins-and-alignment.md
:header: **From DNA Letters to AlphaFold 3 & Evo 2**

Align real hemoglobin sequences, visualize Transformer **self-attention**, build a mini **RAG** pipeline to curb hallucinations, and tour **AlphaFold 3, ESM3, Evo 2, AlphaGenome, and Virtual Cells**.
:::

::::

::::{grid} 1 1 1 1

:::{card} Part 5 · Responsible, Safe & Equitable Health AI
:link: content/5_1-algorithmic-fairness.md
:header: **Fairness, Privacy, Governance & Your Capstone**

Audit proxy-label and representation bias, code **differential privacy** and **federated learning** across three hospitals, study **FDA PCCPs, the EU AI Act, and TRIPOD+AI**, and complete a biomedical capstone project.
:::

::::

## Who this book is for

- **Advanced high-school students** exploring careers in medicine, biology, data science, or AI.
- **First-year undergraduates** in biology, pre-med, public health, biomedical engineering, or computer science looking for an applied, intuition-first gateway.
- **Curious self-learners and instructors** seeking a classroom-ready, zero-install interactive curriculum.

**Prerequisites:** Comfort with high-school algebra (variables, slopes, reading graphs) and willingness to run and edit short Python snippets. **No calculus, linear algebra, statistics, or prior biology coursework is assumed** — every technical term is defined on first use and collected in the [Glossary](back/glossary.md).

## How to run the code

This book uses a **two-tier execution design** so nothing gets between you and the ideas:

::::{tab-set}

:::{tab-item} Tier 1 · In-Page (Zero Install)
Every chapter's code labs use lightweight, Pyodide-compatible Python (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`) with built-in or self-contained biomedical datasets. They execute cleanly during the build and run right in your browser via **JupyterLite** — no account, no setup, no downloads.
:::

:::{tab-item} Tier 2 · Google Colab (Free GPU)
Six chapters pair with **companion Colab notebooks** (`notebooks/*.ipynb`) for heavyweight GPU work: training PyTorch neural networks, classifying real pediatric chest X-rays from **MedMNIST**, inspecting live next-token probabilities in an open LLM, computing real **ESM-2** protein embeddings, auditing clinical fairness, and launching your capstone. Look for the *Open in Colab* badge at the top of those chapters.
:::

:::{tab-item} Local (`uv`)
Want full offline control? Clone the repo and launch the entire textbook locally in two commands:

```bash
uv sync
uv run jupyter book start --execute
```

See [Reproducibility & Running the Code](back/reproducibility.md) for details.
:::

::::

```{tip}
**First time here?** Read the [Preface](preface.md) for why we wrote this book, [How to Learn from This Book](advice.md) to master our **PRIMM** coding habit, and the [Course Map & Pacing Guide](pacing.md) to pick a reading schedule.
```
