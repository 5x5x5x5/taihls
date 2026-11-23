---
title: 'TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences'
short_title: 'Home'
---

# Artificial Intelligence for Health & Life Sciences

*A full-semester interactive online textbook for advanced high-school students and introductory undergraduates — where every idea starts with a real health or biology puzzle, a simple hand-worked example, and short Python code you can run right in your browser.*

Artificial intelligence is transforming medicine and biology: helping doctors spot cancers on mammograms, folding proteins and designing new medicines, reading genomes, and drafting patient summaries. Yet most introductions either bury beginners under college calculus and linear algebra or hand out buzzwords without ever showing how anything works.

**TAIHLS** is built specifically for a **16-week semester**. Every chapter focuses on **one clear big idea**, explains every term in plain English the first time it appears, and builds up from high-school algebra and short, readable Python code — no calculus, no linear algebra, and no prior biology or AI coursework required.

## Course roadmap (6 Units · 28 Bite-Sized Lessons)

::::{grid} 1 1 2 2

:::{card} Unit 0 · Getting Started (Week 1)
:link: content/0_1-python-and-data-warmup.md
:header: **Python & Data Warm-Up**

Refresh variables, lists, loops, functions, and simple charts using a morning heart-rate log — so coding never gets in the way of the science.
:::

:::{card} Unit 1 · Models, Data & Scientific Thinking (Weeks 1–3)
:link: content/1_1-exponential-and-logistic-growth.md
:header: **5 Lessons · Outbreaks, Data & Evidence**

Compare exponential and limited growth, simulate a school outbreak (SIR), explore real diabetes data, untangle correlation from causation, and master base rates.
:::

:::{card} Unit 2 · Teaching a Computer to Decide (Weeks 4–6)
:link: content/2_1-nearest-neighbors.md
:header: **5 Lessons · Classical Machine Learning**

Build a breast-tumor classifier from scratch, watch overfitting happen, catch sneaky **data leakage**, weigh false alarms against missed cases, and compare lines, trees, and forests.
:::

:::{card} Unit 3 · How Neural Networks Learn and See (Weeks 7–9)
:link: content/3_1-fitting-a-line-and-loss.md
:header: **5 Lessons · Deep Learning (No Calculus!)**

Measure mistakes with loss, roll downhill with calculus-free gradient descent, train a tiny neural network, write a sliding image filter by hand, and catch a cheating hospital X-ray AI.
:::

:::{card} Unit 4 · DNA, Language & Biological AI (Weeks 10–12)
:link: content/4_1-dna-and-proteins-as-text.md
:header: **6 Lessons · From Hemoglobin to AlphaFold 3**

Decode real human hemoglobin DNA, align mutated sequences, see how chatbots predict words, unpack embeddings & attention, build an open-book RAG helper, and tour AlphaFold 3, ESM3, Evo 2, and AlphaGenome.
:::

:::{card} Unit 5 · Fairness, Privacy, Safety & Capstone (Weeks 13–16)
:link: content/5_1-fairness-and-bias.md
:header: **6 Lessons · Responsible Health AI**

Audit hidden healthcare bias, test re-identification and k-anonymity, try differential privacy and federated learning, keep humans in the loop, and finish with your own 2-week biomedical capstone.
:::

::::

## Who this course is for

- **Advanced high-school students (grades 11–12)** taking a one-semester AI, data science, or computational biology elective.
- **First-year undergraduates** looking for a welcoming, hands-on introduction before upper-level courses.
- **What you need:** High-school algebra (slopes, percentages, reading x–y graphs) and curiosity. **Unit 0** covers every Python skill you need to begin.

## How to run the code

::::{tab-set}

:::{tab-item} Tier 1 · In Your Browser (No Install)
Every chapter's code cells use simple, lightweight Python (`numpy`, `pandas`, `matplotlib`, `scikit-learn`) with built-in or self-contained datasets. They run right in the page via **JupyterLite** — zero setup.
:::

:::{tab-item} Tier 2 · Google Colab (6 Optional GPU Labs)
Six lessons pair with **companion Google Colab notebooks** (`notebooks/*.ipynb`) for bigger experiments: PyTorch neural nets, real pediatric chest X-rays, a live language model, real ESM-2 protein embeddings, a clinical audit lab, and the capstone starter. Look for the *Open in Colab* button at the top of those chapters.
:::

:::{tab-item} Local (`uv`)
Instructors or students running locally can launch the full site in two commands:

```bash
uv sync
uv run jupyter book start --execute
```

See [Reproducibility & Running the Code](back/reproducibility.md) for details.
:::

::::

```{tip}
**Start here:** Read the [Preface](preface.md), skim [How to Learn from This Book](advice.md), check the [16-Week Semester Pacing Guide](pacing.md), and jump straight into [Chapter 0.1 — Python Warm-Up](content/0_1-python-and-data-warmup.md)!
```
