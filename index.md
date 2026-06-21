---
title: 'TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences'
---

# Welcome

This is an interactive textbook about how artificial intelligence is used in
health and the life sciences. You don't just read it — you run it. Small examples
run right on the page, and the heavier ones open in Google Colab with a single click.

## What you'll learn

The book moves in five parts, from "is there a signal in this data?" all the way to
the foundation models behind today's headlines — and how to use them responsibly.

- **Part A — Seeing patterns in biological data.** Load real data, draw it, summarize
  it, and learn the most important habit in all of science: telling *association*
  apart from *causation*.
- **Part B — Teaching a computer to decide.** Build your first classifiers and learn
  why a model that is "99% accurate" can still be dangerous in medicine.
- **Part C — Learning from examples at scale.** From fitting a line to building a
  neural network and reading medical images.
- **Part D — Life as sequences, language as data.** DNA and proteins as text,
  language models that read and write, and the foundation models (AlphaFold, ESM,
  scGPT) transforming biology.
- **Part E — Using AI responsibly in health.** Bias, privacy, and keeping humans in
  the loop — ending with a capstone project that's all yours.

## Who it's for

Advanced high-school students and first-year undergraduates. We assume you're
curious and comfortable with **basic algebra** and a **little** programming. We do
**not** assume calculus or linear algebra, and we define every new term the first
time we use it.

## Prerequisites

- Comfort with algebra (variables, formulas, reading a graph).
- Willingness to run a line of code and see what happens.
- No prior AI, statistics, or biology coursework required.

## How to run the code

There are two tiers, and you need **nothing installed** for either:

1. **Light examples — in the page or in your browser.** Most chapters carry small,
   self-contained code cells. They run during the build, and (best-effort) right in
   your browser via JupyterLite.
2. **Heavy examples — in Google Colab.** Training real neural networks, running a
   language model, or computing protein embeddings needs a GPU. Those live in
   **companion notebooks** with an *Open in Colab* badge at the top of the chapter —
   click it and you get a free GPU.

See [](back/reproducibility.md) for all the details, including how to run everything
locally with `uv`.

```{tip}
New here? Read the [Preface](preface.md) for *why* this book exists, then the
[Advice to the student](advice.md) for *how* to get the most from it.
```
