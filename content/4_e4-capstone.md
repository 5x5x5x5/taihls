---
title: 'Chapter E4 — Capstone: your turn'
short_title: 'E4 · Capstone'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter E4 — Capstone: your turn

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/e4-capstone-starter.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- frame an **end-to-end** mini-project: a question, a dataset, a model, an honest evaluation;
- reuse the tools from earlier parts — exploring data, training, testing, checking fairness;
- evaluate your model **per group**, not just overall, and write up its **limitations**;
- follow a clear **rubric** and milestone plan, and start from a ready-made notebook.
```

## Start with a story — yours

You've now seen the whole arc of this book: spotting patterns in data, teaching a
computer to decide, learning from examples, reading sequences and language, and using
all of it **responsibly**. The best way to make it stick is to build something small
and complete, start to finish.

This chapter is different. There is no new concept to memorize — instead, here is a
project to make your own, a rubric to aim for, and a starter to get you moving today.
You won't need anything heavier than the tools you already have.

## The shape of a good mini-project

Every solid data project, from a classroom to a hospital research lab, walks through
the same five stages. Keep them in mind as a map.

```{admonition} Definition — the project arc
:class: important
1. **Question** — a specific, answerable question ("Can these measurements predict X?").
2. **Explore** — look at the data before modeling: sizes, balance, surprises.
3. **Model** — train a simple model first (you can always add complexity later).
4. **Evaluate** — measure accuracy, a baseline, *and* fairness across groups.
5. **Limitations** — write honestly about what your model cannot or should not do.
```

The last two stages are where this part of the book lives. A project that stops at
"96% accurate!" is only half done; a project that asks *accurate for whom?* and *where
would this be unsafe?* is the real thing.

## A suggested offline dataset

You don't need to download anything. scikit-learn ships small, clean datasets that run
anywhere — including in the in-page notebook. A good choice is `load_wine`: 178
samples, 13 chemical measurements, and three wine cultivars to tell apart. (Prefer a
medical flavour? `load_breast_cancer` from [](1_b1-knn-tumor.md) or `load_diabetes` work
the same way.)

```{tip}
Think of the wine cultivars (classes 0, 1, 2) as stand-ins for three patient
*subgroups*. Checking accuracy **per class** then becomes a direct rehearsal of the
**per-group fairness check** from [](1_e1-bias-in-medical-ai.md) — the same code, the
same question.
```

## A tiny worked starter

Here is a complete, runnable skeleton that touches all five stages. It is deliberately
minimal — your job is to extend it.

```{code-cell} python
:label: e4-starter
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. QUESTION: can the 13 chemical measurements predict the wine's cultivar?
wine = load_wine()
X, y = wine.data, wine.target

# 2. EXPLORE: how big and how balanced is the data?
print(f"{X.shape[0]} samples, {X.shape[1]} measurements, "
      f"{len(np.unique(y))} classes")
print("samples per class:", np.bincount(y))

# 3. MODEL: a simple, honest split + a simple classifier.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
model = LogisticRegression(max_iter=10000).fit(X_train, y_train)

# 4. EVALUATE: overall accuracy AND per-class ("per-group") accuracy.
pred = model.predict(X_test)
print(f"\nOverall accuracy: {model.score(X_test, y_test):.1%}")
for c in np.unique(y_test):
    mask = y_test == c
    print(f"  class {c} ({wine.target_names[c]}): "
          f"{(pred[mask] == y_test[mask]).mean():.1%} on {mask.sum()} samples")
```

When you run this you'll see roughly **96%** overall — but notice the per-class lines.
One class sits at **100%** while another (here, class 1) trails a little. That small
gap is your starting point for stage 5: *why* is one group harder, and would that
matter if these were patients?

## Take it further

The starter stops at the baseline. Your capstone is to push past it. A few directions,
all using only the light toolkit:

- **Add a baseline to beat.** What accuracy would "always guess the most common class"
  give? (Compare against [](#e4-starter).)
- **Add abstention.** Use `predict_proba` and the coverage–accuracy idea from
  [](3_e3-humans-in-the-loop.md): how much can the model auto-decide at 95% confidence?
- **Stress-test fairness.** Make one class artificially rare in training (the E1 trick)
  and watch the per-class gap open up.
- **Check privacy thinking.** If these rows were patients, which columns would be
  quasi-identifiers? (See [](2_e2-privacy-consent.md).)

## The starter notebook

A ready-made scaffold with TODO cells for each milestone lives in the companion
notebook. Open it, fill in the TODOs, and you have your capstone.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/e4-capstone-starter.ipynb)

`notebooks/e4-capstone-starter.ipynb` — runs in Colab or any Jupyter; light tools only.

## The rubric

Aim for all five, in order. You don't need a fancy model — you need an honest one.

```{admonition} Capstone rubric & milestones
:class: important
**Milestone 1 — Question (15%).** State one specific, answerable question and name your
dataset. One paragraph.

**Milestone 2 — Exploration (20%).** Report the dataset's size, the number of classes,
and how *balanced* they are. Note at least one thing that surprised you. Include one
small plot or table.

**Milestone 3 — Model (20%).** Split into train/test, train a simple classifier, and
report **overall accuracy** *and* the **baseline** ("always guess the common class").
Your model should beat the baseline.

**Milestone 4 — Fairness evaluation (25%).** Break accuracy down **per group** (per
class, or per a subgroup you define). Report the **gap**. Add either abstention
(coverage vs. accuracy) or a deliberate stress-test of representation.

**Milestone 5 — Limitations write-up (20%).** In a short paragraph, answer: For whom
does this work worst? What data would you need to do better? Where would deploying
this be unsafe, and would you keep a **human in the loop**? Run your project through
the **checklist** in [](3_e3-humans-in-the-loop.md).
```

## Exercise

```{admonition} Exercise E4.1 — Find your baseline
:class: hint
Before training anything fancy, compute the **baseline** for the wine data: the
accuracy you'd get by always predicting the *most common* class in the training set.
Your real model in [](#e4-starter) should comfortably beat it — by how much?
```

```{admonition} Solution
:class: dropdown
```python
from collections import Counter
most_common = Counter(y_train).most_common(1)[0][0]
baseline_acc = (y_test == most_common).mean()
print(f"Baseline (always guess class {most_common}): {baseline_acc:.1%}")
print(f"Model accuracy: {model.score(X_test, y_test):.1%}")
```

The most common class covers only about **40%** of samples, so the baseline is roughly
**40%**. Our model's ~96% towers over it — confirming the measurements really do carry
signal, not just a lucky guess at the majority class. *Always* report this comparison:
a number means little until you know what beating chance looks like.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A complete project follows five stages: **question → explore → model → evaluate →
  limitations** — and the last two are what make it responsible.
- Beat a **baseline**, evaluate **per group**, and write your **limitations** honestly:
  for whom does it fail, and where would it be unsafe?
- Everything you need is light and offline (`load_wine`, `load_breast_cancer`,
  `load_diabetes`) — start from the companion notebook and fill in the TODOs.
- You've reached the end of TAIHLS. You can now read data critically, build a model,
  and ask the hard questions about who it serves. That last skill — the responsible
  one — is the most valuable thing you'll carry out of this book. Go build something.
```
