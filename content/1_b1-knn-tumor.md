---
title: 'Chapter B1 — Is this tumor benign or malignant?'
short_title: 'B1 · Nearest neighbours'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter B1 — Is this tumor benign or malignant?

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain how a **classifier** turns measurements into a decision;
- describe the **k-nearest-neighbours** idea in one sentence;
- build a working classifier from scratch and measure how often it is right;
- explain why measurements must be put on a **common scale** before comparing them.
```

## Start with a story

A pathologist looks at a sample of breast tissue under a microscope and measures the
cells: how big they are, how rough their edges look, how much they vary. From years
of experience, they form a judgement — does this look **benign** (harmless) or
**malignant** (cancerous)?

Could a computer learn that same judgement from past, already-diagnosed samples? The
surprisingly simple answer is yes, and the first method we'll meet barely looks like
"learning" at all. Let's define the goal precisely.

```{admonition} Definition — classifier
:class: important
A **classifier** is a rule that takes some measurements about a thing and outputs a
**category** for it — here, *benign* or *malignant*.
```

## The idea: look at your neighbours

Here is the whole trick. To classify a new tumor, find the past tumors whose
measurements are **most similar** to it, and let them vote. If most of the nearest
ones were malignant, guess malignant. That's it.

```{admonition} Definition — k-nearest-neighbours (k-NN)
:class: important
**k-nearest-neighbours** classifies a new example by finding the $k$ already-labelled
examples closest to it (its "neighbours") and taking the **majority vote** of their
labels. "Closest" means the smallest straight-line distance between their
measurements.
```

We'll use a famous real dataset: 569 breast-tumor samples, each already diagnosed,
each described by measurements of the cell nuclei. To keep things easy to picture,
we'll start with just two measurements per tumor.

```{code-cell} python
:label: load-data
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
# Two interpretable measurements: mean radius and mean texture.
features = [0, 1]
X = data.data[:, features]
y = data.target            # 0 = malignant, 1 = benign

print(f"{X.shape[0]} tumors, described by {X.shape[1]} measurements each")
print(f"Malignant: {(y == 0).sum()}   Benign: {(y == 1).sum()}")
```

## Putting measurements on a common scale

There's a catch. "Mean radius" runs from about 7 to 28, while "mean texture" runs
from about 10 to 40. Straight-line distance would let the bigger numbers shout over
the smaller ones. So we first **standardize** each measurement.

```{admonition} Definition — standardize
:class: important
To **standardize** a measurement, subtract its average and divide by its spread, so
every measurement is recentred to average 0 with a comparable range. Now no single
measurement dominates the distance just because its numbers happen to be larger.
```

We split the data into a part to learn from and a part to test on, then standardize
both using the *training* numbers only.

```{code-cell} python
:label: split-scale
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

mean, spread = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / spread
X_test = (X_test - mean) / spread
print(f"Learning from {len(X_train)} tumors, testing on {len(X_test)}.")
```

## Building k-NN from scratch

We don't need a library for the classifier itself — it's a few lines. For one new
tumor, measure its distance to every training tumor, take the $k$ closest, and vote.

```{code-cell} python
:label: knn
def classify_one(x, X_train, y_train, k=5):
    # Straight-line distance from x to every training tumor.
    distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))
    nearest = np.argsort(distances)[:k]      # the k closest
    vote = y_train[nearest].mean()           # fraction that are benign
    return 1 if vote >= 0.5 else 0           # majority wins

# Classify every test tumor and measure how often we are right.
predictions = np.array([classify_one(x, X_train, y_train, k=5) for x in X_test])
accuracy = (predictions == y_test).mean()
print(f"Correct on {accuracy:.1%} of held-out tumors using k = 5.")
```

Nearly nine out of ten correct — from a rule you can read in one breath, using only
two of the thirty available measurements.

## Seeing the decision

Let's look at the training tumors and where one test tumor's five neighbours fall.

```{code-cell} python
:label: knn-fig
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))
for label, name, color in [(0, "malignant", "#c0392b"), (1, "benign", "#2980b9")]:
    pts = X_train[y_train == label]
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, alpha=0.5, label=name)

x_new = X_test[0]
d = np.sqrt(((X_train - x_new) ** 2).sum(axis=1))
neighbours = X_train[np.argsort(d)[:5]]
ax.scatter(*x_new, s=180, marker="*", color="black", label="new tumor", zorder=5)
ax.scatter(neighbours[:, 0], neighbours[:, 1], s=90, facecolors="none",
           edgecolors="black", linewidths=1.5, label="its 5 neighbours")
ax.set_xlabel("mean radius (standardized)")
ax.set_ylabel("mean texture (standardized)")
ax.legend(loc="upper right", fontsize=8)
fig
```

In [](#knn-fig) the two classes form two loose clouds. The new tumor (★) is judged
by the company it keeps: its five circled neighbours vote, and the majority wins.

## Exercise

```{admonition} Exercise B1.1 — Does k matter?
:class: hint
We used `k = 5`. Re-run the accuracy for `k = 1`, `k = 51`, and `k = 301`. What
happens at each extreme, and what does the accuracy at `k = 301` remind you of?

*Hint: about 63% of the training tumors are benign.*
```

```{admonition} Solution
:class: dropdown
```python
for k in (1, 51, 301):
    preds = np.array([classify_one(x, X_train, y_train, k=k) for x in X_test])
    print(f"k = {k:>3}: {(preds == y_test).mean():.1%}")
```

You should see roughly `82%`, `92%`, `63%`. With `k = 1` a single noisy neighbour can
flip the answer — it *overfits* to quirks. A moderate `k` does best. But push `k` too
far and almost every training tumor gets a vote: the answer becomes "whatever class is
most common overall," so accuracy collapses to the **63% benign** baseline — the score
you'd get by ignoring the measurements entirely. The best `k` is in between, and
choosing it well is exactly what [](2_b2-train-test.md) is about.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **classifier** turns measurements into a category; k-NN does it by **majority
  vote of the nearest examples**.
- The whole method is a few lines of code and needs no training step — it just
  remembers the data.
- **Standardizing** measurements first is essential, or large-numbered features
  dominate the distance.
- The choice of `k` is a real decision: too small overfits, too large washes out the
  signal — motivating the careful evaluation in the next chapter.
```
