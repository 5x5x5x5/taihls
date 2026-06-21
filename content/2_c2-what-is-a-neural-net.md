---
title: 'Chapter C2 — What is a neural network?'
short_title: 'C2 · Neural networks'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter C2 — What is a neural network?

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/5x5x5x5/taihls/HEAD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/c2-first-neural-net.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- describe a **neuron**, a **weight**, an **activation**, and a **layer** in plain words;
- build a tiny neural network by hand in NumPy and run a **forward pass**;
- explain why a single straight line can't separate every dataset;
- see how stacking neurons lets a network draw a **curved** decision boundary.
```

## Start with a story

In [](1_c1-fitting-a-line.md) we fit a straight line to dose–response data. Straight
lines are wonderfully simple — but biology is rarely straight. Imagine two kinds of cells
that, when you plot two of their measurements, fall into two interlocking crescents. No
single straight line can cleanly separate them. We need a model that can **bend**.

A **neural network** is exactly such a model: bend enough simple pieces together and you
can trace almost any shape. The pieces are called neurons, loosely inspired by how brain
cells pass signals along. Let's build one from scratch — small enough to read every
number — and watch it carve a curved boundary.

→ Train the real network in the [companion notebook](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/c2-first-neural-net.ipynb).

## The pieces of a network

```{admonition} Definition — neuron
:class: important
A **neuron** takes several numbers in, multiplies each by its own importance, adds the
results into one number, and passes that through a simple bending function. One neuron is
a tiny decision-maker; a network is many of them working together.
```

```{admonition} Definition — weight
:class: important
A **weight** is the importance a neuron gives to one of its inputs — exactly like the
*slope* parameter from the last chapter, but now there is one weight per input. Big
positive weight means "this input pushes my answer up"; negative means "it pushes it
down". The weights (plus a **bias**, a constant nudge) are the knobs a network learns.
```

```{admonition} Definition — activation
:class: important
An **activation** is the bending function applied to a neuron's summed input. We use the
**sigmoid**, which smoothly squashes any number into the range 0 to 1. Without this bend,
stacking neurons would just give another straight line — the bend is what lets a network
curve.
```

```{admonition} Definition — layer
:class: important
A **layer** is a group of neurons that all read the same inputs at the same time. Our
network has one **hidden layer** of a few neurons feeding a single **output** neuron. The
hidden layer invents useful in-between features; the output neuron combines them into a
final answer.
```

## Data that defies a straight line

`make_moons` gives us two interleaving crescents — a classic shape no straight line can
split. We'll standardize it as we did before so every measurement is on a common scale.

```{code-cell} python
:label: moons-data
import numpy as np
from sklearn.datasets import make_moons

np.random.seed(0)
X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
X = (X - X.mean(axis=0)) / X.std(axis=0)     # standardize both measurements
print(f"{X.shape[0]} points, {X.shape[1]} measurements each")
print(f"Class 0: {(y == 0).sum()}   Class 1: {(y == 1).sum()}")
```

## One neuron, one straight line

A single neuron can only draw a straight boundary — its weights tilt a line and its bias
slides it. Let's prove the limit by building one neuron with hand-picked weights and
seeing how often it's right on the moons.

```{code-cell} python
:label: one-neuron
def sigmoid(z):
    return 1 / (1 + np.exp(-z))               # squash to the range 0..1

# One neuron: two weights (one per input) and a bias.
w = np.array([1.2, -0.6])
b = 0.0
scores = sigmoid(X @ w + b)                   # forward pass through one neuron
preds = (scores >= 0.5).astype(int)
print(f"One straight-line neuron is right on {(preds == y).mean():.1%} of points.")
```

A single straight cut leaves many crescent points on the wrong side. We need to bend.

## A tiny network: one hidden layer

Now the real thing. We give a **hidden layer** of four neurons their own random weights,
let each one draw its own slanted line, and then let an **output** neuron combine their
four bent signals into a final yes/no. We don't train anything here — random weights are
enough to *show the machinery*; the companion notebook does the training.

```{code-cell} python
:label: tiny-net
rng = np.random.default_rng(7)

# Hidden layer: 4 neurons, each reading the 2 inputs -> a 2x4 weight grid + 4 biases.
W1 = rng.normal(0, 1.5, size=(2, 4))
b1 = rng.normal(0, 1.0, size=4)

# Output layer: 1 neuron reading the 4 hidden signals -> 4 weights + 1 bias.
W2 = rng.normal(0, 1.5, size=4)
b2 = 0.0

def network(points):
    hidden = sigmoid(points @ W1 + b1)        # 4 bent features per point
    output = sigmoid(hidden @ W2 + b2)        # combine them into one answer
    return output

scores = network(X)
preds = (scores >= 0.5).astype(int)
acc = max((preds == y).mean(), (1 - preds == y).mean())   # random net may flip labels
print(f"Untrained 4-neuron network agrees with the true split on {acc:.1%} of points.")
```

Even with **random, untrained** weights the little network already does better than the
single straight-line neuron — it is not stuck with one straight cut. Training (next chapters,
and the companion notebook) is just the process of turning those random knobs until the bend
lands in exactly the right place.

## Seeing the curved boundary

Let's colour the whole plane by what the network predicts, and overlay the two crescents.
A straight-line model would split this picture with one diagonal; watch what the network
does instead.

```{code-cell} python
:label: boundary-fig
import matplotlib.pyplot as plt

# Evaluate the network on a fine grid covering the plot.
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 300), np.linspace(-2.5, 2.5, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = network(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.6)
for label, color in [(0, "#2980b9"), (1, "#c0392b")]:
    pts = X[y == label]
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=color, edgecolors="white",
               linewidths=0.3, label=f"class {label}")
ax.set_xlabel("measurement 1 (standardized)")
ax.set_ylabel("measurement 2 (standardized)")
ax.set_title("A tiny network bends the boundary")
ax.legend(loc="upper right", fontsize=8)
fig
```

The shaded regions in [](#boundary-fig) are **not** split by a single straight line — the
boundary curves and wiggles. That flexibility is the whole reason neural networks matter
for messy biological and medical data. Each hidden neuron contributes one bend; stack
enough of them and the network can trace shapes far more intricate than this.

## Exercises

```{admonition} Exercise C2.1 — Why the bend matters
:class: hint
In the `tiny-net` cell, replace the sigmoid in the hidden layer with no bend at all — i.e.
make `hidden = points @ W1 + b1` (delete the `sigmoid(...)` wrapper). Re-run the boundary
plot. What happens to the curve, and why?
```

```{admonition} Solution
:class: dropdown
The boundary collapses back to a single straight line. Without the bending activation,
the hidden layer just does multiply-and-add, and a stack of multiply-and-adds is itself
only a multiply-and-add — mathematically equivalent to one straight-line neuron. The
**activation** is the ingredient that makes depth worth having.
```

```{admonition} Exercise C2.2 — More neurons, more bends
:class: hint
Change the hidden layer from 4 neurons to 12 (update the `size=` arguments in `W1`, `b1`,
and `W2`). Re-run the boundary plot a few times with different seeds. What kind of
boundaries can the larger network draw that the small one couldn't?
```

```{admonition} Solution
:class: dropdown
With more hidden neurons the boundary can take on more wiggles and enclosed pockets —
each extra neuron adds another possible bend. More capacity means the network *can* fit
more intricate shapes, but as we'll see in [](4_c4-how-models-learn.md) that same power
lets it fit noise it should ignore. Bigger is not automatically better.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **neuron** weights its inputs, sums them, and passes the total through a bending
  **activation**; the **weights** are its learnable knobs.
- A **layer** is a group of neurons reading the same inputs; a hidden layer plus an output
  neuron is the smallest useful **neural network**.
- A single neuron draws only a straight boundary; stacking neurons with a bending
  activation lets the network draw **curved** boundaries that straight lines cannot.
- We ran only a **forward pass** with fixed weights — *learning* those weights is the
  subject of [](4_c4-how-models-learn.md) and the companion PyTorch notebook.
```
</content>
</invoke>
