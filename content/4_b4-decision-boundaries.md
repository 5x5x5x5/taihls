---
title: 'Chapter B4 — Drawing the line'
short_title: 'B4 · Decision boundaries'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter B4 — Drawing the line

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- define a **decision boundary** and read one off a 2-D plot;
- fit a **logistic regression** classifier and draw the region it predicts;
- tell **linearly separable** data from data that needs a **non-linear** boundary;
- describe "learning" as adjusting numbers until the boundary sits in a good place.
```

## Start with a story

So far our classifiers have felt a little like black boxes: data goes in, a label
comes out. But when a model uses just two measurements, we can actually *see* what it
is doing. It divides the picture into regions — "I'll call everything over here
malignant, everything over there benign" — and the line between those regions tells
the whole story of the model's decision. In [](1_b1-knn-tumor.md) we plotted the
tumors as two clouds of points; now we will draw the fence the model puts between
them.

```{admonition} Definition — decision boundary
:class: important
A **decision boundary** is the dividing line (or surface) where a classifier switches
from predicting one class to predicting the other. On one side it says "benign," on
the other "malignant"; the boundary is the exact set of borderline cases between them.
```

## A model that draws a straight fence

We'll meet a new classifier, **logistic regression**, which despite its name is a
workhorse for *classification*. Its idea is simple: combine the measurements into a
single weighted score, and predict one class when the score is high and the other when
it is low. The place where the score tips over is — for two measurements — a straight
line.

```{admonition} Definition — logistic regression
:class: important
**Logistic regression** is a classifier that gives each measurement a *weight*, adds
up the weighted measurements into a score, and turns that score into a probability of
the positive class. With two measurements its decision boundary is a straight line.
```

Let's use the same two interpretable features from B1 — mean radius and mean texture —
and fit the model.

```{code-cell} python
:label: b4-fit
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X = data.data[:, [0, 1]]                  # mean radius, mean texture
y = data.target                           # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

clf = LogisticRegression().fit(X_train, y_train)
print(f"Test accuracy with a straight-line boundary: {clf.score(X_test, y_test):.1%}")
```

## Drawing the region

To *see* the boundary we use a classic trick: cover the plot with a fine grid of
points (a **meshgrid**), ask the model to classify every grid point, and shade the two
predicted regions different colors. The colour change traces out the decision boundary
for free.

```{code-cell} python
:label: b4-boundary
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# A grid of points covering the feature space.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Classify every grid point, then reshape to the grid.
grid = np.c_[xx.ravel(), yy.ravel()]
zz = clf.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, zz, alpha=0.25,
            cmap=ListedColormap(["#c0392b", "#2980b9"]))
for label, name, color in [(0, "malignant", "#c0392b"), (1, "benign", "#2980b9")]:
    pts = X_train[y_train == label]
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, alpha=0.6, label=name)
ax.set_xlabel("mean radius")
ax.set_ylabel("mean texture")
ax.set_title("The straight line logistic regression learned")
ax.legend(loc="upper right", fontsize=8)
fig
```

In [](#b4-boundary) the shaded regions show *every* prediction the model would make,
and the seam between the red and blue shading is the decision boundary. Notice it is a
single straight line — that is the signature of a **linear** classifier.

## When a straight line is not enough

The tumor clouds happen to be roughly separable by a line. But many problems are not.

```{admonition} Definition — linear vs. non-linear separation
:class: important
Data is **linearly separable** if a single straight line can put (almost) all of one
class on one side and the other class on the other. When no straight line works and
the boundary has to *curve*, the problem needs **non-linear** separation.
```

The `make_moons` dataset is the classic example: two interleaving crescents that no
straight line can cleanly split. Watch logistic regression struggle.

```{code-cell} python
:label: b4-moons
from sklearn.datasets import make_moons

Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=0)
moon_clf = LogisticRegression().fit(Xm, ym)
print(f"Straight-line accuracy on the two moons: {moon_clf.score(Xm, ym):.1%}")

mx, my = np.meshgrid(np.linspace(Xm[:, 0].min() - .5, Xm[:, 0].max() + .5, 300),
                     np.linspace(Xm[:, 1].min() - .5, Xm[:, 1].max() + .5, 300))
mz = moon_clf.predict(np.c_[mx.ravel(), my.ravel()]).reshape(mx.shape)

fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(mx, my, mz, alpha=0.25, cmap=ListedColormap(["#c0392b", "#2980b9"]))
ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap=ListedColormap(["#c0392b", "#2980b9"]),
           s=14, edgecolors="white", linewidths=0.3)
ax.set_title("A straight line cannot follow the curve")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
fig
```

[](#b4-moons) shows the limit plainly: a straight fence slices right through both
crescents, misclassifying the parts that curve back. To follow that curve we would
need a model that can *bend* its boundary — which is exactly what neural networks, the
subject of Part C, are built to do.

## Learning is just moving the line

It is worth pausing on what "learning" meant in this chapter. Logistic regression has
a weight for each measurement; those weights are simply *numbers*, and changing them
slides and tilts the boundary. Training the model means searching for the numbers that
place the line where it separates the classes best.

```{code-cell} python
:label: b4-weights
weights = clf.coef_[0]
print(f"Weight on mean radius:  {weights[0]:+.3f}")
print(f"Weight on mean texture: {weights[1]:+.3f}")
print("These two numbers fully determine the slope of the straight boundary above.")
```

That is the whole game, and it is the bridge to everything that follows: a model is a
set of adjustable numbers, and **learning is the search for the numbers that draw the
best boundary**. In [](1_c1-fitting-a-line.md) we start that search from scratch.

## Exercises

```{admonition} Exercise B4.1 — Read the boundary
:class: hint
Looking at [](#b4-boundary), a tumor with large mean radius and large mean texture
falls in which shaded region — and so which label would the model give it? Does that
match what you learned about malignant cells in [](1_b1-knn-tumor.md)?
```

```{admonition} Solution
:class: dropdown
Large radius and large texture sit toward the upper-right of the plot, which is in the
**malignant** (red) region, so the model predicts *malignant*. That fits the biology:
malignant cells tend to be larger and more irregular, so both measurements run high —
the model has rediscovered a real pattern just by placing its line.
```

```{admonition} Exercise B4.2 — Tame the moons
:class: hint
We saw logistic regression score poorly on `make_moons`. Without changing the data,
swap in a `KNeighborsClassifier` (from B1) and report its accuracy on the moons. Why
can it do so much better here?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15).fit(Xm, ym)
print(f"{knn.score(Xm, ym):.1%}")   # well into the 90s
```
k-NN decides by local majority vote, so its boundary can **curve** to hug each
crescent instead of being forced straight. The lesson is not that one model is "better"
— it is that the *shape* a model can draw must match the shape of the data, the very
idea that motivates the flexible models of Part C.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **decision boundary** is where a classifier flips its prediction; with two
  features we can draw it directly with a meshgrid and `contourf`.
- **Logistic regression** weights the measurements into a score, giving a **straight**
  boundary — great when the data is **linearly separable**.
- Curved, interleaved data (like `make_moons`) needs **non-linear** boundaries that a
  straight line cannot provide.
- "Learning" is adjusting a model's numbers until the boundary lands well — the search
  that Part C, beginning with [](1_c1-fitting-a-line.md), makes its central theme.
```
