---
title: 'Chapter 2.1 — Learning from Similar Patients: k-Nearest Neighbors'
short_title: '2.1 · Nearest neighbors'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.1 — Learning from Similar Patients: k-Nearest Neighbors

*Week 4 · ~40 minutes · Prerequisites: Unit 1*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **classifier** does using a real breast-tumor **biopsy** dataset;
- calculate the distance between two patients by hand and build **k-nearest neighbors (k-NN)** in plain Python;
- explain why **standardizing** measurements onto a common scale is essential before measuring distance;
- compare your model's score against a simple **majority-class baseline**.
```

## Start with a case: matching a new biopsy to past cases

When a doctor finds a suspicious lump in a patient's breast, they take a tiny sample of cells — called a {term}`Biopsy` — and look at the cell centers (nuclei) under a microscope. Cancerous (**{term}`Malignant`**) cells tend to be larger and have rougher, more indented edges than harmless (**{term}`Benign`**) cells.

Suppose you want a computer to help flag suspicious biopsies. Where do you even begin? Surprisingly, our very first AI {term}`Classifier` works just like a doctor recalling past patients:

> **Find the $k$ past tumors whose measurements are closest to this new tumor, and let them vote!**

```{admonition} Definition — k-nearest neighbors (k-NN)
:class: important
{term}`k-Nearest neighbors (k-NN)` classifies a new example by measuring the straight-line distance to every already-labeled training example, picking the **$k$ closest neighbors**, and choosing whichever category wins the majority vote.
```

## Measure distance by hand first!

Remember the Pythagorean theorem from algebra? If Tumor A has `(size = 2, roughness = 1)` and Tumor B has `(size = 5, roughness = 5)`:

$$ \text{distance} = \sqrt{(5 - 2)^2 + (5 - 1)^2} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5} $$ (eq-21-dist)

Now notice a sneaky trap! In our real Wisconsin Breast Cancer dataset (**569 real biopsies** built right into `scikit-learn`), let's look at two measurements:
- **mean area** (how big the cell nucleus is): runs from **`143` to `2,501`**!
- **worst concave points** (how sharply indented the nucleus edge is): runs from **`0.00` to `0.29`**!

If Tumor A has `(area = 600, indent = 0.04)` and Tumor B has `(area = 605, indent = 0.24)`:
- Area changed by just a tiny sliver (`5` units out of `2,350`), contributing $5^2 = \mathbf{25}$.
- Edge indentation jumped almost across its *entire* range (`0.20`), contributing $0.20^2 = \mathbf{0.04}$!

Because area uses much bigger numbers, **area shouts over edge indentation** unless we put both columns on the same yardstick first!

```{admonition} Definition — standardization
:class: important
{term}`Standardization` rescales each column by subtracting its training average and dividing by its training spread (standard deviation): $z = (x - \text{mean}) / \text{std}$. Now every measurement sits on the same scale (average `0`, typical spread `1`).
```

:::{margin}
**Label convention in TAIHLS:** We always label the disease we want to catch as **`1` (malignant)** and harmless cases as **`0` (benign)**. (Since `scikit-learn` stores the reverse, we flip it on line 7 below.)
:::

## Build k-NN from scratch in Python!

Let's load the 569 real tumors, split them into **training** (past cases to learn from) and **test** (new cases to grade on), and compare k-NN **before** and **after** standardizing!

```{admonition} Predict before you run
:class: tip
In this dataset, about **63%** of biopsies are benign (`0`).
1. What accuracy would a "lazy" rule get by simply guessing **"benign" for every single patient**?
2. Will standardizing `mean area` and `worst concave points` improve k-NN's test accuracy?
```

```{code-cell} python
:label: ch21-knn-code
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = cancer.data[:, [3, 27]]            # column 3 = mean area, column 27 = worst concave points
y = (cancer.target == 0).astype(int)   # 1 = malignant (cancer), 0 = benign (harmless)

# Split into 70% past training cases and 30% held-out test cases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0, stratify=y
)

# Standardize using ONLY the training set's mean and spread!
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train_s = (X_train - mean) / std
X_test_s  = (X_test - mean) / std

def predict_knn(X_past, y_past, X_new, k=7):
    predictions = []
    for tumor in X_new:
        # Straight-line distance from this new tumor to every past tumor
        dists = np.sqrt(((X_past - tumor) ** 2).sum(axis=1))
        closest_k = np.argsort(dists)[:k]
        vote = y_past[closest_k].mean()        # fraction of k neighbors that are malignant
        predictions.append(1 if vote >= 0.5 else 0)
    return np.array(predictions)

baseline = (y_test == 0).mean()                # always guessing "benign"
acc_raw  = (predict_knn(X_train,   y_train, X_test,   k=7) == y_test).mean()
acc_std  = (predict_knn(X_train_s, y_train, X_test_s, k=7) == y_test).mean()

print(f"Always-guess-benign baseline : {baseline:.1%}")
print(f"k-NN on RAW numbers          : {acc_raw:.1%}")
print(f"k-NN on STANDARDIZED numbers : {acc_std:.1%}  <-- big jump once both features count!")
```

Look at that leap: standardizing lifted held-out test accuracy from **86.5%** to **92.4%** — using just **two** measurements and a 6-line function you can read in one breath!

## See why rescaling worked

Let's plot one test tumor (the yellow star ★) and circle its **7 nearest neighbors** before and after standardization.

```{code-cell} python
:label: ch21-knn-plot
import matplotlib.pyplot as plt

q = 12   # pick one borderline test tumor to inspect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

for ax, X_pts, q_pt, title, xl, yl in [
    (ax1, X_train,   X_test[q],   "A. Raw numbers (Area dominates!)",
     "Mean area", "Worst concave points"),
    (ax2, X_train_s, X_test_s[q], "B. Standardized (Both features count)",
     "Mean area (std units)", "Worst concave points (std units)"),
]:
    dists = np.sqrt(((X_pts - q_pt) ** 2).sum(axis=1))
    nn = np.argsort(dists)[:7]

    ax.scatter(X_pts[y_train == 0, 0], X_pts[y_train == 0, 1], s=16, color="#0072B2", alpha=0.45, label="Benign")
    ax.scatter(X_pts[y_train == 1, 0], X_pts[y_train == 1, 1], s=16, color="#D55E00", alpha=0.45, label="Malignant")
    ax.scatter(X_pts[nn, 0], X_pts[nn, 1], s=90, facecolors="none", edgecolors="black", lw=1.5, label="7 neighbors")
    ax.scatter(q_pt[0], q_pt[1], s=180, marker="*", color="#E69F00", edgecolors="black", zorder=5, label="New biopsy")
    ax.set_title(title)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

fig.tight_layout()
plt.show()
```

Look at Panel A! On raw numbers, the 7 circled neighbors line up in a tall vertical column — matching area closely while totally ignoring how rough the edges are! In Panel B, the 7 neighbors sit in a genuine tight circle around the new biopsy.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.1.1 (Modify) — Does the choice of k matter?
:class: hint
Test `predict_knn(X_train_s, y_train, X_test_s, k=k)` for `k` in `[1, 3, 7, 25, 301]`. What happens when `k = 1`? What happens when `k = 301`, and what number does that score remind you of?
```

```{admonition} Solution
:class: dropdown
```python
for k in [1, 3, 7, 25, 301]:
    acc = (predict_knn(X_train_s, y_train, X_test_s, k=k) == y_test).mean()
    print(f"k = {k:>3}: test accuracy = {acc:.1%}")
```
At `k = 1`, a single unusual neighbor next door can flip the answer. Moderate `k` (`7`–`25`) works best (`~92%`). At `k = 301`, almost the entire training set votes, so the majority class (**benign**, `62.6%`) wins every single time — collapsing right back to the lazy baseline!
```

```{admonition} Exercise 2.1.2 (Make) — Try all 30 tumor measurements
:class: hint
Change `X = cancer.data[:, [3, 27]]` to `X = cancer.data` (all 30 columns!), split and standardize on training data, and run `predict_knn(..., k=7)`. Does giving k-NN all 30 measurements improve accuracy further?
```

```{admonition} Solution
:class: dropdown
```python
X_tr30, X_te30, y_tr30, y_te30 = train_test_split(
    cancer.data, y, test_size=0.30, random_state=0, stratify=y
)
m30, s30 = X_tr30.mean(axis=0), X_tr30.std(axis=0)
preds30 = predict_knn((X_tr30 - m30) / s30, y_tr30, (X_te30 - m30) / s30, k=7)
print(f"All 30 features standardized k-NN accuracy: {(preds30 == y_te30).mean():.1%}")
```
Accuracy climbs to roughly **95–96%** — and our exact same 6-line `predict_knn` function worked unchanged!
```

```{admonition} Exercise 2.1.3 (Critical Reasoning) — Why standardize using training data only?
:class: hint
In [](#ch21-knn-code), we computed `mean` and `std` using `X_train` and applied those numbers to `X_test`. Why didn't we compute the mean and spread across the entire dataset `X` before splitting?
```

```{admonition} Solution
:class: dropdown
In a real hospital, future test patients haven't walked through the door yet — you cannot peek at their averages ahead of time! Freezing the scaling numbers on `X_train` keeps the test exam 100% honest (preventing a sneaky mistake called **data leakage**, which we'll explore in Chapter 2.3).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **classifier** turns measurements into categories; **k-NN** decides by majority vote of the $k$ closest past examples.
- Always compare your model to a **majority-class baseline** (here `62.6%`).
- **Standardizing** features (using training mean and spread only) stops big-numbered columns from drowning out small-numbered ones.
- Choosing $k$ too small reacts to noise; choosing $k$ too large washes out the pattern.
```

*Next up:* How do we pick settings like $k$ — or know whether a model truly learned biology instead of just memorizing the training set? Let's find out in [](2_2-train-test-and-overfitting.md)!
