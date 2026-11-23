---
title: 'Chapter 2.1 — Learning from Similar Patients: k-Nearest Neighbors & Feature Scaling'
short_title: '2.1 · Nearest neighbors'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.1 — Learning from Similar Patients: k-Nearest Neighbors & Feature Scaling

*Estimated time: ~40 minutes · Prerequisites: Part 1*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain how a **classifier** turns patient measurements into a clinical prediction;
- implement **k-nearest neighbors (k-NN)** from scratch in pure NumPy;
- demonstrate why **feature standardization** is essential whenever measurements have different units — and why scaling must be fit on training data only;
- diagnose how choosing $k$ balances local detail against stability.
```

## Start with a case: under the pathologist's microscope

When a patient has a suspicious breast lump, a clinician extracts a tiny sample of cells with a fine needle (a *fine-needle aspirate* biopsy) and photographs the cell nuclei under a microscope. Malignant (cancerous) cell nuclei tend to be larger, more jagged, and less uniform than benign (harmless) ones.

In 1993, physicians and computer scientists at the University of Wisconsin measured 30 geometric features from **569 real breast biopsies** — radius, area, perimeter, smoothness, concavity, texture — and recorded each tumor's confirmed diagnosis. That dataset ships right inside `scikit-learn`.

How could a computer learn to assist the pathologist? Our first algorithm mimics the way doctors actually think when confronting a tricky case: *"Which past patients looked most similar to this one, and what happened to them?"*

```{admonition} Definition — classifier and k-nearest neighbors (k-NN)
:class: important
A {term}`Classifier` is a rule that takes measured features of an example and predicts its category (here, *malignant* vs. *benign*). {term}`k-Nearest neighbors (k-NN)` classifies a new patient by measuring the straight-line distance to every labeled past patient, finding the **$k$ closest neighbors**, and taking a majority vote.
```

:::{margin}
**Convention across TAIHLS:** In `load_breast_cancer()`, `scikit-learn` stores `0 = malignant` and `1 = benign`. In medicine, the disease we are trying to *detect* is always the **positive class (`1`)**, so we flip the labels right away: `1 = malignant`, `0 = benign`.
:::

## Lab 1: Measuring straight-line distance in feature space

To keep things visual, let's pick two real tumor features with dramatically different numerical scales:
- **mean area** (ranges roughly from `140` to `2,500` square micrometers), and
- **worst concave points** (the sharpest inward indentations along the cell nucleus edge — a key cancer hallmark ranging from `0.00` to `0.29`).

For any two tumors $A = (x_1, x_2)$ and $B = (x'_1, x'_2)$, ordinary Pythagorean straight-line distance is:

$$ d(A, B) = \sqrt{(x_1 - x'_1)^2 + (x_2 - x'_2)^2} $$ (eq-21-euclid)

```{admonition} Predict before you run
:class: tip
Suppose Tumor A has `(area = 600, worst concave points = 0.04)` and Tumor B has `(area = 605, worst concave points = 0.24)`. Even though concave points jumped across most of its *entire* biological range (`0.04 → 0.24`), area only changed by `5` units (`600 → 605`). Inside [](#eq-21-euclid), which feature will completely dominate the distance calculation if we don't rescale first?
```

```{code-cell} python
:label: ch21-load-and-scale
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
# Feature 3 = mean area (~143 to 2501), Feature 27 = worst concave points (~0.00 to 0.29)
idx = [3, 27]
X = cancer.data[:, idx]
y = (cancer.target == 0).astype(int)    # 1 = malignant (positive), 0 = benign

print(f"Dataset: {len(y)} tumors | Malignant: {y.sum()} ({y.mean():.1%}) | Benign: {(1-y).sum()}")
for col_i, name in zip([0, 1], [cancer.feature_names[i] for i in idx]):
    print(f"  {name:<22}: min = {X[:, col_i].min():7.3f}, max = {X[:, col_i].max():8.3f}")

# Split BEFORE computing any scaling statistics (crucial anti-leakage habit!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0, stratify=y
)

# Standardize using ONLY training mean and training SD
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train_s = (X_train - train_mean) / train_std
X_test_s = (X_test - train_mean) / train_std
```

Look at those ranges: `mean area` spans over **2,350 units**, while `worst concave points` spans **0.29 units** — a factor of **8,000×**! Without rescaling, `worst concave points` contributes virtually zero to straight-line distance, as if we deleted that column entirely.

```{admonition} Definition — standardization
:class: important
{term}`Standardization` rescales each feature by subtracting its training mean and dividing by its training standard deviation: $z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$. Every standardized feature has center `0` and spread `1`, letting features measured in different units vote fairly in distance calculations.
```

## Lab 2: k-NN from scratch — before vs. after standardization

Let's write k-NN from scratch in six lines of NumPy and compare its held-out test accuracy on **raw** measurements vs. **standardized** measurements!

```{code-cell} python
:label: ch21-knn-scratch
def knn_predict(X_tr, y_tr, X_query, k=7):
    preds = []
    for x in X_query:
        dists = np.sqrt(((X_tr - x) ** 2).sum(axis=1))
        nearest_idx = np.argsort(dists)[:k]
        preds.append(1 if y_tr[nearest_idx].mean() >= 0.5 else 0)
    return np.array(preds)

# Baseline: always guess the most common class in training data (benign = 0)
majority_class = int(y_train.mean() >= 0.5)
baseline_acc = (y_test == majority_class).mean()

pred_raw = knn_predict(X_train, y_train, X_test, k=7)
pred_scaled = knn_predict(X_train_s, y_train, X_test_s, k=7)

print(f"Majority-class baseline accuracy     : {baseline_acc:.1%}")
print(f"k-NN (k=7) on RAW features           : {(pred_raw == y_test).mean():.1%}")
print(f"k-NN (k=7) on STANDARDIZED features  : {(pred_scaled == y_test).mean():.1%}")
```

Look at that jump! Standardizing immediately unlocked the signal hiding in `worst concave points`, lifting test accuracy from **86.5%** to **92.4%** — using just 2 of the 30 available biopsy features and six lines of algebra!

Let's *see* why by plotting the exact 7 neighbors picked for the **same** test tumor before and after standardization.

```{code-cell} python
:label: ch21-knn-fig
import matplotlib.pyplot as plt

BLUE, VERMILION, AMBER = "#0072B2", "#D55E00", "#E69F00"

# Pick a borderline test patient where scaling changes the neighborhood
q_idx = 12
q_raw, q_s = X_test[q_idx], X_test_s[q_idx]

d_raw = np.sqrt(((X_train - q_raw) ** 2).sum(axis=1))
nn_raw = np.argsort(d_raw)[:7]

d_s = np.sqrt(((X_train_s - q_s) ** 2).sum(axis=1))
nn_s = np.argsort(d_s)[:7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

for ax, X_pts, q_pt, nn_idx, title, xl, yl in [
    (ax1, X_train, q_raw, nn_raw,
     "A. Raw units: neighbors chosen by Area ONLY",
     "Mean area (µm²)", "Worst concave points"),
    (ax2, X_train_s, q_s, nn_s,
     "B. Standardized: both Area and Concave Points count",
     "Mean area (SD units)", "Worst concave points (SD units)"),
]:
    for lbl, name, col in [(0, "Benign", BLUE), (1, "Malignant", VERMILION)]:
        m = y_train == lbl
        ax.scatter(X_pts[m, 0], X_pts[m, 1], s=18, color=col, alpha=0.45, label=name)
    ax.scatter(X_pts[nn_idx, 0], X_pts[nn_idx, 1], s=95, facecolors="none",
               edgecolors="#111111", lw=1.6, label="7 nearest neighbors")
    ax.scatter(*q_pt, s=190, marker="*", color=AMBER, edgecolors="#111111",
               lw=1.0, zorder=6, label="New biopsy")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

Look at Panel A of [](#ch21-knn-fig)! Because raw `mean area` is thousands of times wider than `worst concave points`, the 7 circled "nearest" neighbors form a tall vertical strip — all sharing nearly the exact same area, while completely ignoring huge differences in nuclear concavity! In Panel B, after standardization, the 7 neighbors sit in a genuine circular neighborhood around the new biopsy.

## When does k-NN struggle? (The curse of dimensionality)

k-NN is wonderfully transparent, requires zero training loop, and adapts naturally to curved boundaries. Yet it has two major limits in modern biomedicine:
1. **Slow at prediction time:** With 1,000,000 hospital patients, finding distances to everyone for every new scan is sluggish.
2. **The curse of dimensionality:** Imagine measuring 20,000 gene expression levels per cell instead of 2 tumor features. In 20,000 dimensions, almost every pair of cells looks equally far apart, so "nearest neighbors" lose meaning unless we first compress the data into a compact, meaningful representation (an **embedding** — which we will explore in Part 4!).

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.1.1 (Modify) — Sweeping the neighborhood size k
:class: hint
Using `knn_predict` on the **standardized** features (`X_train_s`, `X_test_s`), evaluate test accuracy for `k` in `(1, 3, 7, 25, 101, 301)`. Describe what happens at the two extremes (`k = 1` vs. `k = 301`)!
```

```{admonition} Solution
:class: dropdown
```python
for k in (1, 3, 7, 25, 101, 301):
    preds = knn_predict(X_train_s, y_train, X_test_s, k=k)
    print(f"k = {k:>3}: test accuracy = {(preds == y_test).mean():.1%}")
```

At `k = 1`, a single unusual or mislabeled tumor next door flips the diagnosis (high sensitivity to noise → ~87%). Moderate `k` (7 to 25) averages out individual quirks while staying local (~91–93%). By `k = 301`, almost the entire training set votes, so the majority class (**benign**, ~62.6%) always wins — accuracy collapses right down to the dumb majority-class baseline!
```

```{admonition} Exercise 2.1.2 (Make) — All 30 biopsy measurements at once
:class: hint
Repeat the standardization + `knn_predict(..., k=7)` experiment using **all 30 features** of `cancer.data` instead of just 2. Remember to fit `mean` and `std` on `X_train` only! How much does held-out accuracy improve?
```

```{admonition} Solution
:class: dropdown
```python
X_all = cancer.data
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X_all, y, test_size=0.30, random_state=0, stratify=y
)
mu, sd = X_tr_a.mean(axis=0), X_tr_a.std(axis=0)
X_tr_as = (X_tr_a - mu) / sd
X_te_as = (X_te_a - mu) / sd

preds_30 = knn_predict(X_tr_as, y_tr_a, X_te_as, k=7)
print(f"30-feature standardized k-NN accuracy: {(preds_30 == y_te_a).mean():.1%}")
```

Accuracy climbs from **~91.8%** to roughly **95–96%**! Those extra 28 morphology features (concavity, worst perimeter, fractal dimension) carry substantial extra diagnostic signal — and our exact same 6-line `knn_predict` function worked unchanged in 30 dimensions!
```

```{admonition} Exercise 2.1.3 (Critical Reasoning) — Why standardize using training statistics only?
:class: hint
In [](#ch21-load-and-scale), we computed `train_mean` and `train_std` on `X_train` and applied those numbers to `X_test` — instead of standardizing the entire dataset `X` first and then splitting. Why does standardizing *before* splitting violate scientific honesty?
```

```{admonition} Solution
:class: dropdown
Standardizing the full table before splitting lets information about the **future test patients** (their mean and spread) leak into the training preprocessing — a subtle form of **data leakage**. In real clinical deployment, future patients haven't walked through the hospital door yet, so your preprocessing pipeline can only ever use statistics frozen from past training data. We will expose much more dramatic forms of clinical data leakage in the very next chapter!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **classifier** maps features to categories; **k-NN** classifies a new case by majority vote among its $k$ closest neighbors in feature space.
- Always compare your model against a **majority-class baseline** (here ~62.6%) so you know what real skill looks like.
- **Standardizing** features (using training mean and SD only!) prevents large-numbered units from silencing smaller-numbered measurements.
- The choice of $k$ trades noise sensitivity ($k=1$) against over-smoothing ($k \approx N$).
```

*Next up:* How do we pick settings like $k$ honestly without fooling ourselves — and what hidden mistakes can make a broken clinical AI look 99% accurate on paper? Find out in [](2_2-generalization-and-leakage.md).
