---
title: 'Chapter 2.4 — From Lines to Forests: Decision Boundaries, Ensembles, and Interpretability'
short_title: '2.4 · Trees, forests & boundaries'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.4 — From Lines to Forests: Decision Boundaries, Ensembles, and Interpretability

*Estimated time: ~45 minutes · Prerequisites: [](2_3-clinical-metrics-and-thresholds.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- visualize and contrast the **decision boundaries** of **logistic regression**, a **decision tree**, and a **random forest**;
- explain the **ensemble** principle (*wisdom of crowds*) behind **random forests** and **gradient-boosted trees**;
- inspect **permutation feature importance** to see *which* clinical measurements drive a model's predictions;
- recognize when classical tabular models shine — and when we need deep neural networks (Part 3).
```

## Start with a case: three doctors, three ways of thinking

Watch three experienced clinicians evaluate a suspicious biopsy:
- **Clinician 1 (Weighted checklist):** Adds up points — *"plus 2 points for large radius, plus 1.5 for irregular texture, minus 1 for smooth border — if the total passes the line, biopsy."* That is **logistic regression**.
- **Clinician 2 (Flowchart):** Follows strict cutoffs — *"Is worst radius > 16.8? If yes, check worst concave points > 0.14..."* That is a **decision tree**.
- **Clinician 3 (Tumor board):** Convenes **100 independent specialists**, each looking at a slightly different sample of past cases and subsets of measurements, and averages their probability estimates. Individual quirks cancel out, while consistent cancer signals reinforce one another. That is a {term}`Random forest`!

In this chapter, we put all three families side by side on the exact same real breast-tumor measurements so you can *see* the geometric shapes each one can draw.

## Seeing the geometry: decision boundaries

```{admonition} Definition — decision boundary and ensemble
:class: important
A {term}`Decision boundary` is the dividing line, staircase, or curve in feature space where a classifier's predicted probability crosses the decision threshold (`0.50`). An {term}`Ensemble` combines many diverse models (such as 200 decision trees in a **random forest**) so individual errors cancel out, dramatically reducing overfitting.
```

## Lab 1: Three classifiers, one clinical plane

Let's plot the learned decision boundaries of **Logistic Regression**, a **Single Decision Tree (depth 5)**, and a **Random Forest (200 trees)** on two standardized biopsy measurements: `mean radius` and `mean texture`.

```{admonition} Predict before you run
:class: tip
Before running the cell, picture the exact **shape** of the red/blue boundary each model will carve:
1. Logistic regression → straight diagonal line, staircase, or smooth curve?
2. Single decision tree → straight diagonal line, horizontal/vertical rectangles, or circle?
3. Random forest (average of 200 trees) → sharper rectangles or smoother blended boundary?
```

```{code-cell} python
:label: ch24-boundaries-fig
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BLUE, VERMILION = "#0072B2", "#D55E00"

cancer = load_breast_cancer()
X2 = cancer.data[:, [0, 1]]                       # mean radius, mean texture
y = (cancer.target == 0).astype(int)              # 1 = malignant

X_tr, X_te, y_tr, y_te = train_test_split(X2, y, test_size=0.30, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

models = [
    ("A. Logistic Regression\n(linear weighted sum)", LogisticRegression()),
    ("B. Single Decision Tree\n(axis-aligned splits, depth 5)", DecisionTreeClassifier(max_depth=5, random_state=0)),
    ("C. Random Forest\n(ensemble of 200 trees)", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)),
]

# Fine grid covering standardized feature space
xx, yy = np.meshgrid(np.linspace(-2.6, 3.8, 240), np.linspace(-2.6, 3.8, 240))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
for ax, (title, clf) in zip(axes, models):
    clf.fit(X_tr_s, y_tr)
    acc = clf.score(X_te_s, y_te)
    zz = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.55)
    ax.contour(xx, yy, zz, levels=[0.5], colors=["#111111"], linewidths=1.8)
    for lbl, col, name in [(0, BLUE, "Benign"), (1, VERMILION, "Malignant")]:
        m = y_te == lbl
        ax.scatter(X_te_s[m, 0], X_te_s[m, 1], s=18, color=col,
                   edgecolors="white", lw=0.4, label=name)
    ax.set_title(f"{title}\nTest acc = {acc:.1%}", fontsize=9.8)
    ax.set_xlabel("Mean radius (standardized)")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Mean texture (standardized)")
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
plt.show()
```

Look at those three black 0.50 contour lines in [](#ch24-boundaries-fig):
- **Panel A (Logistic Regression):** One crisp straight diagonal — easy to interpret (*"larger radius and rougher texture both push risk upward"*), impossible to bend if classes curl around each other.
- **Panel B (Single Decision Tree):** Rectangular horizontal and vertical staircases! Each split asks about one feature at a time — occasionally carving narrow, brittle pockets around isolated training points.
- **Panel C (Random Forest):** Averaging 200 diverse trees smooths away the single tree's jagged pockets while still gently curving to follow the clouds!

## Lab 2: Benchmark on all 30 clinical features + Interpretability

Real electronic health records (EHRs) have dozens or hundreds of tabular columns: lab panels, vitals, age, medications. On structured clinical tables, **tree ensembles** (Random Forests and **Gradient-Boosted Decision Trees**, where each new tree specifically corrects the errors of the previous trees) often rival or outperform deep neural networks!

Just as importantly, clinicians need to know **which features mattered**. One clean, model-agnostic tool is **permutation feature importance**:
1. Record the model's test score (say, AUROC).
2. Randomly **shuffle** one column (e.g., `worst radius`) across test patients — destroying its relationship with cancer while keeping its distribution intact — and measure how much the score drops!
3. If shuffling a column wrecks accuracy, the model leaned heavily on it; if accuracy barely budges, that feature was redundant or unused.

```{code-cell} python
:label: ch24-benchmark-importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

X_all = cancer.data
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X_all, y, test_size=0.30, random_state=0, stratify=y
)

suite = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree (d=4)": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Random Forest (200)": RandomForestClassifier(n_estimators=200, random_state=0),
    "Gradient Boosted Trees": HistGradientBoostingClassifier(max_depth=4, random_state=0),
}

print(f"{'Model':<24} | {'Test Acc':>8} | {'Test AUROC':>10}")
print("-" * 48)
fitted = {}
for name, clf in suite.items():
    clf.fit(X_tr_a, y_tr_a)
    fitted[name] = clf
    acc = clf.score(X_te_a, y_te_a)
    auroc = roc_auc_score(y_te_a, clf.predict_proba(X_te_a)[:, 1])
    print(f"{name:<24} | {acc:8.1%} | {auroc:10.3f}")

# Inspect which clinical measurements matter most to the Random Forest
perm = permutation_importance(
    fitted["Random Forest (200)"], X_te_a, y_te_a,
    n_repeats=15, random_state=0, scoring="roc_auc"
)
top_idx = np.argsort(perm.importances_mean)[-8:]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.barh(cancer.feature_names[top_idx], perm.importances_mean[top_idx],
        xerr=perm.importances_std[top_idx], color="#009E73", alpha=0.85, capsize=3)
ax.set_xlabel("Drop in test AUROC when feature is shuffled")
ax.set_title("Top 8 biopsy measurements driving the Random Forest")
ax.grid(axis="x", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

:::{margin}
**Important caveat on correlated features:** Notice that `worst area`, `worst radius`, and `worst perimeter` all measure tumor size! When you shuffle *one* of them, the other two can partially substitute for it — so permutation importance shows *unique* impact, not biological irrelevance.
:::

Look at how close the simple linear model (`Logistic Regression`) comes to the complex ensembles on this clean 30-feature table! **Always start with a simple baseline** before reaching for complex models — a principle that holds from tiny clinical tables all the way up to billion-parameter single-cell foundation models [@ahlmann2025perturbation].

## When tabular models hit their ceiling

Tree ensembles and logistic regression are phenomenal when a scientist has *already* extracted meaningful tabular features (`mean radius`, `blood pressure`, `hemoglobin`).

But what if your raw input is:
- a **1,000 × 1,000 pixel chest X-ray** (1,000,000 pixels — no pre-measured "pneumonia radius" column!),
- a **free-text clinical note** written by an ER doctor, or
- a **300-letter amino-acid sequence** whose 3D folded shape determines whether a drug binds?

That is where **representation learning** — **deep neural networks** — enters the stage.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.4.1 (Modify) — Watching ensemble stability grow
:class: hint
Train `RandomForestClassifier(n_estimators=n, random_state=0)` on `X_tr_a, y_tr_a` for `n` in `(1, 5, 25, 200)` and print test accuracy and test AUROC for each. How fast do the gains appear as you move from 1 tree to a forest?
```

```{admonition} Solution
:class: dropdown
```python
for n in (1, 5, 25, 200):
    rf = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_tr_a, y_tr_a)
    auroc = roc_auc_score(y_te_a, rf.predict_proba(X_te_a)[:, 1])
    print(f"n_estimators = {n:>3} | acc = {rf.score(X_te_a, y_te_a):.1%} | AUROC = {auroc:.3f}")
```

Going from 1 tree to just 25 trees produces a massive jump in both accuracy and AUROC, because averaging independent trees cuts prediction variance sharply; going from 25 to 200 polishes stability further with diminishing returns.
```

```{admonition} Exercise 2.4.2 (Make) — Reading the clinical weights of Logistic Regression
:class: hint
Extract the learned coefficients from `fitted["Logistic Regression"].named_steps["logisticregression"].coef_[0]` and print the **top 5 features with the largest positive weights** (pushing risk toward *malignant*). Because the features were standardized first, bigger positive weight directly means stronger push toward malignancy. Do they make biological sense?
```

```{admonition} Solution
:class: dropdown
```python
lr = fitted["Logistic Regression"].named_steps["logisticregression"]
coefs = lr.coef_[0]
top5 = np.argsort(coefs)[-5:][::-1]
for i in top5:
    print(f"{cancer.feature_names[i]:<24}: weight = {coefs[i]:+.3f}")
```

Features such as `worst texture`, `radius error`, `worst concave points`, and `worst symmetry` sit right at the top — matching exactly what pathologists look for under a microscope (irregular, rough, asymmetric cell nuclei)!
```

```{admonition} Exercise 2.4.3 (Critical Reasoning) — Choosing a model for an ICU triage tool
:class: hint
An intensive-care team builds two sepsis risk models on 25 routine lab and vital-sign columns:
- Model A (Logistic Regression): `AUROC = 0.87`, completely transparent weights, easy to audit.
- Model B (Deep Neural Network): `AUROC = 0.88`, opaque black box, harder to calibrate and monitor.

Which model would you recommend deploying first, and what extra evidence would Model B need to justify the added complexity?
```

```{admonition} Solution
:class: dropdown
Most clinical safety experts would start with **Model A**: a tiny 0.01 AUROC gain on retrospective data rarely translates into measurable bedside benefit, whereas transparency makes it vastly easier to catch data bugs, audit calibration, explain alerts to nurses, and spot shortcut features. To justify Model B, you would want to see a **meaningful improvement on external validation** across multiple hospitals (including higher AUPRC and well-verified calibration) — and ideally a prospective silent-mode or randomized clinical study proving real patient benefit.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Logistic regression** draws a straight hyperplane; **decision trees** draw axis-aligned boxes; **random forests** average many diverse trees to smooth variance.
- **Tree ensembles** (Random Forests and Gradient-Boosted Trees) remain powerhouse models for structured tabular clinical data.
- **Permutation importance** measures how much test performance drops when a feature is scrambled — a practical window into what a model relies on.
- Tabular models shine on engineered columns, but raw images, sequences, and text require neural networks that learn features automatically.
```

*End of Part 2!* In **Part 3**, we open the black box of **deep learning** — starting with how *every* neural network learns its parameters by rolling downhill on a loss landscape, with zero calculus required, in [](3_1-loss-and-gradient-descent.md).
