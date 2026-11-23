---
title: 'Chapter 2.5 — Three Clinical Models: Straight Lines, Trees, and Forests'
short_title: '2.5 · Lines, trees & forests'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.5 — Three Clinical Models: Straight Lines, Trees, and Forests

*Week 6 · ~40 minutes · Prerequisites: [](2_4-mistakes-and-thresholds.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- describe how **logistic regression**, a **decision tree**, and a **random forest** each make decisions in plain words;
- read a **decision boundary** plot to see the difference between a straight cut, boxy steps, and an averaged ensemble;
- explain the **ensemble** (*wisdom of crowds*) idea behind random forests;
- check which clinical features a model relied on most — and notice when a simple model beats a fancy one!
```

## Start with a case: three ways doctors reach a diagnosis

Imagine three doctors reviewing the same breast biopsy measurements:
- **Doctor 1 (Weighted checklist):** Gives points to each measurement (*"+2 points for big radius, +1.5 for rough texture"*) and adds them up. That is {term}`Logistic regression`.
- **Doctor 2 (Flowchart):** Follows strict yes/no steps (*"Is radius > 15? If yes, check texture..."*). That is a **decision tree**.
- **Doctor 3 (Medical board):** Asks **200 independent decision trees** — each trained on a slightly different random sample of past patients — and averages their votes! Individual quirks cancel out while strong cancer clues reinforce each other. That is a {term}`Random forest` (a classic {term}`Ensemble`).

## Seeing where each model draws the line

When a classifier uses just two measurements (say, `mean radius` and `mean texture`), we can color every spot on the x–y graph by what the model predicts. The border where blue turns to orange is the model's {term}`Decision boundary`.

```{admonition} Predict before you run
:class: tip
Before running the cell below, match each model to its boundary shape:
1. **Logistic Regression** → straight line, boxy staircase, or smooth blended border?
2. **Single Decision Tree** → straight line, boxy staircase, or smooth blended border?
3. **Random Forest (200 trees)** → sharper boxes or smoother blended border?
```

```{code-cell} python
:label: ch25-three-boundaries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay

cancer = load_breast_cancer()
X2 = cancer.data[:, [0, 1]]              # mean radius and mean texture
y  = (cancer.target == 0).astype(int)    # 1 = malignant

X_tr, X_te, y_tr, y_te = train_test_split(X2, y, test_size=0.30, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

models = [
    ("A. Logistic Regression\n(straight line)", LogisticRegression()),
    ("B. Single Decision Tree\n(boxy flowchart)", DecisionTreeClassifier(max_depth=5, random_state=0)),
    ("C. Random Forest\n(200 trees voting)", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)),
]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True, sharey=True)
for ax, (title, clf) in zip(axes, models):
    clf.fit(X_tr_s, y_tr)
    acc = clf.score(X_te_s, y_te)
    DecisionBoundaryDisplay.from_estimator(
        clf, X_tr_s, response_method="predict_proba", cmap="RdBu_r", alpha=0.5, ax=ax
    )
    ax.scatter(X_te_s[y_te == 0, 0], X_te_s[y_te == 0, 1], s=16, color="#0072B2", edgecolors="white", lw=0.4, label="Benign")
    ax.scatter(X_te_s[y_te == 1, 0], X_te_s[y_te == 1, 1], s=16, color="#D55E00", edgecolors="white", lw=0.4, label="Malignant")
    ax.set_title(f"{title}\nTest acc = {acc:.1%}", fontsize=9.8)
    ax.set_xlabel("Mean radius (std)")

axes[0].set_ylabel("Mean texture (std)")
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
plt.show()
```

:::{margin}
Notice `DecisionBoundaryDisplay.from_estimator(...)` — `scikit-learn` draws the shaded decision boundary for us in a single readable line!
:::

Look at those three panels:
- **Panel A (Logistic Regression):** One clean straight cut — super easy to explain, though it cannot bend if classes curve around each other.
- **Panel B (Single Decision Tree):** Crisp horizontal and vertical boxes! Notice how a few narrow stripes try to chase single odd training points.
- **Panel C (Random Forest):** Averaging 200 trees smooths out those jagged stripes while gently curving around the clouds!

## All 30 measurements — and which clues mattered most?

Now let's test all three models on **all 30 biopsy measurements** and ask the Random Forest which measurements helped it split malignant from benign tumors the most!

```{code-cell} python
:label: ch25-benchmark-30
X_tr30, X_te30, y_tr30, y_te30 = train_test_split(cancer.data, y, test_size=0.30, random_state=0, stratify=y)
sc30 = StandardScaler().fit(X_tr30)

lr = LogisticRegression(max_iter=1000).fit(sc30.transform(X_tr30), y_tr30)
dt = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr30, y_tr30)
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr30, y_tr30)

print(f"Logistic Regression (30 features): {lr.score(sc30.transform(X_te30), y_te30):.1%}")
print(f"Single Decision Tree (depth 5)   : {dt.score(X_te30, y_te30):.1%}")
print(f"Random Forest (200 trees)        : {rf.score(X_te30, y_te30):.1%}")

# Show the top 6 most useful biopsy measurements according to the Random Forest
top6 = np.argsort(rf.feature_importances_)[-6:]
plt.figure(figsize=(7.0, 3.6))
plt.barh(cancer.feature_names[top6], rf.feature_importances_[top6] * 100, color="#009E73")
plt.xlabel("Relative importance (%)")
plt.title("Top 6 biopsy measurements driving the Random Forest")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
```

Notice two huge practical lessons right there:
1. **Simple models are mighty!** Plain Logistic Regression (`~95%`) actually edges out or ties the 200-tree Random Forest (`~94%`) on this clean medical table. *Never assume fancier automatically means better!*
2. **Checking feature importance** confirms the model is relying on genuine biology (`worst perimeter`, `worst radius`, `worst concave points`, `worst area`) rather than random noise.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.5.1 (Modify) — Watch the wisdom of crowds kick in
:class: hint
Train `RandomForestClassifier(n_estimators=n, random_state=0)` on `X_tr30, y_tr30` for `n` in `[1, 5, 25, 200]` and print test accuracy for each. How quickly does combining trees improve over a single tree?
```

```{admonition} Solution
:class: dropdown
```python
for n in [1, 5, 25, 200]:
    forest = RandomForestClassifier(n_estimators=n, random_state=0).fit(X_tr30, y_tr30)
    print(f"{n:>3} trees: test accuracy = {forest.score(X_te30, y_te30):.1%}")
```
Moving from 1 tree to 25 trees gives a big jump in stability and accuracy, because individual trees' mistakes cancel out when averaged!
```

```{admonition} Exercise 2.5.2 (Make) — Read Logistic Regression's top weights
:class: hint
Since `lr` was trained on standardized features, larger positive weights in `lr.coef_[0]` push risk more strongly toward **malignant**. Print the top 3 feature names with the largest positive weights!
```

```{admonition} Solution
:class: dropdown
```python
top3 = np.argsort(lr.coef_[0])[-3:][::-1]
for i in top3:
    print(f"{cancer.feature_names[i]:<24}: weight = {lr.coef_[0][i]:+.2f}")
```
Features like `worst texture`, `radius error`, and `worst symmetry` sit at the top — telling doctors *exactly* which measurements raised the risk score!
```

```{admonition} Exercise 2.5.3 (Critical Reasoning) — When aren't tables enough?
:class: hint
All of Unit 2 worked on neat tables where doctors had *already* measured tumor radius, texture, and area for us. What kind of medical data arrives as **raw pictures, raw DNA strings, or doctor notes** with no pre-measured table?
```

```{admonition} Solution
:class: dropdown
Chest X-rays, microscope slides, skin photos, genome sequences, and typed clinical notes have no pre-built "tumor radius" column! Teaching computers to discover useful patterns *directly* from raw pixels and letters is the exact job of **neural networks** (Unit 3) and **sequence/language models** (Unit 4).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Logistic regression** draws a straight decision boundary; **decision trees** draw boxy flowchart splits; **random forests** average many trees for steadier predictions.
- Always test a simple baseline model first — on structured clinical tables, simple models often match complex ones!
- Checking **feature importance** or weights helps doctors verify whether a model uses real medical clues.
```

*End of Unit 2!* You now have a complete classical machine-learning toolkit. In **Unit 3**, we open the hood of **deep learning** — starting with how computers measure mistakes using **loss** in [](3_1-fitting-a-line-and-loss.md)!
