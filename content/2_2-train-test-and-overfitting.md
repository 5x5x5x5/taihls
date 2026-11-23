---
title: 'Chapter 2.2 — Grading Honestly: Training Sets, Test Sets, and Overfitting'
short_title: '2.2 · Overfitting & validation'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.2 — Grading Honestly: Training Sets, Test Sets, and Overfitting

*Week 4 · ~40 minutes · Prerequisites: [](2_1-nearest-neighbors.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why grading a model on its own **training set** measures memorization, not skill;
- describe how a **decision tree** works like a medical flowchart with a complexity knob (`max_depth`);
- recognize **underfitting** and **overfitting** from a training-vs-validation graph;
- explain why we keep a **validation set** (or **cross-validation**) separate from the final locked **test set**.
```

## Start with a case: studying vs. memorizing the answer key

Imagine two students preparing for a biology exam:
- **Student A** memorizes the exact word-for-word answers to the 20 practice questions without understanding the concepts.
- **Student B** learns the underlying biological principles.

If the teacher grades both students on those **exact same 20 practice questions**, Student A scores **100%**! Only when the teacher hands out **brand-new exam questions** do we discover who actually learned and who just memorized.

AI models behave the exact same way. The skill we care about in medicine is {term}`Generalization` — performing well on **new patients** the model has never seen before.

```{admonition} Definition — training set, validation set, and overfitting
:class: important
- The {term}`Training set` is the practice data the model learns from.
- A {term}`Validation set` is a practice exam used to pick model settings (like $k$ or tree depth), while the final **test set** stays locked in a vault until the very end.
- {term}`Overfitting` happens when a model grows so complicated that it memorizes random quirks in the training set — driving training accuracy up to 100% while accuracy on new patients drops!
```

## Meet the Decision Tree (a medical flowchart)

To watch overfitting happen right in front of our eyes, let's meet a {term}`Decision tree`: a model that asks a chain of simple yes/no questions — just like a nurse's triage checklist (*"Is worst radius above 16.8? If yes, is worst concave points above 0.14?"*).

A decision tree has a super intuitive complexity knob: **`max_depth`** — how many yes/no questions in a row it is allowed to ask!
- **`max_depth = 1`:** Only 1 question allowed (too simple → **underfitting**).
- **`max_depth = 12`:** Up to 12 stacked questions — enough to carve a private tiny rule for almost every single training patient (**overfitting**!).

```{mermaid}
flowchart LR
    All["All 569 Biopsies"] --> Dev["Development Data (80%)<br/>Used for training &<br/>5-fold cross-validation"]
    All --> Vault["Locked Test Vault (20%)<br/>Opened ONCE at the end!"]
    Dev --> Best["Pick Best Depth"]
    Best --> Vault
```

:::{margin}
**What is 5-fold {term}`Cross-validation`?** Instead of relying on one single validation split, we chop the development data into 5 equal pieces, train on 4, grade on the 5th, and rotate 5 times — averaging the 5 grades for a much steadier score!
:::

## Watch overfitting happen in code

Let's train decision trees of depths `1` through `12` on all 30 Wisconsin breast-cancer measurements and record **two grades** for every depth: its score on the training biopsies it studied, and its 5-fold cross-validation score on hidden biopsies!

```{admonition} Predict before you run
:class: tip
As `max_depth` climbs from `1` up to `12`:
1. Will **training accuracy** keep climbing all the way to `100%`?
2. Will **validation (hidden) accuracy** also climb to `100%`, or peak early and start sagging?
```

```{code-cell} python
:label: ch22-overfit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)    # 1 = malignant

# Lock 20% in the final test vault before doing ANY tuning!
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y
)

depths = list(range(1, 13))
train_acc = []
val_acc = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0)
    tree.fit(X_dev, y_dev)
    train_acc.append(tree.score(X_dev, y_dev) * 100)
    val_acc.append(cross_val_score(tree, X_dev, y_dev, cv=5).mean() * 100)

best_depth = depths[int(np.argmax(val_acc))]
final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=0).fit(X_dev, y_dev)

print(f"Best depth on 5-fold validation : depth = {best_depth} (val = {max(val_acc):.1f}%)")
print(f"Overfit depth-12 tree           : train = {train_acc[-1]:.1f}%, val = {val_acc[-1]:.1f}%")
print(f"Final grade on locked test vault: {final_tree.score(X_test, y_test):.1%}")

plt.figure(figsize=(7.2, 4.0))
plt.plot(depths, train_acc, "o-", color="#0072B2", lw=2.2, label="Training accuracy (seen data)")
plt.plot(depths, val_acc,   "s-", color="#D55E00", lw=2.4, label="5-fold validation accuracy (unseen data)")
plt.axvline(best_depth, color="#4D4D4D", ls="--", label=f"Best depth = {best_depth}")
plt.xlabel("Tree depth (number of yes/no questions)")
plt.ylabel("Accuracy (%)")
plt.title("Memorizing vs. generalizing as a decision tree grows deeper")
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.show()
```

Look at that widening gap between the blue and orange lines!
- At `depth = 1`, one single yes/no question is too simple (**underfitting**).
- Around `depth = 4–5`, validation accuracy peaks cleanly (`~93%`), and our locked test vault confirms it (`~94%`)!
- Past `depth = 6`, the blue training line marches straight to **`100.0%`** while the orange unseen-data line drifts downward — the unmistakable fingerprint of **overfitting**!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.2.1 (Modify) — Inspect the actual questions your tree asks!
:class: hint
Train a tiny, super-readable tree with `max_depth=2`: `small_tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_dev, y_dev)`. Print its exact flowchart rules using `from sklearn.tree import export_text; print(export_text(small_tree, feature_names=list(cancer.feature_names)))`. Can a doctor read and understand that tree in 10 seconds?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.tree import export_text
small_tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_dev, y_dev)
print(export_text(small_tree, feature_names=list(cancer.feature_names)))
print(f"2-question tree test accuracy: {small_tree.score(X_test, y_test):.1%}")
```
Just **two** simple threshold questions (on `worst perimeter` and `worst concave points`) already get over **90%** of held-out biopsies right — and any clinician can read the entire rule at a glance!
```

```{admonition} Exercise 2.2.2 (Make) — Tune k-NN honestly with 5-fold cross-validation
:class: hint
Use `KNeighborsClassifier` from `sklearn.neighbors` and `cross_val_score` on standardized `X_dev` to test `k` in `[1, 3, 5, 11, 21]`. Which `k` gets the best 5-fold validation score?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

for k in [1, 3, 5, 11, 21]:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    score = cross_val_score(model, X_dev, y_dev, cv=5).mean()
    print(f"k = {k:>2}: 5-fold CV accuracy = {score:.1%}")
```
Moderate values (`k = 3` to `11`) score around **96–97%**, comfortably beating `k = 1`!
```

```{admonition} Exercise 2.2.3 (Critical Reasoning) — Why not pick the best depth directly on the test set?
:class: hint
Suppose a student tries **50 different models** on the final test set and reports only the single highest test score (`98.2%`) in their science-fair poster. Why is that score likely over-optimistic?
```

```{admonition} Solution
:class: dropdown
If you try 50 models on the exact same test set, one of them will get lucky on those particular test patients purely by chance! Once you use a test set to *choose* a model, it stops being an unseen exam. That is why we tune with **validation / cross-validation** and touch the locked test set **once** at the very end.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Never grade a model on its own **training set** — only unseen data measures true **generalization**.
- **Overfitting** shows up when training accuracy approaches 100% while unseen validation accuracy stops improving or drops.
- Use **validation / cross-validation** to tune complexity knobs (like tree depth), keeping the final **test set** locked away until the end.
```

*Next up:* Even when scientists *think* they used a clean train/test split, one sneaky bug has ruined hundreds of published medical AI studies. Let's catch **data leakage** red-handed in [](2_3-data-leakage.md)!
