---
title: 'Chapter 2.4 — The Cost of Being Wrong: False Alarms, Missed Cases, and Thresholds'
short_title: '2.4 · Mistakes & thresholds'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.4 — The Cost of Being Wrong: False Alarms, Missed Cases, and Thresholds

*Week 5 · ~45 minutes · Prerequisites: [](1_5-base-rates-and-uncertainty.md), [](2_3-data-leakage.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- read and build a 2×2 **confusion matrix** (TP, TN, FP, FN) by hand and in Python;
- explain **sensitivity**, **specificity**, and **precision** in everyday words;
- explain why a classifier outputs a **risk percentage (0%–100%)** and how sliding the **decision threshold** trades false alarms against missed cases;
- justify why choosing a medical threshold is a *human values* decision, not just a math default.
```

## Start with a case: why "99% accurate" can be deadly

Imagine a clinic where **1%** of patients have a dangerous hidden infection, and someone builds a lazy program that ignores every lab test and simply prints **"Healthy"** for every single person.

Its overall accuracy is **99.0%** — yet it misses **100% of sick patients**!

In medicine, the two ways to be wrong carry wildly unequal human costs:
- A {term}`False positive (FP)` (false alarm) tells a healthy person they might be sick — causing worry and an extra follow-up test.
- A {term}`False negative (FN)` (missed case) sends a sick person home untreated — potentially costing a life.

Plain accuracy mixes both errors into one number. To judge a medical AI responsibly, we count mistakes **by type** using a {term}`Confusion matrix`.

## Count a 20-patient clinic by hand first!

Suppose 20 patients take a cancer screen: **5 truly have cancer**, and **15 are truly healthy**. Our AI flags **6** people total: it catches **4** of the 5 real cancers and falsely alarms on **2** healthy people.

| | AI says **Healthy (0)** | AI says **Cancer (1)** |
|---|---|---|
| **Truly Healthy (15)** | **13** True Negatives (`TN`) | **2** False Positives (`FP` — false alarms) |
| **Truly Cancer (5)** | **1** False Negative (`FN` — missed!) | **4** True Positives (`TP` — caught!) |

From those four boxes come three essential questions:
1. **{term}`Sensitivity` (Recall):** *"Of the 5 truly sick people, how many did we catch?"* → $4 / 5 = \mathbf{80\%}$ (`TP / (TP + FN)`).
2. **{term}`Specificity`:** *"Of the 15 truly healthy people, how many did we clear?"* → $13 / 15 = \mathbf{87\%}$ (`TN / (TN + FP)`).
3. **{term}`Precision` (PPV):** *"Of the 6 people we flagged, how many really have cancer?"* → $4 / 6 = \mathbf{67\%}$ (`TP / (TP + FP)`).

## The hidden dial: moving the decision threshold!

Most beginners assume a classifier outputs `"Benign"` or `"Malignant"`. Actually, models like logistic regression output a **risk percentage** (like `18%`, `62%`, or `94%`). Software converts that risk into yes/no using a {term}`Decision threshold` — by default **`50%` (`0.50`)**.

**But 50% has nothing sacred about it in medicine!**
- Lower the dial to **`15%`** → flag any biopsy with ≥15% risk → **fewer missed cancers** (higher sensitivity!), more false alarms.
- Raise the dial to **`85%`** → flag only near-certainties → **almost zero false alarms** (higher specificity & precision!), more missed cancers.

```{admonition} Predict before you run
:class: tip
We will train a classifier on our real breast-cancer biopsies and test three dials: `0.15`, `0.50`, and `0.85`. As we move the dial **down** from `0.50` to `0.15`, will **missed cancers (FN)** go up or down? What about **false alarms (FP)**?
```

```{code-cell} python
:label: ch24-threshold-sweep
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)    # 1 = malignant

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_tr)
model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)

# Continuous predicted risk of malignancy (0.0 to 1.0)
risk = model.predict_proba(scaler.transform(X_te))[:, 1]

print(f"{'Threshold':>9} | {'Caught (TP)':>11} {'Missed (FN)':>11} {'Alarms (FP)':>11} | {'Sensitivity':>11} {'Specificity':>11}")
print("-" * 74)
for t in [0.15, 0.50, 0.85]:
    preds = (risk >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
    print(f"{t:9.2f} | {tp:11d} {fn:11d} {fp:11d} | {tp/(tp+fn):11.1%} {tn/(tn+fp):11.1%}")

# Plot the smooth tug-of-war across every threshold from 5% to 95%
ts = np.linspace(0.05, 0.95, 40)
sens_list, spec_list = [], []
for t in ts:
    tn, fp, fn, tp = confusion_matrix(y_te, (risk >= t).astype(int), labels=[0, 1]).ravel()
    sens_list.append(tp / (tp + fn) * 100)
    spec_list.append(tn / (tn + fp) * 100)

plt.figure(figsize=(7.2, 4.0))
plt.plot(ts * 100, sens_list, color="#D55E00", lw=2.4, label="Sensitivity (% cancers caught)")
plt.plot(ts * 100, spec_list, color="#0072B2", lw=2.4, label="Specificity (% healthy cleared)")
plt.axvline(50, color="#4D4D4D", ls="--", lw=1.2, label="Software default (50%)")
plt.xlabel("Decision threshold (% risk required to flag)")
plt.ylabel("Score (%)")
plt.title("Sliding the threshold trades catching cancers against raising false alarms")
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.show()
```

Look at that crisp visual tug-of-war! Without retraining a single weight, simply moving the threshold from `0.50` down to `0.15` cuts missed cancers from `4` down to `3` (and raising it to `0.85` cuts false alarms all the way to **`0`**, at the cost of `7` missed cancers!).

```{admonition} Going deeper — ROC curves, AUROC, and Calibration
:class: dropdown
Researchers often summarize a model's ranking skill across *all* thresholds at once using an **ROC curve** (plotting Sensitivity against False Positive Rate). The area under that curve — **AUROC** — runs from `0.50` (random guessing) to `1.00` (perfect ranking).

Two important advanced cautions:
1. For **very rare** diseases (like 1% prevalence), AUROC can look impressively high (`>0.93`) even when most alerts are false alarms — so medical AI papers also check **Precision–Recall curves**.
2. High AUROC only checks *ranking*, not {term}`Calibration` — whether a predicted "30% risk" truly happens about 30% of the time!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.4.1 (Modify) — Aim for near-zero missed cancers
:class: hint
In [](#ch24-threshold-sweep), try threshold `t = 0.05` (flagging any biopsy with ≥5% predicted risk). How many cancers are missed (`FN`), and how many false alarms (`FP`) does that cost?
```

```{admonition} Solution
:class: dropdown
```python
tn, fp, fn, tp = confusion_matrix(y_te, (risk >= 0.05).astype(int), labels=[0, 1]).ravel()
print(f"At t = 0.05: caught = {tp}, missed (FN) = {fn}, false alarms (FP) = {fp}")
```
Missed cancers drop to just **1** (`98.4%` sensitivity!), while false alarms rise to ~`18` benign biopsies sent for closer specialist review — a very common trade-off for a first-line cancer screen!
```

```{admonition} Exercise 2.4.2 (Make) — Compute precision at all three thresholds
:class: hint
Write a 4-line loop printing **precision** (`tp / (tp + fp)`) for `t` in `[0.15, 0.50, 0.85]`. At which threshold is precision highest?
```

```{admonition} Solution
:class: dropdown
```python
for t in [0.15, 0.50, 0.85]:
    tn, fp, fn, tp = confusion_matrix(y_te, (risk >= t).astype(int), labels=[0, 1]).ravel()
    print(f"t = {t:.2f}: precision = {tp / (tp + fp):.1%}")
```
Precision climbs from `82.4%` (`t=0.15`) → `93.8%` (`t=0.50`) → **`100.0%`** (`t=0.85`), because being pickier eliminates false alarms!
```

```{admonition} Exercise 2.4.3 (Critical Reasoning) — Two different clinics, two different thresholds
:class: hint
Compare two medical decisions:
- **Case A:** A quick, painless blood screen deciding who gets an inexpensive second check for a dangerous infection.
- **Case B:** A test deciding whether to start a harsh chemotherapy drug with severe side effects.

Which case calls for a **low threshold** (favoring sensitivity) and which calls for a **high threshold** (favoring specificity/precision)? Explain why.
```

```{admonition} Solution
:class: dropdown
- **Case A → Low threshold (high sensitivity):** Missing a dangerous infection is catastrophic, while a false alarm only means a cheap, painless follow-up check.
- **Case B → High threshold (high specificity & precision):** Giving harsh chemotherapy to a healthy person causes terrible harm, so you want near-certainty before starting treatment!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **confusion matrix** separates errors into **false positives** (false alarms) and **false negatives** (missed cases).
- **Sensitivity** measures how many sick patients you caught; **specificity** measures how many healthy patients you cleared; **precision** measures how often a positive flag is real.
- Classifiers output **risk scores**, and humans pick the **decision threshold** based on which mistake hurts more.
```

*Next up:* We've now met k-NN, decision trees, and logistic regression — how do these classifiers actually carve up a graph, and why do **forests of trees** often win on hospital tables? Let's compare them side by side in [](2_5-lines-trees-and-forests.md)!
