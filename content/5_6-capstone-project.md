---
title: 'Chapter 5.6 — Semester Capstone: Build and Audit Your Own Health AI'
short_title: '5.6 · Semester capstone'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.6 — Semester Capstone: Build and Audit Your Own Health AI

*Weeks 15–16 · 2-Week Final Project · Prerequisites: Units 0–5*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/06-capstone-starter.ipynb)

```{admonition} Learning objectives
:class: note
By completing your semester capstone you will:
- run a complete health-AI project from start to finish: **question → data exploration → leak-free models → fairness & safety audit → honest Model Card**;
- compare at least **two classifiers** against a simple **baseline** and justify your clinical **decision threshold**;
- test **per-group performance** AND **abstention** ("saying I don't know") on real biomedical data;
- communicate both your model's strengths and its honest **limitations** to classmates.
```

## Your turn in the scientist's chair!

Over the past fourteen weeks, you learned Python from scratch, simulated outbreaks, untangled Simpson's paradox, caught 97%-accurate data leakage, trained a neural network, wrote a sliding image filter, built an open-book RAG helper, mapped real human hemoglobin and proteins, and tested fairness, differential privacy, and federated learning.

For the final two weeks, you will build and audit **one complete, honest health-AI mini-project**. Remember the golden rule of the entire semester: **a simple, leak-free, fairness-audited model you truly understand beats a fancy black box every single time.**

## Pick one of three authentic health & biology tracks

All three tracks run 100% offline with zero downloads using our classroom Python toolkit:

::::{grid} 1 1 3 3

:::{card} Track A · Breast Biopsy Triage
:header: **Oncology (`load_breast_cancer`)**

Use 569 real biopsies to compare Logistic Regression vs. Random Forest, tune a safe screening threshold (minimizing missed cancers!), audit smaller vs. larger tumors, and add an abstention rule!
:::

:::{card} Track B · Diabetes Risk & Age Equity
:header: **Endocrinology (`load_diabetes`)**

Predict high 1-year diabetes progression (`target ≥ 180`) on 442 real patients in clinical units. Check which measurements matter most, and audit sensitivity separately for younger (`< 50`) vs. older (`≥ 50`) patients!
:::

:::{card} Track C · DNA Sequence Classifier
:header: **Bioinformatics (GC + 2-mers)**

Build a classifier telling high-GC human coding DNA apart from AT-rich promoter regions using Chapter 4.1's k-mer fingerprints — then deliberately inject a **shortcut motif** (Chapter 3.5) and watch external accuracy drop!
:::

::::

## A clean worked starter (Track A skeleton)

Here is a complete, readable skeleton touching every stage of a responsible project. Study how short and clean it is — your job in [Companion Notebook 06](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/06-capstone-starter.ipynb) is to customize, visualize, and extend it!

```{code-cell} python
:label: ch56-capstone-starter
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score

# 1. QUESTION & EXPLORE
cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)    # 1 = malignant
print(f"Dataset: {len(y)} biopsies, {X.shape[1]} measurements | Malignant: {y.mean():.1%}")

# 2. LEAK-FREE SPLIT, BASELINE & TWO MODELS
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)

lr = LogisticRegression(max_iter=1000).fit(sc.transform(X_tr), y_tr)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0).fit(X_tr, y_tr)

baseline = max(y_te.mean(), 1 - y_te.mean())
print(f"\nMajority-class baseline : {baseline:.1%}")
print(f"Logistic Regression     : {lr.score(sc.transform(X_te), y_te):.1%}")
print(f"Random Forest (200)     : {rf.score(X_te, y_te):.1%}")

# 3. CLINICAL THRESHOLD (favor catching cancers: tau = 0.25)
risk = lr.predict_proba(sc.transform(X_te))[:, 1]
pred = (risk >= 0.25).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
print(f"\nAt threshold 0.25 -> Sensitivity: {tp/(tp+fn):.1%} (missed {fn}) | Specificity: {tn/(tn+fp):.1%} ({fp} alarms)")

# 4. SUBGROUP FAIRNESS AUDIT (smaller vs. larger tumors)
mid_size = np.median(X_tr[:, 0])
for label, mask in [("Smaller tumors", X_te[:, 0] < mid_size),
                    ("Larger tumors ", X_te[:, 0] >= mid_size)]:
    sens = recall_score(y_te[mask], pred[mask])
    print(f"  {label} (n={mask.sum():>2}): sensitivity = {sens:.1%}")

# 5. ABSTENTION ("I don't know" below 90% confidence)
conf = np.maximum(risk, 1 - risk)
keep = conf >= 0.90
auto_pred = (risk >= 0.50).astype(int)
print(f"\nAbstention @ 90% conf: auto-handles {keep.mean():.1%} at {(auto_pred[keep] == y_te[keep]).mean():.1%} acc "
      f"({(~keep).sum()} sent to human)")
```

Notice that **Subgroup Audit** right away: smaller tumors (the hardest early-stage cases!) have noticeably lower sensitivity (`75.0%`) than large obvious tumors (`98.0%`) — an exact, honest finding to highlight in your Model Card!

## 5-Milestone Project Rubric

```{admonition} Capstone Grading Rubric (Weeks 15–16)
:class: important
1. **Milestone 1 — Clinical Question (15%):** Name your track, define class `1`, and explain why a **missed case (FN)** and a **false alarm (FP)** carry different human costs.
2. **Milestone 2 — Data Exploration (20%):** Report patient count, class balance, mean vs. median for two features, missing values, and include **one clear, honest chart** (axes starting cleanly, clear labels).
3. **Milestone 3 — Honest Modeling & Threshold (20%):** Compare **two models** against the **majority-class baseline** using a leak-free split. Show the **confusion matrix** and justify your chosen **decision threshold**.
4. **Milestone 4 — Fairness & Safety Audit (25%):** Split performance across **at least two subgroups** (report the gap!) AND test **abstention** (how much coverage do you get at ~95%+ auto-accuracy?).
5. **Milestone 5 — Honest Model Card (20%):** Write a clear 1-paragraph **Model Card** answering the **8-Question Checklist** (Chapter 5.5): *Who does this work worst for? What data is missing? Where would deploying it be unsafe? How do we keep a human in the loop?*
```

## Launch your capstone starter notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/06-capstone-starter.ipynb)

## Final reflection exercise

```{admonition} Exercise 5.6.1 (Make) — Check how much your threshold affects the smaller-tumor gap
:class: hint
In [](#ch56-capstone-starter), change the screening threshold from `0.25` to `0.50` and compare sensitivity on **Smaller tumors** vs. **Larger tumors**. Does lowering the threshold help the harder-to-catch smaller tumors?
```

```{admonition} Solution
:class: dropdown
```python
for t in [0.50, 0.25, 0.15]:
    p_t = (risk >= t).astype(int)
    s_small = recall_score(y_te[X_te[:, 0] < mid_size], p_t[X_te[:, 0] < mid_size])
    s_large = recall_score(y_te[X_te[:, 0] >= mid_size], p_t[X_te[:, 0] >= mid_size])
    print(f"threshold {t:.2f} -> Smaller tumors sens: {s_small:.1%} | Larger tumors sens: {s_large:.1%}")
```
Lowering the threshold helps catch tricky borderline small tumors that sit right near 0.50 — demonstrating why tuning thresholds *and* checking subgroups together makes medical AI safer!
```

```{admonition} Congratulations — you completed TAIHLS!
:class: note
You started this semester printing six heart-rate readings in Python. You are leaving it able to build, test, and critically question the core ideas behind medical classifiers, neural networks, image filters, language models, biological foundation models, fairness audits, and modern health privacy.

Carry both **curiosity** and **healthy skepticism** into whatever you study next — science needs people who can build *and* ask who these tools serve.
```
