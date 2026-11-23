---
title: 'Chapter 5.4 — Capstone: Build and Audit a Responsible Health AI System'
short_title: '5.4 · Biomedical capstone'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.4 — Capstone: Build and Audit a Responsible Health AI System

*Estimated time: 3–6 hours (project) · Prerequisites: Parts 1–5*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/06-capstone-starter.ipynb)

```{admonition} Learning objectives
:class: note
By completing this capstone project you will:
- frame an **end-to-end biomedical AI workflow**: clinical question → data exploration → leak-free modeling → clinical & fairness audit → honest **Model Card**;
- benchmark at least **two model families** against a simple **baseline** using proper clinical metrics (sensitivity, specificity, AUROC, AUPRC, calibration);
- perform a **disaggregated subgroup fairness audit** AND a **selective-prediction (abstention)** or **stress-test** experiment;
- write a concise, honest **TRIPOD+AI-inspired Model Card** detailing limitations, privacy considerations, and human-in-the-loop safeguards.
```

## Start with a case — yours!

Over nineteen chapters, you have simulated outbreaks, caught Simpson's paradox, exposed patient-level data leakage, trained a neural network from scratch without calculus, written a 2D image convolution, built a clinical Mini-RAG, embedded real proteins, and simulated differential privacy and federated learning across three hospitals.

Now it is time to **build and audit your own complete health-AI system** from start to finish. You don't need a supercomputer or a fancy architecture — **an honest, leak-free, calibrated, equity-audited simple model beats an opaque 99% black box every single time.**

## Choose one of three authentic biomedical tracks

All three tracks run 100% offline with zero downloads using our light toolkit (`numpy`, `pandas`, `matplotlib`, `scikit-learn`):

::::{grid} 1 1 3 3

:::{card} Track A · Oncology Triage
:header: **Clinical Classification (`load_breast_cancer`)**

Build a biopsy malignancy screener (569 tumors, 30 nuclear measurements). Tune the decision threshold for ≥97% sensitivity, audit calibration, compare Logistic Regression vs. Random Forest, and evaluate abstention coverage across tumor-size subgroups!
:::

:::{card} Track B · Diabetes Risk & Equity
:header: **Clinical Prognosis (`load_diabetes`)**

Predict high 1-year diabetes progression (`target ≥ 180`) on 442 real patients in native clinical units (`scaled=False`). Compare models, inspect permutation feature importance, and audit sensitivity & false-negative gaps across **age** and **BMI** strata!
:::

:::{card} Track C · Genomic Sequence Classifier
:header: **Bioinformatics & Shift Stress Test**

Engineer GC content + k-mer features on synthetic/curated promoter vs. coding DNA sequences (Chapter 4.1). Test generalization, deliberately inject a **spurious motif shortcut** (Chapter 3.4) or **distribution shift**, and measure how external accuracy changes!
:::

::::

## The 5-stage responsible health AI workflow

```{mermaid}
flowchart LR
    M1["1. Question &<br/>Clinical Context"] --> M2["2. Explore &<br/>Data Audit"]
    M2 --> M3["3. Leak-Free<br/>Models + Baseline"]
    M3 --> M4["4. Clinical, Fairness<br/>& Abstention Audit"]
    M4 --> M5["5. Limitations &<br/>Model Card"]
```

## Worked mini-capstone example (Track A skeleton)

Below is a compact, end-to-end runnable skeleton on **Track A (Oncology Triage)** demonstrating how clean, honest clinical ML code looks in practice. Notice every single good habit in action: locked test split, leak-free `Pipeline`, majority baseline, custom high-sensitivity threshold, subgroup audit (small vs. large tumors), and selective abstention!

```{code-cell} python
:label: ch54-worked-starter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, recall_score

# --- STAGE 1 & 2: QUESTION & EXPLORE ---
cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)    # 1 = malignant (positive class)
print(f"Cohort: {X.shape[0]} biopsies, {X.shape[1]} features | "
      f"Malignant prevalence: {y.mean():.1%} ({y.sum()}/{len(y)})")

# --- STAGE 3: LEAK-FREE SPLIT, BASELINE & MODEL COMPARISON ---
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

majority_baseline = max(y_te.mean(), 1 - y_te.mean())
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest (200)": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0),
}

print(f"\nMajority-class baseline accuracy: {majority_baseline:.1%}")
for name, clf in models.items():
    cv_auc = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    print(f"  {name:<22}: 5-fold CV AUROC = {cv_auc:.3f}")

# Fit winning model on training set and evaluate ONCE on held-out test set
best_model = models["Logistic Regression"].fit(X_tr, y_tr)
probs = best_model.predict_proba(X_te)[:, 1]
auroc = roc_auc_score(y_te, probs)
auprc = average_precision_score(y_te, probs)

# Choose a screening threshold favoring high sensitivity (tau = 0.25)
tau = 0.25
pred = (probs >= tau).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
print(f"\nHeld-out test @ tau={tau:.2f} | AUROC={auroc:.3f} | AUPRC={auprc:.3f}")
print(f"  Sensitivity = {tp/(tp+fn):.1%} (missed {fn} cancers) | "
      f"Specificity = {tn/(tn+fp):.1%} ({fp} false alarms)")

# --- STAGE 4: SUBGROUP FAIRNESS & ABSTENTION AUDIT ---
# Audit across tumor size strata (smaller tumors are harder to detect early!)
median_rad = np.median(X_tr[:, 0])
small_mask = X_te[:, 0] < median_rad
large_mask = ~small_mask
print("\nSubgroup Sensitivity Audit (Small vs. Large Mean Radius):")
for name, m in [("Smaller tumors (< median)", small_mask), ("Larger tumors  (≥ median)", large_mask)]:
    n_pos = y_te[m].sum()
    sens = recall_score(y_te[m], pred[m]) if n_pos > 0 else np.nan
    print(f"  {name}: n={m.sum():>2} ({n_pos:>2} malignant) | sensitivity = {sens:.1%}")

# Selective prediction: auto-decide only when confidence >= 90%
conf = np.maximum(probs, 1 - probs)
keep = conf >= 0.90
auto_pred = (probs >= 0.50).astype(int)
print(f"\nAbstention @ 90% confidence: coverage = {keep.mean():.1%}, "
      f"auto-accuracy = {(auto_pred[keep] == y_te[keep]).mean():.1%}, "
      f"deferred to human = {(~keep).sum()} cases")
```

Look at what that **Subgroup Sensitivity Audit** just revealed — even on a 97% accurate model! **Smaller tumors** (where early cancer detection matters *most*!) have noticeably fewer malignant cases and sit closer to the decision boundary, making them far more vulnerable to missed diagnoses than obvious large tumors. *That* exact finding belongs front and center in Stage 5!

## Capstone rubric & milestones

```{admonition} 5-Milestone Grading & Self-Review Rubric
:class: important
- **Milestone 1 — Clinical / Biological Framing (15%):** State one crisp, answerable question, identify your exact positive class (`1`), explain the real-world stakes of a **false negative vs. false positive**, and clarify whether your task is *predictive* or *causal*.
- **Milestone 2 — Data Exploration & Quality Audit (20%):** Report cohort size, class balance, units/ranges, missing values, skew/outliers, and include **at least one clean, annotated figure** (honest axes, colorblind-safe palette).
- **Milestone 3 — Leak-Free Modeling & Baselines (20%):** Compare **at least two model families** inside leak-free `Pipeline`s using k-fold cross-validation against a **simple baseline**. Report held-out **AUROC, AUPRC, confusion matrix, sensitivity, specificity**, and justify your chosen **decision threshold**.
- **Milestone 4 — Fairness, Calibration & Safety Audit (25%):** Plot a **calibration curve** or **abstention coverage–accuracy curve**, break down sensitivity/accuracy across **at least two subgroups** (report the gap!), and run **one stress test** (e.g., representation imbalance, threshold shift, or shortcut injection).
- **Milestone 5 — Honest Health AI Model Card (20%):** Complete the 6-part **Model Card** below, walking through the **8-Point Critical Appraisal Checklist** from [](5_3-humans-regulation-deployment.md).
```

## Health AI Model Card template (Stage 5)

Copy this compact template into the final markdown cell of your capstone notebook ([Companion Notebook 06](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/06-capstone-starter.ipynb)):

```markdown
### Model Card — [Your Project Title]
- **Intended use & clinical role:** [Screening triage / risk stratification / research hypothesis — NOT autonomous diagnosis.]
- **Training & evaluation data:** [Dataset, sample size, positive prevalence, split strategy, missingness.]
- **Primary performance:** [Baseline vs. chosen model: AUROC, AUPRC, Sensitivity & Specificity at chosen threshold τ, plus 95% bootstrap CI.]
- **Subgroup fairness & calibration:** [Report worst-performing subgroup and exact sensitivity/accuracy gap; note calibration quality.]
- **Known failure modes & privacy risks:** [Distribution shift risks, potential shortcuts, quasi-identifiers if real patients were used.]
- **Human-in-the-loop deployment plan:** [Abstention rule, deferred-case workflow, and what external validation / clinical trial is required before bedside use.]
```

## Launch your capstone starter notebook

Open the interactive starter notebook — pre-loaded with structured `TODO` cells for all three biomedical tracks and every milestone above:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/06-capstone-starter.ipynb)

## Final exercise: bootstrap your capstone's uncertainty!

```{admonition} Exercise 5.4.1 (Make) — Put a 95% confidence interval on your capstone's AUROC
:class: hint
A single test AUROC on 143 biopsies carries sampling uncertainty (Chapter 1.4!). Write a 7-line bootstrap loop (1,000 resamples of `(y_te, probs)` with replacement) to compute a **95% bootstrap confidence interval** for `auroc` in [](#ch54-worked-starter).
```

```{admonition} Solution
:class: dropdown
```python
rng_b = np.random.default_rng(0)
boot_aucs = []
for _ in range(1000):
    idx = rng_b.integers(0, len(y_te), size=len(y_te))
    if len(np.unique(y_te[idx])) == 2:
        boot_aucs.append(roc_auc_score(y_te[idx], probs[idx]))
lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f"Test AUROC = {auroc:.3f}  |  95% bootstrap CI = [{lo:.3f}, {hi:.3f}]")
```

Reporting `AUROC = 0.996 (95% CI: 0.990–1.000)` instead of a bare point estimate is an unmistakable hallmark of rigorous biomedical research — and fulfills a core item of the **TRIPOD+AI** guideline [@collins2024tripod]!
```

## Closing words

```{admonition} Congratulations — you completed TAIHLS!
:class: note
You started this book simulating a single bacterium dividing on a petri dish. You are leaving it able to build, inspect, critique, and audit the ideas behind modern clinical AI, medical imaging, large language models, protein and genome foundation models, differential privacy, and federated learning.

Keep two companions with you wherever your studies in medicine, biology, or computer science lead next: **curiosity** to explore what these extraordinary tools can unlock, and **scientific rigor** to always ask *what is the baseline, where could it fail, and who does it serve?*
```
