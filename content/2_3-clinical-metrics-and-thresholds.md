---
title: 'Chapter 2.3 — Beyond Accuracy: Confusion Matrices, Thresholds, ROC, and Calibration'
short_title: '2.3 · Clinical metrics & thresholds'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.3 — Beyond Accuracy: Confusion Matrices, Thresholds, ROC, and Calibration

*Estimated time: ~45 minutes · Prerequisites: [](1_4-probability-and-base-rates.md), [](2_2-generalization-and-leakage.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- decompose any binary classifier into a **confusion matrix** (TP, TN, FP, FN) and weigh asymmetric clinical costs;
- explain why a classifier outputs a continuous **risk probability** and how moving the **decision threshold** trades sensitivity against specificity;
- plot and interpret both an **ROC curve (AUROC)** and a **Precision–Recall curve (AUPRC)** — and explain why AUPRC is vital for rare conditions;
- inspect a **calibration curve** (reliability diagram) to check whether a model's predicted percentages can be taken at face value.
```

## Start with a case: the 99% accurate model that catches zero patients

Suppose only **1%** of patients arriving at a rural clinic have sepsis (a life-threatening bloodstream infection). Imagine I build a ridiculously lazy "AI":

```python
def lazy_sepsis_ai(patient):
    return "Healthy"   # ignores every lab value!
```

What is its accuracy? **99.0%!!** Yet it misses **100% of dying patients**.

Even when a model *is* smart, plain accuracy lumps two wildly unequal clinical mistakes together:
- A {term}`False positive (FP)` (flagging a healthy patient for extra review) causes anxiety, extra blood work, or clinician alert fatigue.
- A {term}`False negative (FN)` (clearing a patient with cancer or sepsis) can cost a life.

In this chapter, we move beyond a single "accuracy" grade and open the full clinical evaluation dashboard: **confusion matrices, tunable thresholds, ROC/PR curves, and calibration.**

## From probabilities to decisions: who picks the threshold?

A subtle misconception among beginners is thinking that a classifier outputs `"Malignant"` or `"Benign"`. In truth, modern classifiers (logistic regression, random forests, neural networks) output a continuous **risk probability** $p$ between `0.00` and `1.00`.

Converting $p$ into an action requires choosing a {term}`Decision threshold` $\tau$ (*tau*): *"Flag the patient as high-risk if $p \ge \tau$."* Software libraries default to $\tau = 0.50$, **but 0.50 has no magic medical status!**
- Lowering $\tau$ (say, to `0.20`) catches more true cancers (**higher sensitivity**), at the cost of more false alarms (**lower specificity**).
- Raising $\tau$ (say, to `0.85`) ensures almost every flag is real (**higher precision / specificity**), at the cost of missing borderline cases (**lower sensitivity**).

Picking $\tau$ is not a math problem — it is a **clinical and ethical decision** based on the relative cost of a missed case versus a false alarm.

## Lab 1: Sweeping the threshold on breast-cancer diagnosis

Let's train a leak-free standardized logistic regression model on our Wisconsin Breast Cancer dataset and inspect its {term}`Confusion matrix` at three different thresholds: `0.50` (software default), `0.15` (high-sensitivity cancer screen), and `0.85` (high-specificity rule-in).

```{admonition} Predict before you run
:class: tip
When we lower the threshold from `0.50` down to `0.15` (flagging any biopsy with ≥15% predicted risk of malignancy), what happens to **False Negatives (missed cancers)** and what happens to **False Positives (benign biopsies flagged)**?
```

```{code-cell} python
:label: ch23-thresholds
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)    # 1 = malignant

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]               # P(malignant)

print(f"{'Threshold':>9} | {'TP':>3} {'FN':>3} {'FP':>3} {'TN':>3} | "
      f"{'Sensitivity':>11} {'Specificity':>11} {'Precision':>9}")
print("-" * 68)
for tau in (0.15, 0.50, 0.85):
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    print(f"{tau:9.2f} | {tp:3d} {fn:3d} {fp:3d} {tn:3d} | "
          f"{sens:10.1%} {spec:11.1%} {prec:9.1%}")
```

Stare at the `FN` (missed cancers) and `FP` (false alarms) columns!
- At the default `0.50`, the model catches 60 of 64 cancers (`93.8%` sensitivity, 4 missed) with 4 false alarms.
- Dropping the threshold to `0.15` catches an extra cancer (`95.3%` sensitivity, 3 missed), at the cost of 13 false alarms.
- Raising the threshold to `0.85` drops false alarms all the way to **zero** (`100.0%` specificity and `100.0%` precision!), but lets **7 real cancers** slip through!

## Lab 2: Visualizing every threshold at once — ROC vs. Precision–Recall

Instead of testing three thresholds by hand, we sweep $\tau$ smoothly from `1.0` down to `0.0` and plot two famous summary curves:

::::{tab-set}

:::{tab-item} ROC Curve (AUROC)
Plots **True Positive Rate (Sensitivity)** on the y-axis against **False Positive Rate ($1 - \text{Specificity}$)** on the x-axis.
- A random coin-flip model sits on the diagonal (`AUROC = 0.50`).
- A perfect ranker hugs the top-left corner (`AUROC = 1.00`).
- **Key property:** AUROC does *not* change when the disease becomes rarer — great for comparing models across clinics, **dangerous** if it hides a flood of false alarms in a rare-disease screen!
:::

:::{tab-item} Precision–Recall Curve (AUPRC)
Plots **Precision (PPV)** on the y-axis against **Recall (Sensitivity)** on the x-axis.
- A coin-flip model sits horizontally at the **disease prevalence** (base rate)!
- **Key property:** Directly exposes the false-alarm explosion from Chapter 1.4 when a condition is rare. Always demand **AUPRC alongside AUROC** on imbalanced medical tasks (sepsis, ICU cardiac arrest, rare genetic disorders).
:::

::::

Let's plot **three** clinical evaluation panels side by side: (A) the ROC curve, (B) the Precision–Recall curve, and (C) a **calibration curve**!

```{admonition} Definition — calibration
:class: important
{term}`Calibration` measures whether a model's predicted probabilities match real-world outcomes. Among all patients to whom a well-calibrated model assigns ~30% risk, roughly 30% should truly have the condition. Good ranking (high AUROC) does *not* guarantee good calibration — a model can rank sick patients above healthy ones while still being wildly overconfident!
```

```{code-cell} python
:label: ch23-curves-fig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.naive_bayes import GaussianNB

BLUE, VERMILION, TEAL, SLATE = "#0072B2", "#D55E00", "#009E73", "#4D4D4D"

# Compare Logistic Regression vs. a notoriously overconfident model (Gaussian Naive Bayes)
nb = make_pipeline(StandardScaler(), GaussianNB()).fit(X_train, y_train)
probs_nb = nb.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

# Panel A: ROC Curve
for name, p, col in [("Logistic Reg", probs, BLUE), ("Naive Bayes", probs_nb, VERMILION)]:
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, lw=2.3, color=col, label=f"{name} (AUROC={auc(fpr, tpr):.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color=SLATE, lw=1.2, label="Random (0.500)")
axes[0].set_xlabel("False positive rate (1 − Specificity)")
axes[0].set_ylabel("True positive rate (Sensitivity)")
axes[0].set_title("A. ROC Curve (ranking quality)")
axes[0].legend(frameon=False, loc="lower right", fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].spines[["top", "right"]].set_visible(False)

# Panel B: Precision-Recall Curve
for name, p, col in [("Logistic Reg", probs, BLUE), ("Naive Bayes", probs_nb, VERMILION)]:
    prec, rec, _ = precision_recall_curve(y_test, p)
    ap = average_precision_score(y_test, p)
    axes[1].plot(rec, prec, lw=2.3, color=col, label=f"{name} (AUPRC={ap:.3f})")
axes[1].axhline(y_test.mean(), ls="--", color=SLATE, lw=1.2,
                label=f"Baseline ({y_test.mean():.2f})")
axes[1].set_xlabel("Recall (Sensitivity)")
axes[1].set_ylabel("Precision (PPV)")
axes[1].set_title("B. Precision–Recall Curve")
axes[1].legend(frameon=False, loc="lower left", fontsize=8)
axes[1].grid(alpha=0.25)
axes[1].spines[["top", "right"]].set_visible(False)

# Panel C: Calibration (Reliability Diagram)
for name, p, col in [("Logistic Reg", probs, BLUE), ("Naive Bayes", probs_nb, VERMILION)]:
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=6, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", lw=2.2, color=col, label=name)
axes[2].plot([0, 1], [0, 1], ls="--", color=SLATE, lw=1.2, label="Perfect calibration")
axes[2].set_xlabel("Mean predicted risk")
axes[2].set_ylabel("Observed malignant fraction")
axes[2].set_title("C. Calibration (trusting probabilities)")
axes[2].legend(frameon=False, loc="upper left", fontsize=8)
axes[2].grid(alpha=0.25)
axes[2].spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

Look at Panel C of [](#ch23-curves-fig)! Both models look strong on ROC and PR curves (`AUROC > 0.98`), yet **Naive Bayes** pushes almost every prediction straight to `0.00` or `1.00` — classic overconfidence! Logistic regression stays much closer to the diagonal, meaning its risk percentages are far more trustworthy when a doctor has to communicate risk to a patient or set an abstention rule (which we'll build in Chapter 5.3!).

## Lab 3: Why AUPRC saves lives on rare diseases

Let's prove why rare-disease AI papers that report *only* AUROC can mislead clinicians. Watch what happens when we simulate a **1% rare-disease screening task** using a decent risk scorer:

```{code-cell} python
:label: ch23-rare-auprc
rng = np.random.default_rng(7)
n_screen = 10_000
y_rare = np.zeros(n_screen, dtype=int)
y_rare[:100] = 1                                         # exactly 1% prevalence

# Good but imperfect risk score: sick people average higher scores than healthy
scores = np.where(y_rare == 1,
                  rng.normal(2.2, 1.0, n_screen),
                  rng.normal(0.0, 1.0, n_screen))

fpr_r, tpr_r, _ = roc_curve(y_rare, scores)
auroc_r = auc(fpr_r, tpr_r)
auprc_r = average_precision_score(y_rare, scores)

print(f"Rare disease screen (1% prevalence):")
print(f"  AUROC = {auroc_r:.3f}  <-- sounds phenomenal in a press release!")
print(f"  AUPRC = {auprc_r:.3f}  <-- exposes the real false-alarm burden!")
```

**AUROC = 0.936**, yet **AUPRC = 0.277**! The AUROC makes the tool sound near-perfect because the 9,900 true negatives keep the false-positive *rate* tiny; meanwhile, AUPRC immediately reveals that most flagged alarms are false positives — echoing the base-rate lesson of Chapter 1.4!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.3.1 (Modify) — Hitting a clinical safety target
:class: hint
Suppose an oncology clinic requires **at least 98% sensitivity** (missing almost no cancers) on `probs` from [](#ch23-thresholds). Write a short loop testing thresholds from `0.05` to `0.50` in steps of `0.05` to find the **highest** threshold that still achieves $\ge 98\%$ sensitivity. What specificity and precision do you get there?
```

```{admonition} Solution
:class: dropdown
```python
best_tau = None
for tau in np.arange(0.05, 0.51, 0.05):
    pred = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    sens, spec, prec = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
    if sens >= 0.98:
        best_tau = (tau, sens, spec, prec)
print(f"Highest safe threshold: tau = {best_tau[0]:.2f} | "
      f"sens = {best_tau[1]:.1%}, spec = {best_tau[2]:.1%}, prec = {best_tau[3]:.1%}")
```

Choosing the *highest* threshold that clears the 98% sensitivity bar maximizes specificity and minimizes unnecessary biopsies while preserving patient safety!
```

```{admonition} Exercise 2.3.2 (Make) — Checking F1 score vs. clinical weights
:class: hint
The **F1 score** is the harmonic mean of precision and recall: $F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$. Compute $F_1$ at `tau = 0.15`, `0.50`, and `0.85` on our breast-cancer predictions. Why might an oncologist still prefer `tau = 0.15` even if `tau = 0.50` has a slightly higher $F_1$?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.metrics import f1_score
for tau in (0.15, 0.50, 0.85):
    print(f"tau = {tau:.2f}: F1 = {f1_score(y_test, (probs >= tau).astype(int)):.3f}")
```

`F1` treats precision and recall with equal weight. In cancer screening, a missed malignancy (FN) is far costlier than a follow-up confirmatory biopsy (FP), so clinicians intentionally favor recall over precision rather than blindly maximizing symmetric `F1`.
```

```{admonition} Exercise 2.3.3 (Critical Reasoning) — When does high AUROC fail the patient?
:class: hint
Imagine two cardiac-risk calculators both achieve `AUROC = 0.88`. Calculator A is well-calibrated, while Calculator B systematically predicts **double** every patient's true 10-year heart-attack risk (e.g., predicting 24% when real risk is 12%). Clinical guidelines recommend starting statin medication when 10-year risk exceeds **10%**. Why is Calculator B dangerous despite having the exact same AUROC?
```

```{admonition} Solution
:class: dropdown
Because `AUROC` only measures **ranking order** (multiplying every score by 2 doesn't change who ranks above whom!), it is completely blind to miscalibration! Calculator B would put thousands of low-risk patients (true risk 5–9%, predicted risk 10–18%) onto lifelong medication unnecessarily. Whenever a clinical guideline uses an explicit probability cutoff, **calibration is just as critical as AUROC** [@rajpurkar2022ai].
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Plain accuracy hides asymmetric clinical errors; always break predictions into **TP, TN, FP, FN** via a **confusion matrix**.
- Models output **probabilities**; humans choose the **decision threshold** to trade sensitivity against specificity and precision.
- **AUROC** measures overall ranking independent of prevalence; **AUPRC** reveals false-alarm overload on rare diseases.
- **Calibration** checks whether "30% predicted risk" truly happens ~30% of the time — essential for real clinical guidelines.
```

*Next up:* We've met k-NN, decision trees, and logistic regression — how do these model families actually carve up feature space, and why do **ensembles of trees** still dominate structured hospital tables? Let's compare them visually in [](2_4-trees-ensembles-and-boundaries.md).
