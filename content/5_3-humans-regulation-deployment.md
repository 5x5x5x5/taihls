---
title: 'Chapter 5.3 — From Prototype to Bedside: Human-in-the-Loop, Clinical Trials, and Governance'
short_title: '5.3 · Humans, trials & regulation'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.3 — From Prototype to Bedside: Human-in-the-Loop, Clinical Trials, and Governance

*Estimated time: ~45 minutes · Prerequisites: [](2_3-clinical-metrics-and-thresholds.md), [](5_1-algorithmic-fairness.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/05-fairness-and-abstention-audit.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- implement **abstention (selective prediction)** using predicted probabilities and analyze the **coverage–accuracy trade-off** — including checking *per-group coverage*;
- contrast two real clinical human-factors hazards: **automation bias** and **algorithmic aversion**;
- trace the **5-stage clinical evidence ladder** from retrospective benchmark to prospective **randomized controlled trial** and post-market monitoring;
- explain modern medical-AI governance frameworks: **FDA Predetermined Change Control Plans (PCCPs)**, the **EU AI Act**, and **TRIPOD+AI**.
```

## Start with a case: the humblest AI in the radiology room

Imagine an AI triage system scanning hundreds of breast biopsies or chest X-rays every morning. On 85% of scans, the signs are textbook clear — predicted confidence sits above `95%`. On the remaining 15%, the tissue is blurry, rare, or borderline — predicted probability hovers near `55%`, barely better than a coin flip.

A naive AI forces itself to output a confident-looking `"Normal"` or `"Cancer"` on *every single case*. A **safe {term}`Human-in-the-loop`** system knows when to say **"I am not sure — send this case straight to a specialist!"**

```{admonition} Definition — abstention (selective prediction)
:class: important
{term}`Abstention` lets a model automatically handle high-confidence cases while deferring uncertain cases (where confidence falls below a threshold) to a human expert. **Coverage** is the fraction of cases the model auto-decides; **auto-accuracy** is its accuracy on the cases it keeps.
```

## Lab 1: Selective prediction — and checking who gets deferred!

Let's apply abstention to our calibrated breast-cancer classifier (`load_breast_cancer`) — AND add a critical equity check from Chapter 5.1: **does abstention defer one clinical subgroup (say, borderline small tumors vs. large tumors) more often than another?**

```{admonition} Predict before you run
:class: tip
When our model must decide **100%** of held-out biopsies, accuracy is roughly **96–97%**. If we let the model **abstain** whenever its confidence is below **90%** (deferring uncertain cases to a pathologist), what will happen to (1) **coverage** (share auto-decided) and (2) **accuracy on the auto-decided cases**?
```

```{code-cell} python
:label: ch53-abstention
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BLUE, VERMILION, TEAL, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)    # 1 = malignant

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)

proba = model.predict_proba(X_te)
preds = proba.argmax(axis=1)
conf  = proba.max(axis=1)                               # confidence in chosen class (0.5..1.0)
full_acc = (preds == y_te).mean()

# Sweep confidence thresholds from 0.50 up to 0.99
thresholds = np.linspace(0.50, 0.99, 45)
covs, accs = [], []
for t in thresholds:
    keep = conf >= t
    covs.append(keep.mean())
    accs.append((preds[keep] == y_te[keep]).mean() if keep.sum() > 0 else np.nan)

# Inspect one concrete clinical operating point: t = 0.92
t_op = 0.92
keep_op = conf >= t_op
print(f"Full automation (100% coverage) : accuracy = {full_acc:.1%} ({(preds != y_te).sum()} errors)")
print(f"Abstention at t = {t_op:.2f}          : coverage = {keep_op.mean():.1%} | "
      f"auto-accuracy = {(preds[keep_op] == y_te[keep_op]).mean():.1%} "
      f"| deferred to specialist = {(~keep_op).sum()}/{len(y_te)} cases")

# Subgroup coverage check: malignant vs benign cases!
print("\nPer-class deferral check at t = 0.92:")
for lbl, name in [(0, "Benign"), (1, "Malignant")]:
    m = y_te == lbl
    print(f"  {name:<10}: auto-decided = {keep_op[m].mean():.1%}, "
          f"deferred to human = {(~keep_op[m]).mean():.1%}")

fig, ax = plt.subplots(figsize=(7.4, 4.1))
ax.plot(thresholds * 100, np.array(covs) * 100, "o-", ms=3.5, lw=2.2, color=BLUE,
        label="Coverage (% auto-decided)")
ax.plot(thresholds * 100, np.array(accs) * 100, "s-", ms=3.5, lw=2.4, color=TEAL,
        label="Accuracy on auto-decided cases")
ax.axhline(full_acc * 100, color=VERMILION, ls="--", lw=1.4,
           label=f"Decide-everything baseline ({full_acc:.1%})")
ax.axvline(t_op * 100, color="#4D4D4D", ls=":", lw=1.3)
ax.set_xlabel("Confidence threshold (%)")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(55, 102)
ax.set_title("Defer the hardest ~14% to a specialist → 99.3% accuracy on the rest!")
ax.legend(frameon=False, loc="lower left", fontsize=8.5)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

Look at [](#ch53-abstention)! Letting the AI defer just the **13.5% most uncertain cases** (`23/171` biopsies) to a pathologist slashes errors from 8 down to 1 — raising auto-accuracy from `95.3%` to **`99.3%`**! Remember two vital caveats:
1. Abstention ONLY works if confidence scores are **well-calibrated** (Chapter 2.3). An overconfident model that says `"99.9% sure!"` when wrong won't abstain when it should!
2. You must audit **who gets deferred**: if an AI defers an under-represented patient group at 3× the rate of the majority group — and deferred patients wait weeks longer for review — abstention can accidentally worsen health disparities!

## Human factors in the clinic: why "human-in-the-loop" isn't magic

Putting a tired physician in front of an AI alert does not automatically guarantee safety. Real clinical studies reveal two opposite human traps:

::::{tab-set}

:::{tab-item} Automation Bias (Over-Trust)
{term}`Automation bias` happens when busy clinicians reflexively accept a wrong AI recommendation — even overriding their own accurate judgment! Studies in radiology and clinical LLM evaluation show that when an AI sounds fluent and authoritative, humans often stop checking carefully (especially late in a 12-hour shift!).
:::

:::{tab-item} Algorithmic Aversion & Alert Fatigue
The opposite extreme: if an early-warning AI fires **too many false alarms** (remember rare-disease PPV from Chapter 1.4!), exhausted nurses click *"Dismiss"* on every popup without reading it — missing the rare true crisis (**alert fatigue**). Or after watching an AI make one silly mistake, clinicians stop trusting it even where it genuinely excels (**algorithmic aversion**).
:::

::::

## The 5-stage clinical evidence ladder

How does a promising Python notebook actually earn the right to touch real patients? Passing a textbook test set is only **Step 1 of 5**!

```{mermaid}
flowchart LR
    S1["1. Retrospective<br/>Benchmark<br/>(Held-out historical data)"] --> S2["2. External<br/>Validation<br/>(New hospitals & scanners)"]
    S2 --> S3["3. Silent Shadow<br/>Deployment<br/>(Runs live, hidden from MDs)"]
    S3 --> S4["4. Prospective RCT<br/>(e.g., MASAI trial:<br/>proves real patient benefit)"]
    S4 --> S5["5. Post-Market<br/>Drift Monitoring<br/>(Track updates & safety)"]
```

## Governing health AI in 2025–2026: FDA, EU AI Act, and TRIPOD+AI

Because health AI affects human lives, three major global governance pillars now shape how tools reach hospitals:

- **FDA AI/ML-Enabled Medical Devices & PCCPs (United States):** Over a thousand AI-enabled medical devices have been authorized by the U.S. FDA (heavily concentrated in radiology, cardiology, and neurology). Historically, devices had to stay completely "locked" after approval. Under the FDA's finalized **Predetermined Change Control Plan ({term}`Predetermined Change Control Plan (PCCP)`)** framework, manufacturers can pre-specify *how* an AI model will safely retrain or update post-market — along with exact validation guardrails — so models can adapt to distribution shift without compromising safety.
- **The EU AI Act (Europe):** Classifies most diagnostic and clinical decision-support AI tools as **High-Risk AI systems**, requiring documented data governance, bias audits, human-oversight design, traceability logs, and Fundamental Rights Impact Assessments alongside GDPR and Medical Device Regulation (MDR).
- **TRIPOD+AI Reporting Standard [@collins2024tripod]:** An international medical-journal checklist requiring every clinical AI paper to transparently report data sources, handling of missing values, leak-free validation, calibration, subgroup fairness, and code availability!

## Your portable 8-point critical appraisal checklist

Carry this checklist into every news article, hospital pitch, research paper, or capstone project you ever encounter:

```{admonition} The TAIHLS 8-Point Health AI Appraisal Checklist
:class: important
1. **Clinical question & baseline:** What exact decision does this assist, and does it meaningfully beat a simple linear or standard-of-care baseline?
2. **Causal vs. predictive:** Does the claim confuse *association* with *causation* (confounders, Simpson's paradox)?
3. **Honest validation & leakage:** Were splits grouped by *patient*, and was the frozen model tested on **external hospitals / future years**?
4. **Right clinical metrics:** Beyond accuracy/AUROC, what are **sensitivity, specificity, AUPRC** (for rare diseases), and **calibration**?
5. **Shortcuts & failure modes:** Could the model exploit scanner tags, rulers, or proxy billing labels instead of real biology?
6. **Accuracy for whom?** Are metrics broken down across age, sex, race/ethnicity, skin tone, and hospital site?
7. **Privacy & consent:** How was patient data protected (consent, HIPAA/GDPR, de-identification, DP, or federated learning)?
8. **Human-in-the-loop & lifecycle:** Can the system **abstain** when uncertain, guard against automation bias, and monitor drift after launch?
```

```{admonition} Run a full clinical audit lab (Companion Colab Notebook 5)
:class: tip
Open [Companion Notebook 05](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/05-fairness-and-abstention-audit.ipynb) to run a complete end-to-end clinical audit combining subgroup sensitivity gaps, reliability diagrams, abstention coverage curves, and automated Model Card summaries!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.3.1 (Modify) — Finding the minimum threshold for 99.5% auto-accuracy
:class: hint
Using `thresholds`, `accs`, and `covs` from [](#ch53-abstention), write a 4-line loop to find the **lowest** confidence threshold achieving **≥ 99.5% auto-accuracy**. What percentage of biopsies can the AI still handle automatically at that strict safety bar?
```

```{admonition} Solution
:class: dropdown
```python
for t, acc, cov in zip(thresholds, accs, covs):
    if acc >= 0.995:
        print(f"Lowest threshold for >=99.5% auto-accuracy: t = {t:.2f} "
              f"-> auto-acc = {acc:.1%}, coverage = {cov:.1%}")
        break
```

At roughly `t ≈ 0.88–0.92`, the system reaches **~100% accuracy** while still automating **~86–90%** of routine scans — lifting massive workload off pathologists while funneling the riskiest ~10–14% straight to human specialists!
```

```{admonition} Exercise 5.3.2 (Make) — Checking whether errors concentrate in low-confidence cases
:class: hint
In [](#ch53-abstention), split `conf` into two groups: **correct predictions** (`preds == y_te`) vs. **incorrect predictions** (`preds != y_te`). Print the **median confidence** of each group. Why is a wide gap between those two numbers essential for abstention to work?
```

```{admonition} Solution
:class: dropdown
```python
print(f"Median confidence on CORRECT predictions  : {np.median(conf[preds == y_te]):.1%}")
print(f"Median confidence on INCORRECT predictions: {np.median(conf[preds != y_te]):.1%}")
```

Correct predictions cluster near `99.9%` confidence while errors cluster much lower (`~70–85%`). If a poorly calibrated model were just as overconfident on its mistakes (`99%`) as on its hits (`99%`), thresholding confidence would fail to catch the errors!
```

```{admonition} Exercise 5.3.3 (Critical Reasoning) — Silent shadow mode before clinical launch
:class: hint
Why do leading hospitals run a new AI sepsis alert in **"silent shadow mode"** (Stage 3 of the evidence ladder: running live on real incoming hospital streams for 3 months *without* displaying alerts to doctors yet) before turning popups on in the ER?
```

```{admonition} Solution
:class: dropdown
Silent shadow mode safely exposes real-world failures before a single patient is affected:
1. **IT & missing-data bugs:** Real hospital feeds arrive late, change lab unit codes, or drop vitals — breaking models that looked spotless on cleaned research spreadsheets.
2. **Real alert-volume audit:** Measures the true daily false-alarm rate (PPV!) on live patients so the hospital can tune the threshold *before* drowning ER nurses in alert fatigue!
3. **Subgroup & drift verification:** Confirms calibration and equity on *this* hospital's exact population.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Abstention (selective prediction)** defers low-confidence cases to humans — dramatically raising reliability on auto-decided cases when probabilities are calibrated and per-group deferral is audited.
- Real clinical safety must address human factors (**automation bias** and **alert fatigue**), not just algorithmic metrics.
- Reaching bedside care requires climbing the **5-stage evidence ladder** (ending in prospective trials like **MASAI** [@lang2026masai] and post-market monitoring) under modern governance (**FDA PCCPs**, **EU AI Act**, **TRIPOD+AI** [@collins2024tripod]).
- The **8-Point Critical Appraisal Checklist** turns everything in this book into a lifelong habit.
```

*Next up:* It is your turn to sit in the lead scientist's chair! Bring all five parts of TAIHLS together in your **Biomedical Capstone Project** in [](5_4-capstone-project.md).
