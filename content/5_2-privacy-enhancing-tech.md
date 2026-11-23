---
title: 'Chapter 5.2 — Protecting Patients: Re-Identification, Differential Privacy, and Federated Learning'
short_title: '5.2 · Privacy-enhancing tech'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.2 — Protecting Patients: Re-Identification, Differential Privacy, and Federated Learning

*Estimated time: ~50 minutes · Prerequisites: [](2_3-clinical-metrics-and-thresholds.md), [](5_1-algorithmic-fairness.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- demonstrate why deleting names is *not* enough: **quasi-identifiers** enable **re-identification** attacks — and how **k-anonymity** mitigates (yet does not completely eliminate) them;
- implement **differential privacy** (Laplace noise) in 10 lines of Python and plot the **privacy budget ($\varepsilon$) vs. utility** curve;
- simulate **federated learning** across three separate hospitals so models learn jointly **without raw patient rows ever leaving any hospital**;
- explain the roles of informed **consent**, **HIPAA** (including "Shadow AI" risks), **GDPR**, and **synthetic data** audits.
```

## Start with a case: the governor's "anonymous" hospital record

In the late 1990s, a state health commission released a "de-identified" database of hospital visits for medical researchers — dutifully stripping out every patient's name, address, and Social Security number. Yet it kept each patient's **ZIP code, birth date, and sex**.

Graduate student Latanya Sweeney bought the public voter-registration list for Cambridge, Massachusetts for $20, matched just those three columns against the hospital table, and mailed the state's own governor his private medical diagnoses and prescriptions [@sweeney2002kanonymity]. Her landmark analysis showed that **5-digit ZIP + exact birth date + sex uniquely identify roughly 87% of the U.S. population**!

In healthcare AI, we face a deep tension: **diverse, large-scale clinical data saves lives, while leaked health records ruin lives.** Modern medicine resolves this tension with a layered toolkit of **Privacy-Enhancing Technologies (PETs)**: *k-anonymity*, *differential privacy*, and *federated learning*.

## Lab 1: Quasi-identifiers, re-identification, and k-anonymity

```{admonition} Definition — quasi-identifier, re-identification, and k-anonymity
:class: important
- A {term}`Quasi-identifier` is an attribute (age, ZIP code, sex) that is not unique alone, yet singles out a person when combined with others — enabling {term}`Re-identification` via linkage to external public tables.
- A table achieves {term}`k-Anonymity` if every combination of quasi-identifiers is shared by **at least $k$ distinct people** [@sweeney2002kanonymity].
```

```{code-cell} python
:label: ch52-kanon
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 30
patients = pd.DataFrame({
    "name": [f"Patient_{i+1:02d}" for i in range(n)],           # direct identifier (PII)
    "age": rng.integers(22, 74, n),                             # quasi-identifier
    "zip": rng.choice(["02138", "02139", "02140"], n),          # quasi-identifier
    "sex": rng.choice(["F", "M"], n),                           # quasi-identifier
    "diagnosis": rng.choice(["asthma", "diabetes", "migraine", "hypertension"], n),
})

# Step 1: Naive "de-identification" (drop names only)
naive = patients.drop(columns="name")
qi = ["age", "zip", "sex"]
n_unique = (~naive.duplicated(subset=qi, keep=False)).sum()
print(f"After dropping names ONLY : {n_unique}/{n} patients are STILL uniquely re-identifiable!")

# Step 2: Generalize quasi-identifiers into broader buckets to reach k-anonymity
gen = naive.copy()
gen["age"] = np.where(gen["age"] < 45, "20–44", "45–75")
gen["zip"] = "021**"
group_sizes = gen.groupby(qi).size()
print(f"After generalizing        : smallest crowd has k = {group_sizes.min()} identical patients -> {group_sizes.min()}-anonymous!")
```

:::{margin}
**Limits of k-anonymity:** Even if 5 patients share `(20–44, 021**, F)`, if all 5 happen to have `diagnosis = "diabetes"`, an attacker who spots the group *still* learns their diagnosis (**homogeneity attack**)! That weakness drove computer scientists to invent a mathematically airtight guarantee: **differential privacy**.
:::

## Lab 2: Differential privacy — protecting every patient with calibrated noise

Suppose a hospital wants to publish a summary statistic — say, *"What fraction of our 1,000 cardiology patients carry a rare genetic risk factor?"* Even aggregated counts can leak secrets if an attacker compares counts *before* and *after* one patient joins the clinic!

{term}`Differential privacy` [@dwork2014dp] solves this with a brilliantly simple guarantee: add a small amount of mathematically calibrated **Laplace random noise** to the answer so that **whether YOUR record is in the database or deleted from it makes virtually zero detectable difference to the output!**

The knob controlling the strength of that guarantee is the **privacy budget $\varepsilon$ (epsilon)**:
- **Small $\varepsilon$ (e.g., `0.1`):** Huge noise → ironclad privacy, fuzzier hospital statistics.
- **Moderate $\varepsilon$ (e.g., `1.0`):** Modest noise → strong mathematical privacy *and* accurate population insights!
- **Huge $\varepsilon$ (e.g., `20.0`):** Tiny noise → near-exact numbers, weak privacy guarantee.

For a simple count of `n` patients (changing one patient changes the true count by at most `sensitivity = 1`), the Laplace noise scale is simply `1 / epsilon`!

```{admonition} Predict before you run
:class: tip
Suppose out of `2,000` hospital patients, the true count carrying a biomarker is `340` (`17.0%`). If we publish that percentage with differential privacy at $\varepsilon = 0.05$ vs. $\varepsilon = 1.0$, roughly how many percentage points off from `17.0%` will each noisy answer typically land?
```

```{code-cell} python
:label: ch52-diff-privacy
import matplotlib.pyplot as plt

BLUE, VERMILION, TEAL = "#0072B2", "#D55E00", "#009E73"
rng_dp = np.random.default_rng(7)

n_cohort = 2_000
true_count = 340                                    # true prevalence = 17.0%
true_pct = true_count / n_cohort * 100

epsilons = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
median_abs_err = []

print(f"True cohort prevalence: {true_pct:.2f}% ({true_count}/{n_cohort} patients)")
for eps in epsilons:
    # Run 1,000 independent differentially private releases to measure typical error
    noise = rng_dp.laplace(loc=0.0, scale=1.0 / eps, size=1000)
    dp_pcts = np.clip((true_count + noise) / n_cohort * 100, 0, 100)
    mae = float(np.median(np.abs(dp_pcts - true_pct)))
    median_abs_err.append(mae)
    if eps in (0.05, 0.5, 1.0, 5.0):
        print(f"  epsilon = {eps:>4.2f} -> typical error = ±{mae:.2f} percentage points "
              f"(sample release: {dp_pcts[0]:.2f}%)")

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.plot(epsilons, median_abs_err, "o-", lw=2.4, color=BLUE)
ax.set_xscale("log")
ax.set_xlabel("Privacy budget ε (smaller ε = stronger privacy)")
ax.set_ylabel("Typical error in prevalence (percentage points)")
ax.set_title("Differential privacy: the privacy–utility trade-off (n = 2,000 patients)")
ax.annotate("Strong privacy,\nmodest ±0.04% error!", xy=(1.0, median_abs_err[4]),
            xytext=(1.15, 0.45), arrowprops=dict(arrowstyle="->", color=TEAL),
            fontsize=9, color=TEAL, fontweight="bold")
ax.grid(alpha=0.25, which="both")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

Look how magical that curve in [](#ch52-diff-privacy) is! On a 2,000-patient cohort, setting a strong privacy budget of $\varepsilon = 1.0$ adds only **$\pm 0.04$ percentage points** of error — completely preserving scientific usefulness while mathematically shielding every individual patient's presence in the cohort! (And when datasets are tiny — say, 20 patients — the exact same `1 / epsilon` noise becomes huge, automatically warning you that releasing stats on tiny groups risks privacy!)

## Lab 3: Federated learning — bring the AI to the hospital!

What if three hospitals — say, a rural clinic, a city cancer center, and a children's hospital — want to build a joint diagnostic model, **yet privacy laws forbid pooling their raw patient records into one central cloud**?

{term}`Federated learning` [@rieke2020federated] flips the architecture upside down: **never move the patient data — move the model instead!**

```{mermaid}
flowchart TD
    Coord["Central Coordinator<br/>Averages ONLY model weights<br/>(NEVER sees a single patient!)"]
    H1["Hospital 1<br/>Local private data<br/>trains local weights W₁"]
    H2["Hospital 2<br/>Local private data<br/>trains local weights W₂"]
    H3["Hospital 3<br/>Local private data<br/>trains local weights W₃"]
    Coord -- "Send global weights" --> H1 & H2 & H3
    H1 & H2 & H3 -- "Return weights only" --> Coord
```

Let's simulate **real Federated Averaging (FedAvg)** on our Wisconsin Breast Cancer biopsies split across **three separate hospitals**, where NO raw patient data ever leaves any hospital!

```{admonition} Predict before you run
:class: tip
We hold out 20% of biopsies as an external test clinic, and divide the remaining training patients across **3 independent hospitals** (each sees only ~1/3 of the data!). If each hospital trains its own local Logistic Regression model and sends **only its 30 learned weights + 1 intercept** to be averaged into a **Federated Model**, will the Federated Model outperform the isolated single-hospital models and rival a Centralized model that pooled all raw data?
```

```{code-cell} python
:label: ch52-federated
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)

X_pool, X_ext, y_pool, y_ext = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
# Shared standardization using agreed clinical reference ranges (computed on pool for clean demo)
scaler = StandardScaler().fit(X_pool)
X_pool_s, X_ext_s = scaler.transform(X_pool), scaler.transform(X_ext)

# 1. Centralized benchmark (all raw data pooled in one place)
central = LogisticRegression(max_iter=1000).fit(X_pool_s, y_pool)

# 2. Split training patients across 3 isolated hospitals (NO raw rows shared!)
splits_X = np.array_split(X_pool_s, 3)
splits_y = np.array_split(y_pool, 3)

local_models = [LogisticRegression(max_iter=1000).fit(Xh, yh)
                for Xh, yh in zip(splits_X, splits_y)]

# 3. Federated Averaging (FedAvg): average ONLY weights & intercepts!
fed_model = LogisticRegression()
fed_model.classes_ = np.array([0, 1])
fed_model.coef_ = np.mean([m.coef_ for m in local_models], axis=0)
fed_model.intercept_ = np.mean([m.intercept_ for m in local_models], axis=0)

print(f"{'Training Strategy':<28} | {'Test Acc':>8} | {'Test AUROC':>10}")
print("-" * 52)
for i, m in enumerate(local_models, 1):
    print(f"Hospital {i} alone (isolated)  | {m.score(X_ext_s, y_ext):8.1%} | "
          f"{roc_auc_score(y_ext, m.predict_proba(X_ext_s)[:, 1]):10.3f}")
print(f"Federated (weights ONLY!)    | {fed_model.score(X_ext_s, y_ext):8.1%} | "
      f"{roc_auc_score(y_ext, fed_model.predict_proba(X_ext_s)[:, 1]):10.3f}")
print(f"Centralized (raw data pooled)| {central.score(X_ext_s, y_ext):8.1%} | "
      f"{roc_auc_score(y_ext, central.predict_proba(X_ext_s)[:, 1]):10.3f}")
```

**Look at that result:** Averaging ONLY the 31 learned numbers from the three hospitals — without moving a single patient biopsy across a hospital firewall! — matches the centralized model's **97%+ accuracy and 0.995 AUROC**! In real multi-center deep learning (like international brain-tumor MRI consortia [@rieke2020federated]), hospitals repeat this weight-averaging loop across multiple rounds and often pair it with **differential privacy** so even the shared weights cannot memorize individual patients!

## Laws, consent, and modern healthcare privacy traps

Technical PETs sit inside a legal and ethical framework built on **informed consent** — patients deserve to know how their bodies and records are used and have the right to decline:

::::{tab-set}

:::{tab-item} HIPAA (United States)
Protects **Protected Health Information (PHI)**. Permits sharing de-identified data via *Safe Harbor* (removing 18 explicit identifiers) or *Expert Determination*. **Major 2025–2026 hazard — "Shadow AI":** Typing a real patient's chart into a consumer chatbot without a signed Business Associate Agreement (BAA) is a serious HIPAA violation!
:::

:::{tab-item} GDPR (European Union)
Treats health data as a special protected category requiring explicit lawful basis, data minimization, and the right to withdraw consent — enforced alongside the **EU AI Act** (Chapter 5.3).
:::

:::{tab-item} Synthetic Data (Promise & Caution)
Generative AI can synthesize artificial patient tables mimicking real clinical correlations without 1-to-1 patient records. **Caution:** Overfit generators can still accidentally **memorize and copy** rare real patients — so synthetic data must *always* be audited for leakage and demographic bias before release!
:::

::::

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.2.1 (Modify) — How cohort size helps differential privacy
:class: hint
In [](#ch52-diff-privacy), change `n_cohort` from `2_000` down to `50` (with `true_count = 9`, ~18% prevalence) and up to `20_000` (`true_count = 3_400`, 17%). Keep $\varepsilon = 1.0$. Compare the typical percentage-point error across `n = 50`, `2_000`, and `20_000`. Why does differential privacy give a "nearly free lunch" on large hospital networks yet heavy noise on tiny clinics?
```

```{admonition} Solution
:class: dropdown
```python
for n_c, tc in [(50, 9), (2_000, 340), (20_000, 3_400)]:
    noise = rng_dp.laplace(0.0, 1.0 / 1.0, size=2000)
    err = np.median(np.abs((tc + noise) / n_c * 100 - tc / n_c * 100))
    print(f"n = {n_c:>6,}: typical DP error at ε=1.0 is ±{err:.3f} percentage points")
```

Because the absolute noise needed to hide **one single person** stays constant (`scale = 1 / ε` people), dividing that ~1-person noise by `50` shifts the rate by `~1.4%`, whereas dividing by `20,000` shifts it by only `~0.003%`! Hiding in a huge crowd is effortless; hiding in a tiny room requires loud static.
```

```{admonition} Exercise 5.2.2 (Make) — Weighted Federated Averaging when hospitals differ in size
:class: hint
Suppose Hospital 1 is a large center with 300 patients while Hospitals 2 and 3 are small clinics with 60 patients each (`n_sizes = [300, 60, 60]`). Instead of an unweighted `np.mean(...)`, write 3 lines of Python showing how **weighted FedAvg** combines `m.coef_` proportionally to each hospital's sample size (`w_h = n_h / sum(n_sizes)`). Why is size-weighting better than unweighted averaging?
```

```{admonition} Solution
:class: dropdown
```python
n_sizes = np.array([len(yh) for yh in splits_y], dtype=float)
weights = n_sizes / n_sizes.sum()
fed_coef_w = sum(w * m.coef_ for w, m in zip(weights, local_models))
fed_int_w  = sum(w * m.intercept_ for w, m in zip(weights, local_models))
print("Weighted FedAvg weight vector shape:", fed_coef_w.shape)
```

If a tiny 30-patient clinic got equal weight to a 3,000-patient center, its noisy local estimates would degrade the global model. Weighting updates proportionally to patient count (the actual **FedAvg** algorithm!) balances stability with cross-site diversity.
```

```{admonition} Exercise 5.2.3 (Critical Reasoning) — Can an attacker still learn things from a federated model?
:class: hint
Even when raw patient rows stay inside each hospital under federated learning, privacy researchers discovered that **model weights and gradients themselves can sometimes leak whether a specific rare patient was in the training set** (*membership inference*!). Which *second* technology from this chapter can be layered on top of federated learning to block that leak?
```

```{admonition} Solution
:class: dropdown
**Differential privacy!** By clipping each hospital's weight update and adding calibrated random noise *before* sending it to the central coordinator (**DP-FedAvg**), the system combines **data localization** (raw rows never leave the hospital) with a **mathematical bound** on how much any single patient can influence the shared model weights!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Removing names is never true anonymization: **quasi-identifiers** (age, ZIP, sex) re-identify most people; **k-anonymity** hides individuals in crowds of at least $k$ [@sweeney2002kanonymity].
- **Differential privacy** adds calibrated Laplace noise controlled by privacy budget **$\varepsilon$**, mathematically guaranteeing that any single patient's participation is undetectable [@dwork2014dp].
- **Federated learning** lets hospitals collaboratively train shared AI models by exchanging **only model weights**, keeping raw patient records inside each hospital firewall [@rieke2020federated].
- Privacy protection combines technical PETs, informed **consent**, legal safeguards (**HIPAA**, **GDPR**), and zero tolerance for pasting PHI into consumer "Shadow AI" tools.
```

*Next up:* You've built accurate, fair, privacy-respecting models — now how do we actually **deploy** AI alongside busy doctors and nurses safely under real-world **FDA and EU regulation**? Let's walk from prototype to bedside in [](5_3-humans-regulation-deployment.md).
