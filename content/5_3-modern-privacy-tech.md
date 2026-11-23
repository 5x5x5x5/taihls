---
title: 'Chapter 5.3 — Modern Privacy Shields: Differential Privacy and Federated Learning'
short_title: '5.3 · DP & federated learning'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.3 — Modern Privacy Shields: Differential Privacy and Federated Learning

*Week 14 · ~40 minutes · Prerequisites: [](5_2-anonymity-and-privacy.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **differential privacy** using the everyday "plausible deniability" random-noise idea;
- run a short simulation showing how adding calibrated random noise protects individuals while staying remarkably accurate on large hospitals;
- explain **federated learning** ("move the AI, not the patients!") using a simple diagram;
- simulate three separate hospitals training a shared breast-cancer classifier **without sharing a single patient row**.
```

## Start with a case: learning together without sharing secrets

Suppose three hospitals — a rural community clinic, a city hospital, and a university medical center — want to build a joint AI model to spot breast cancer accurately across diverse populations. There is just one rule: **privacy laws forbid sending raw patient records outside each hospital's firewall.**

How can science learn from all three hospitals without exposing a single patient? Modern health AI relies on two ingenious shields: **differential privacy** and **federated learning**.

## Shield 1: Differential privacy (hiding in tiny random noise)

Imagine your class wants to find out what fraction of students ever skipped sleep before a test — without *anyone* being embarrassed! Here is a famous privacy trick:
1. Before answering, each student privately flips a coin.
2. Heads → answer honestly. Tails → flip again and answer randomly!

Because of the coin flip, **any single person has total plausible deniability** — yet across 1,000 students, you can still calculate the true percentage accurately!

{term}`Differential privacy` [@dwork2014dp] works on the exact same principle for hospital statistics: add a tiny, carefully sized dash of random noise to a published count so that **whether YOUR record was in the hospital or deleted makes virtually zero detectable difference!**

The knob controlling how much noise gets added is the **privacy budget $\varepsilon$ (epsilon)**:
- **Tiny $\varepsilon$ (like `0.05`):** Bigger noise → ironclad privacy, fuzzier numbers.
- **Moderate $\varepsilon$ (like `1.0`):** Small noise → strong privacy *and* crisp hospital statistics!

```{admonition} Predict before you run
:class: tip
Suppose a hospital of **2,000 patients** has `340` patients (`17.0%`) with a condition, and we add differentially private random noise at `epsilon = 1.0` (typical wiggle of roughly ±1 person!). When divided by 2,000 patients, will that ±1-person wiggle distort `17.0%` by a lot (like ±5%) or barely at all (under ±0.1%)?
```

```{code-cell} python
:label: ch53-dp-sim
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
patients = 2_000
true_sick = 340
true_pct = true_sick / patients * 100

epsilons = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
typical_errors = []

for eps in epsilons:
    # Add random noise scaled by 1 / eps (1,000 simulated releases to see typical error)
    noise = rng.laplace(loc=0.0, scale=1.0 / eps, size=1000)
    noisy_pct = (true_sick + noise) / patients * 100
    err = float(np.median(np.abs(noisy_pct - true_pct)))
    typical_errors.append(err)
    if eps in [0.05, 0.5, 1.0, 5.0]:
        print(f"epsilon = {eps:>4.2f} -> typical error is only ±{err:.2f} percentage points!")

plt.figure(figsize=(7.0, 3.7))
plt.plot(epsilons, typical_errors, "o-", color="#0072B2", lw=2.3)
plt.xscale("log")
plt.xlabel("Privacy budget ε (smaller = stronger privacy)")
plt.ylabel("Typical error in reported % (percentage points)")
plt.title("Differential privacy on 2,000 patients: strong privacy with tiny error!")
plt.grid(alpha=0.25, which="both")
plt.show()
```

At $\varepsilon = 1.0$, the reported percentage stays within **$\pm 0.04$ percentage points** of the exact truth (`17.00%`) while mathematically protecting every single patient!

## Shield 2: Federated learning (send the AI to the hospital!)

Now back to our three hospitals! Instead of copying every patient's chart to one giant cloud server, {term}`Federated learning` [@rieke2020federated] keeps **100% of raw patient records locked inside each hospital**. Each hospital trains a local model on its own patients and sends **only the learned weights** (just a few numbers — zero patient rows!) to be averaged together!

```{mermaid}
flowchart TD
    Hub["Central Coordinator<br/>Averages ONLY model weights<br/>(NEVER sees raw patient rows!)"]
    H1["Hospital 1<br/>Private patients → Weights W₁"]
    H2["Hospital 2<br/>Private patients → Weights W₂"]
    H3["Hospital 3<br/>Private patients → Weights W₃"]
    H1 & H2 & H3 -- "Send weights only" --> Hub
    Hub -- "Return shared AI" --> H1 & H2 & H3
```

Let's simulate **three isolated hospitals** collaborating on real breast-cancer biopsies (`load_breast_cancer`) without sharing a single patient row!

```{code-cell} python
:label: ch53-federated
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)

X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_pool)
X_pool_s, X_test_s = scaler.transform(X_pool), scaler.transform(X_test)

# 1. Centralized model (all raw data pooled together)
central = LogisticRegression(max_iter=1000).fit(X_pool_s, y_pool)

# 2. Split training patients across 3 isolated hospitals (NO raw rows leave any hospital!)
hosp_X = np.array_split(X_pool_s, 3)
hosp_y = np.array_split(y_pool, 3)
local_models = [LogisticRegression(max_iter=1000).fit(Xh, yh) for Xh, yh in zip(hosp_X, hosp_y)]

# 3. Federated model: average ONLY the learned weights and intercepts!
fed = LogisticRegression()
fed.classes_   = np.array([0, 1])
fed.coef_      = np.mean([m.coef_      for m in local_models], axis=0)
fed.intercept_ = np.mean([m.intercept_ for m in local_models], axis=0)

for i, m in enumerate(local_models, 1):
    print(f"Hospital {i} alone             : {m.score(X_test_s, y_test):.1%}")
print(f"Federated (weights ONLY!)    : {fed.score(X_test_s, y_test):.1%}  <-- matches pooled data!")
print(f"Centralized (raw data pooled): {central.score(X_test_s, y_test):.1%}")
```

Averaging **only the learned weights** across three hospitals — with **zero** raw patient records ever leaving any hospital — jumps to **`97.2%`**, virtually matching the centralized model!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.3.1 (Modify) — Tiny clinic vs. big hospital under differential privacy
:class: hint
In [](#ch53-dp-sim), keep `eps = 1.0`, and compare the typical percentage error when `patients = 50` (`true_sick = 9`) versus `patients = 20_000` (`true_sick = 3_400`). Why is differential privacy almost "free" on huge hospitals yet noisy on tiny clinics?
```

```{admonition} Solution
:class: dropdown
```python
for n_p, sick in [(50, 9), (2_000, 340), (20_000, 3_400)]:
    noise = rng.laplace(0.0, 1.0, size=2000)
    err = np.median(np.abs((sick + noise) / n_p * 100 - sick / n_p * 100))
    print(f"n = {n_p:>6,}: typical DP error at ε=1.0 is ±{err:.3f} percentage points")
```
Adding ~1 person's worth of noise is huge out of `50` (`±1.4%`), yet tiny out of `20,000` (`±0.003%`)! Hiding inside a huge crowd is almost effortless.
```

```{admonition} Exercise 5.3.2 (Make) — Split across 5 smaller hospitals
:class: hint
In [](#ch53-federated), change `3` to `5` (`np.array_split(..., 5)`) so five smaller clinics collaborate via federated averaging. Does the federated model still beat the weakest single clinic?
```

```{admonition} Solution
:class: dropdown
```python
hX5, hy5 = np.array_split(X_pool_s, 5), np.array_split(y_pool, 5)
mods5 = [LogisticRegression(max_iter=1000).fit(Xh, yh) for Xh, yh in zip(hX5, hy5)]
fed5 = LogisticRegression()
fed5.classes_ = np.array([0, 1])
fed5.coef_ = np.mean([m.coef_ for m in mods5], axis=0)
fed5.intercept_ = np.mean([m.intercept_ for m in mods5], axis=0)
print("5 clinics alone:", [f"{m.score(X_test_s, y_test):.1%}" for m in mods5])
print(f"5-clinic Federated: {fed5.score(X_test_s, y_test):.1%}")
```
Yes! Pooling only the weights smooths out each small clinic's noise and stays right near **~97%**!
```

```{admonition} Exercise 5.3.3 (Critical Reasoning) — Combining BOTH privacy shields!
:class: hint
Even when hospitals keep raw patient rows inside their own walls using federated learning, clever attackers can sometimes inspect shared neural-network weights to guess whether a rare patient was in training. How does combining **federated learning + differential privacy** plug that last hole?
```

```{admonition} Solution
:class: dropdown
Each hospital adds a tiny dash of **differential-privacy noise** to its weight updates *before* sending them to the central coordinator! That gives double protection: raw patient files never leave the hospital, *and* the shared numbers mathematically mask every single individual.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Differential privacy** adds small, calibrated random noise (controlled by budget **$\varepsilon$**) to hide any individual's presence while preserving accuracy on large cohorts [@dwork2014dp].
- **Federated learning** lets hospitals train shared AI by exchanging **only model weights**, keeping raw patient files locked inside each hospital [@rieke2020federated].
```

*Next up:* Even with fair, private models, what should an AI do when it meets a blurry, confusing patient case? Let's teach models to say **"I'm not sure — ask a doctor!"** in [](5_4-humans-in-the-loop.md)!
