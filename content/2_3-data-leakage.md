---
title: 'Chapter 2.3 — How Smart Scientists Fool Themselves: Data Leakage'
short_title: '2.3 · Data leakage'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.3 — How Smart Scientists Fool Themselves: Data Leakage

*Week 5 · ~35 minutes · Prerequisites: [](2_2-train-test-and-overfitting.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what **data leakage** is and why it creates wildly fake accuracy scores;
- reproduce **patient-level data leakage** in Python and watch a model score **>95% on pure random coin flips**;
- fix the bug by splitting strictly by **patient ID**;
- spot three common forms of clinical leakage (repeated visits, preprocessing before splitting, and future clues).
```

## Start with a case: the 98% accurate lab model that crashed on Day 1

A research team collects **600 blood-test visits** from **100 patients** (six visits per patient). They randomly shuffle all 600 rows, put 75% into training and 25% into testing, and train a classifier. Test accuracy hits **97%**!

Excited, they test the exact same model on **25 brand-new patients** at a clinic across town. Accuracy collapses to **52%** — no better than flipping a coin!

What went wrong? Look at the shuffling step: because they shuffled **visits** instead of **patients**, four visits from **Patient #42** went into training while the other two visits from **Patient #42** went into the test set! The model never learned disease biology — it simply recognized **Patient #42's personal baseline numbers** and looked up their label!

```{admonition} Definition — data leakage and external validation
:class: important
- {term}`Data leakage` happens whenever clues from the test exam accidentally sneak into training — producing fake, sky-high lab scores that vanish in real life.
- {term}`External validation` means testing the finished model on brand-new patients from a different clinic, scanner, or year [@collins2024tripod].
```

## Watch a model score ~97% on PURE RANDOM NOISE!

Let's build this exact scandal so you can see it with your own eyes!
- We create **100 fake patients**, each with **6 visits** (`600` rows total).
- Every patient has a unique personal baseline (like resting heart rate and typical lab levels) that stays fairly steady across their 6 visits.
- **The twist:** We assign each patient's disease label (`0` or `1`) by **pure random coin flip!** There is **ZERO real biological pattern** — so any honest model on brand-new patients *must* score around **50%**!

```{admonition} Predict before you run
:class: tip
Because the labels are pure coin flips, the true best possible accuracy on brand-new patients is **50%**.
1. When we carelessly split the **600 visit rows** randomly (putting the same patients in train and test), roughly what accuracy will the model report?
2. When we split cleanly by **patient ID** (Patients 0–74 in train, Patients 75–99 in test), what accuracy will it report?
```

```{code-cell} python
:label: ch23-leakage-demo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(123)
n_patients = 100
visits_each = 6

# Assign each patient an ID (0..99) and a PURE COIN-FLIP label (no real signal!)
patient_id = np.repeat(np.arange(n_patients), visits_each)
coin_flip_labels = rng.integers(0, 2, size=n_patients)
y = np.repeat(coin_flip_labels, visits_each)

# Each patient has a steady personal baseline across their 6 visits (+ tiny daily wiggle)
personal_baseline = rng.normal(0, 2.0, size=(n_patients, 10))
X = np.repeat(personal_baseline, visits_each, axis=0) + rng.normal(0, 0.35, size=(n_patients * visits_each, 10))

# MISTAKE (Leaky!): shuffle visit rows randomly — same patients end up in BOTH sets!
X_tr_bad, X_te_bad, y_tr_bad, y_te_bad = train_test_split(X, y, test_size=0.25, random_state=0)
acc_leaky = RandomForestClassifier(random_state=0).fit(X_tr_bad, y_tr_bad).score(X_te_bad, y_te_bad)

# HONEST SPLIT: Patients 0..74 train, Patients 75..99 test (zero overlap!)
train_mask = patient_id < 75
test_mask  = patient_id >= 75
acc_honest = RandomForestClassifier(random_state=0).fit(X[train_mask], y[train_mask]).score(X[test_mask], y[test_mask])

print(f"True possible accuracy on new patients : 50.0% (pure coin flip!)")
print(f"Leaky row split (WRONG!)               : {acc_leaky:.1%}  <-- COMPLETELY FAKE!")
print(f"Patient-ID split (HONEST)              : {acc_honest:.1%}  <-- tells the truth!")

plt.figure(figsize=(6.8, 3.6))
bars = plt.bar(["True ceiling\n(coin flip)", "Honest split\n(by patient ID)", "Leaky split\n(by visit row!)"],
               [50, acc_honest * 100, acc_leaky * 100],
               color=["#4D4D4D", "#009E73", "#D55E00"], width=0.52)
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontweight="bold")
plt.ylim(0, 112)
plt.ylabel("Reported test accuracy (%)")
plt.title("Data leakage manufactures ~97% accuracy out of pure random noise!")
plt.grid(axis="y", alpha=0.25)
plt.show()
```

**97.3% accuracy on pure random coin flips!!** That single experiment is one of the most important things you will learn all semester. Any time a medical dataset has multiple photos, X-rays, or clinic visits per person, **always split by patient ID, never by row!**

## Three sneaky leaks to watch out for

::::{tab-set}

:::{tab-item} 1. Same Patient in Train & Test
Putting three skin photos or five hospital visits from the same person into training and one into testing (what we just simulated!). **Fix:** Group splits by `patient_id`.
:::

:::{tab-item} 2. Future Clues ("Target Leakage")
Using a column recorded *after* the doctor already knew the answer — like using *"sent to cancer ward"* as an input to predict cancer! **Fix:** Only allow measurements collected *before* the prediction moment.
:::

:::{tab-item} 3. Scaling Before Splitting
Computing column averages across the *entire* spreadsheet before separating training and test data (Chapter 2.1). **Fix:** Compute scaling statistics on training data only!
:::

::::

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.3.1 (Modify) — What if each patient's visits varied wildly?
:class: hint
In [](#ch23-leakage-demo), daily visit wiggle is `0.35`. Raise it to `3.5` and re-run both the leaky split and the honest split. Why does the fake leaky score drop toward 50% once daily noise hides each patient's personal baseline?
```

```{admonition} Solution
:class: dropdown
```python
X_noisy = np.repeat(personal_baseline, visits_each, axis=0) + rng.normal(0, 3.5, size=(n_patients * visits_each, 10))
X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_noisy, y, test_size=0.25, random_state=0)
print(f"Leaky accuracy with high daily noise : {RandomForestClassifier(random_state=0).fit(X_tr_b, y_tr_b).score(X_te_b, y_te_b):.1%}")
print(f"Honest accuracy with high daily noise: {RandomForestClassifier(random_state=0).fit(X_noisy[train_mask], y[train_mask]).score(X_noisy[test_mask], y[test_mask]):.1%}")
```
When huge daily noise obscures each person's unique fingerprint, the model can no longer recognize *who* each test visit belongs to — so even the leaky score falls back toward the true 50% coin-flip rate!
```

```{admonition} Exercise 2.3.2 (Make) — Check for overlap with Python `set`
:class: hint
Write a 2-line safety check using Python `set(...)` that verifies **zero patient IDs appear in both** `patient_id[train_mask]` and `patient_id[test_mask]`.
```

```{admonition} Solution
:class: dropdown
```python
overlap = set(patient_id[train_mask]) & set(patient_id[test_mask])
print("Shared patients between train and test:", len(overlap))   # -> 0!
```
Adding `assert len(overlap) == 0` to every clinical AI script guarantees patient-level leakage can never sneak past you!
```

```{admonition} Exercise 2.3.3 (Critical Reasoning) — Spot the future leak!
:class: hint
A team builds an AI to predict **which emergency-room patients will need emergency surgery**. Looking at their input table, you spot two columns: (a) `arrival_blood_pressure` and (b) `operating_room_booked_time`. Which column is valid and which is target leakage? Why?
```

```{admonition} Solution
:class: dropdown
- `arrival_blood_pressure` is **valid** — measured right when the patient walks in.
- `operating_room_booked_time` is **severe future leakage** — a nurse only books an operating room *after* the surgeon has already decided on emergency surgery!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Data leakage** lets answers from the test set sneak into training, manufacturing fake accuracy (even **97%+ on pure coin flips!**).
- Whenever a dataset has multiple visits, photos, or scans per person, **always split by patient ID**.
- Watch out for **future clues** (columns recorded after the diagnosis) and **scaling before splitting**.
- Real clinical trust requires **external validation** on brand-new patients at independent clinics [@collins2024tripod].
```

*Next up:* Even with a 100% honest, leak-free split, plain **accuracy** can still hide life-or-death errors in medicine. Let's learn how doctors count mistakes by type and tune risk **thresholds** in [](2_4-mistakes-and-thresholds.md)!
