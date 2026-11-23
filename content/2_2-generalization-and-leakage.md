---
title: 'Chapter 2.2 — Honest Evaluation: Generalization, Overfitting, and Data Leakage'
short_title: '2.2 · Generalization & leakage'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 2.2 — Honest Evaluation: Generalization, Overfitting, and Data Leakage

*Estimated time: ~45 minutes · Prerequisites: [](2_1-nearest-neighbors-and-features.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- distinguish **underfitting**, good **generalization**, and **overfitting** on training vs. validation curves;
- explain why tuning hyperparameters requires a separate **validation set** or **k-fold cross-validation**, leaving the **test set** untouched;
- expose **patient-level data leakage** in runnable code and see how it creates wildly fake accuracy;
- explain why clinical guidelines like **TRIPOD+AI** demand **external validation** across hospitals.
```

## Start with a case: the 99% accurate model that failed on Day 1

A research team collects **1,200 pathology slides** from **200 patients** (six slides photographed from different angles per patient). They randomly shuffle all 1,200 slides, put 80% into training and 20% into testing, and train a classifier. Test accuracy hits **98.5%**! The paper makes headlines.

Six months later, a neighboring hospital tries the exact same model on 200 brand-new patients. Accuracy plunges to **64%** — barely better than guessing.

What went wrong? Nothing broke in the math — the *evaluation* was rigged by accident. By shuffling **slides** instead of **patients**, five photos of Patient #47 sat in the training set while the sixth photo of Patient #47 sat in the test set! The model didn't learn generalizable cancer biology; it memorized *Patient #47's unique tissue background* and recognized them on the test exam.

In machine learning, building a model is easy. **Grading it honestly** is the true art of science.

## Learning vs. memorizing: overfitting

```{admonition} Definition — generalization and overfitting
:class: important
{term}`Generalization` is a model's ability to predict accurately on **new, unseen patients**. {term}`Overfitting` happens when a model becomes so flexible that it memorizes random quirks and noise in the {term}`Training set` — driving training accuracy toward 100% while performance on fresh patients drops.
```

To watch overfitting happen in slow motion, let's use a {term}`Decision tree` — a clinical flowchart of if/then questions (*"Is worst radius > 16.8?"*) whose complexity knob is its **maximum depth** (`max_depth`).

And notice a crucial discipline: if we test 15 different tree depths directly on our final test set and brag about the best one, we just peeked at the final exam! Instead, disciplined scientists use a **three-way split** or {term}`Cross-validation`:

```{mermaid}
flowchart LR
    All["All Patients<br/>(Split by PATIENT ID!)"] --> Dev["Development Set (80%)<br/>Used for training & 5-fold<br/>cross-validation tuning"]
    All --> Test["Locked Test Vault (20%)<br/>Touched ONCE at the very end"]
    Dev --> Final["Selected Model<br/>(Best complexity setting)"]
    Final --> Test
```

## Lab 1: Watching overfitting with 5-fold cross-validation

In **5-fold cross-validation**, we slice the training data into 5 equal parts, train on 4 parts, grade on the 5th, and rotate 5 times so every patient gets a turn as validation. Averaging those 5 scores gives a much steadier estimate of generalization than any single lucky split!

```{admonition} Predict before you run
:class: tip
We will train decision trees of `max_depth` from `1` up to `12` on all 30 Wisconsin breast cancer features. As depth grows from `1` → `12`, will **training accuracy** keep climbing all the way to 100%? Will **5-fold cross-validation accuracy** keep climbing too, or peak early and sag?
```

```{code-cell} python
:label: ch22-overfit-cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)   # 1 = malignant

# Lock away 20% as an untouched final test set
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
depths = list(range(1, 13))
train_scores, cv_means, cv_stds = [], [], []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0)
    tree.fit(X_dev, y_dev)
    train_scores.append(tree.score(X_dev, y_dev))
    fold_scores = cross_val_score(tree, X_dev, y_dev, cv=cv)
    cv_means.append(fold_scores.mean())
    cv_stds.append(fold_scores.std())

best_depth = depths[int(np.argmax(cv_means))]
final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=0).fit(X_dev, y_dev)

print(f"Best depth chosen via 5-fold CV : max_depth = {best_depth} "
      f"(CV acc = {max(cv_means):.1%})")
print(f"Overfit depth-12 tree           : train = {train_scores[-1]:.1%}, "
      f"CV = {cv_means[-1]:.1%}")
print(f"Locked test vault (touched ONCE): {final_tree.score(X_test, y_test):.1%}")
```

Now let's plot the classic **complexity curve** — with shaded $\pm 1$ standard-deviation uncertainty bands across the 5 cross-validation folds!

```{code-cell} python
:label: ch22-overfit-fig
BLUE, VERMILION, SLATE = "#0072B2", "#D55E00", "#4D4D4D"
cv_means_arr, cv_stds_arr = np.array(cv_means), np.array(cv_stds)

fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.plot(depths, np.array(train_scores) * 100, "o-", lw=2.2, color=BLUE, label="Training accuracy")
ax.plot(depths, cv_means_arr * 100, "s-", lw=2.4, color=VERMILION, label="5-fold CV accuracy (mean)")
ax.fill_between(depths,
                (cv_means_arr - cv_stds_arr) * 100,
                (cv_means_arr + cv_stds_arr) * 100,
                color=VERMILION, alpha=0.16, label="CV ± 1 SD across folds")
ax.axvline(best_depth, color=SLATE, ls="--", lw=1.4)
ax.text(best_depth + 0.2, 89.2, f"Sweet spot\n(depth = {best_depth})", fontsize=9, color=SLATE)
ax.text(1.15, 90.5, "Underfitting\n(too simple)", fontsize=8.5, color=SLATE)
ax.text(8.4, 96.5, "Overfitting gap\n(memorizing noise)", fontsize=8.5, color=BLUE)
ax.set_xlabel("Decision tree max depth (complexity knob)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(88, 101.2)
ax.set_title("Memorizing vs. generalizing across tree complexity")
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

Look at the three zones of [](#ch22-overfit-fig):
- **Left (`depth = 1`, Underfitting):** A single question isn't enough to capture tumor biology — both training and validation scores are modest.
- **Middle (`depth = 4–5`, Sweet Spot):** Cross-validation accuracy peaks cleanly (`93.4%` at depth 5, confirmed at `93.9%` on our locked test vault!) while the model stays compact and readable.
- **Right (`depth = 7–12`, Overfitting):** Training accuracy marches straight to **100.0%**, yet CV accuracy sags downward — every extra branch memorized training quirks instead of generalizable signal!

## Lab 2: Catching patient-level data leakage red-handed

Now let's reproduce the opening hospital scandal! Imagine a dataset where **no biological signal exists at all** — labels (`disease` vs. `healthy`) are assigned purely by coin flip, so the *true* maximum accuracy of any honest model on new patients must be **50%**.

Each of 100 patients has **6 repeated lab visits** (600 total rows), and each patient has their own stable biological baseline fingerprint (like resting heart rate, height, or background image texture) that barely changes from visit to visit. Watch what happens when we make the classic mistake of splitting **rows** randomly instead of grouping by **patient ID**!

```{admonition} Predict before you run
:class: tip
Remember: in the synthetic dataset below, the disease label has **ZERO real relationship** with the measurements (pure 50/50 coin flip per patient!). If we mistakenly split the 600 visit rows randomly (putting visits from the same patient in both train and test), what test accuracy will a 1-nearest-neighbor or random forest model report — ~50%, ~70%, or **over 90%**?!
```

```{code-cell} python
:label: ch22-leakage-sim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(123)
n_patients, visits_per_patient = 100, 6

patient_ids = np.repeat(np.arange(n_patients), visits_per_patient)
# Assign PURE RANDOM coin-flip disease labels to each patient (no true signal!)
patient_labels = rng.integers(0, 2, size=n_patients)
y_leak = np.repeat(patient_labels, visits_per_patient)

# Each patient has a unique personal baseline fingerprint + tiny visit-to-visit noise
baselines = rng.normal(0, 2.0, size=(n_patients, 10))
X_leak = np.repeat(baselines, visits_per_patient, axis=0) + rng.normal(0, 0.35, size=(n_patients * visits_per_patient, 10))

# WRONG split: random row split leaks the same patients into train and test!
X_tr_bad, X_te_bad, y_tr_bad, y_te_bad = train_test_split(
    X_leak, y_leak, test_size=0.25, random_state=0
)
acc_leaky = RandomForestClassifier(random_state=0).fit(X_tr_bad, y_tr_bad).score(X_te_bad, y_te_bad)

# HONEST split: GroupShuffleSplit keeps all visits of a patient together!
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
tr_idx, te_idx = next(gss.split(X_leak, y_leak, groups=patient_ids))
acc_honest = RandomForestClassifier(random_state=0).fit(
    X_leak[tr_idx], y_leak[tr_idx]
).score(X_leak[te_idx], y_leak[te_idx])

print(f"True possible accuracy on new patients : 50.0% (pure coin flip!)")
print(f"Row-level split (LEAKY!)               : {acc_leaky:.1%}  <-- COMPLETELY FAKE!")
print(f"Patient-level split (HONEST)           : {acc_honest:.1%}  <-- tells the truth!")
```

**Over 97% accuracy on pure random noise!!** Pause and let that sink in. That single cell has ruined hundreds of published medical-AI papers. Without a patient-level group split (`GroupShuffleSplit` or `GroupKFold`), a model can score ~97% by simply recognizing *who the patient is* — and then collapse to random guessing (`~52%`) on the very first new patient in the clinic.

```{admonition} Definition — data leakage and external validation
:class: important
{term}`Data leakage` happens whenever information from the evaluation set (or from future clinical events, like a doctor's billing code entered *after* diagnosis!) accidentally enters model training or preprocessing. {term}`External validation` tests the frozen model on completely independent patients from a different hospital, scanner, or year — the gold-standard check recommended by the **TRIPOD+AI** reporting guideline [@collins2024tripod].
```

## Five sneaky forms of clinical data leakage to watch for

::::{tab-set}

:::{tab-item} 1. Patient / Repeat-Scan Split
Putting multiple X-rays, pathology patches, or hospital visits from the *same patient* into both training and test sets (exactly what we simulated above!). **Fix:** Always group splits by `patient_id`.
:::

:::{tab-item} 2. Preprocessing Before Splitting
Computing standardization means, imputing missing lab values, or selecting "top 50 genes" across the *entire* dataset before splitting. **Fix:** Wrap preprocessing inside a `scikit-learn` `Pipeline` so every transform fits on training folds only.
:::

:::{tab-item} 3. Future (Target) Leakage
Including a feature that is only known *after* the outcome — for example, training a sepsis early-warning model using *"prescribed ICU antibiotics"* as an input! The doctor already knew the patient was septic. **Fix:** Freeze features strictly at the *time of prediction*.
:::

:::{tab-item} 4. Temporal Drift Ignored
Randomly mixing 2020–2025 patient records instead of testing whether a model trained on 2020–2023 still works on 2024–2025. **Fix:** Use **temporal validation** (train on past years, test on future years).
:::

::::

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 2.2.1 (Modify) — Watch the leakage fade as visits get noisier
:class: hint
In [](#ch22-leakage-sim), visit-to-visit noise is `0.35`. Raise it to `1.5` and then `4.0` and re-run both the leaky row split and the honest patient split. Why does the leaky accuracy fall as visit noise rises, while the honest accuracy hovers near 50% regardless?
```

```{admonition} Solution
:class: dropdown
```python
for noise in (0.35, 1.5, 4.0):
    X_l = np.repeat(baselines, visits_per_patient, axis=0) + rng.normal(0, noise, size=(n_patients * visits_per_patient, 10))
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_l, y_leak, test_size=0.25, random_state=0)
    leaky = RandomForestClassifier(random_state=0).fit(X_tr_b, y_tr_b).score(X_te_b, y_te_b)
    honest = RandomForestClassifier(random_state=0).fit(X_l[tr_idx], y_leak[tr_idx]).score(X_l[te_idx], y_leak[te_idx])
    print(f"noise = {noise:4.2f} | leaky = {leaky:.1%} | honest = {honest:.1%}")
```

When visit noise is small (`0.35`), every patient's 6 visits form a tight personal cluster, making patient re-identification effortless (~100% fake accuracy). As noise swamps the personal fingerprint (`4.0`), even the leaky model can no longer recognize which patient a visit came from, so leaky accuracy drops toward 50%. Honest accuracy stays around 50% the whole time — reflecting the true zero-signal reality.
```

```{admonition} Exercise 2.2.2 (Make) — Build a leak-free scikit-learn Pipeline
:class: hint
Use `from sklearn.pipeline import make_pipeline`, `StandardScaler`, and `KNeighborsClassifier` to run **5-fold cross-validation** on the Wisconsin Breast Cancer dataset (`X_dev`, `y_dev`) for `n_neighbors` in `(3, 7, 15)`. Why is wrapping `StandardScaler()` inside `make_pipeline` safer than scaling `X_dev` once before calling `cross_val_score`?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

for k in (3, 7, 15):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores = cross_val_score(pipe, X_dev, y_dev, cv=cv)
    print(f"k = {k:>2}: leak-free 5-fold CV = {scores.mean():.1%} ± {scores.std():.1%}")
```

Putting `StandardScaler()` inside the `Pipeline` guarantees that inside *each* of the 5 cross-validation folds, the mean and standard deviation are computed strictly on the 4 training folds and then applied untouched to the held-out fold — completely eliminating preprocessing leakage!
```

```{admonition} Exercise 2.2.3 (Critical Reasoning) — Spot the leakage in a real-world pitch
:class: hint
A hospital startup pitches an AI model that predicts **which emergency-room patients will be admitted to the ICU** with 96% accuracy. Looking at their feature table, you notice two columns: (a) `arrival_oxygen_saturation` and (b) `icu_bed_request_timestamp`. Which column is valid and which is severe target leakage? Why?
```

```{admonition} Solution
:class: dropdown
- `arrival_oxygen_saturation` is **valid**: it is measured right at ER triage, *before* the admission decision is made.
- `icu_bed_request_timestamp` is **severe future/target leakage**: a nurse only enters an ICU bed request *after* a physician has already decided the patient needs the ICU! Training on it gives a useless model that merely reads the doctor's decision back to them.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Overfitting** shows up as training accuracy climbing toward 100% while validation accuracy peaks and falls.
- Never tune model complexity on your final test set — use **k-fold cross-validation** (inside a leak-free `Pipeline`) and keep a locked test vault for the end.
- **Data leakage** (especially splitting repeated visits/scans from the same patient across train and test) can produce ~100% accuracy on pure noise!
- Honest clinical evidence requires **patient-level splits**, **temporal checks**, and **external validation** across hospitals [@collins2024tripod].
```

*Next up:* Even with a 100% leak-free, externally validated classifier, **plain accuracy is still the wrong metric for medicine**. Let's learn how clinicians actually grade risk scores using thresholds, ROC/PR curves, and calibration in [](2_3-clinical-metrics-and-thresholds.md).
