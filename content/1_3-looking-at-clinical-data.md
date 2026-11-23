---
title: 'Chapter 1.3 — Look Before You Leap: Exploring Real Health Data'
short_title: '1.3 · Exploring health data'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.3 — Look Before You Leap: Exploring Real Health Data

*Week 2 · ~35 minutes · Prerequisites: [](0_1-python-and-data-warmup.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- load a real medical dataset into a `pandas` table and read its rows and columns (**features**);
- draw a **histogram** to see the shape (**distribution**) of a measurement;
- explain why **mean** and **median** disagree on **skewed** health data — and why you shouldn't blindly delete extreme patients;
- spot a misleading medical chart caused by a **truncated vertical axis**.
```

## Start with a case: two honest numbers that tell different stories

Suppose you're scheduled for knee surgery and ask the clinic: *"How long do patients usually stay in the hospital?"*
- Nurse A computes the **mean** (average) and says **"5.4 days."**
- Nurse B computes the **median** (middle patient) and says **"3.2 days."**

Both looked at the exact same hospital spreadsheet! Why are they more than two days apart? Because most surgery patients go home smoothly in 2–4 days, while a few patients with serious complications stay 30 or 40 days. Those few very long stays drag the **mean** upward, even though the typical patient leaves in about three days.

Before anyone builds an AI model, good scientists slow down and **look at the data**.

## Loading a real medical table

We'll load a classic real clinical dataset built right into `scikit-learn`: **442 patients with diabetes**, each measured on age, body mass index (**BMI**, a ratio of weight to height), blood pressure, and blood lab tests — plus a score tracking how much their diabetes progressed one year later.

```{admonition} Definition — dataset, row, and feature
:class: important
A **dataset** is a table of data. Each **row** is one patient, and each **column** is one {term}`Feature` — a single measurement recorded for every patient.
```

```{code-cell} python
:label: ch13-load-diabetes
import pandas as pd
from sklearn.datasets import load_diabetes

raw = load_diabetes(as_frame=True, scaled=False)   # keep real clinical units!
df = raw.frame.rename(columns={"target": "progression"})

print(f"Table size: {df.shape[0]} patients, {df.shape[1]} columns")
df[["age", "bmi", "bp", "progression"]].head(6)
```

Pause and check those first six rows like a detective: ages are in the 40s–60s, BMI is around 20–35, blood pressure (`bp`) sits around 70–110, and `progression` is a positive score. Checking whether numbers make real-world sense catches broken spreadsheets immediately!

## Seeing the shape: histograms, mean, and median

To see the overall {term}`Distribution` of a column — which values are common and which are rare — we draw a **histogram** (chop the range into bins and count how many patients fall into each bin).

First let's check **mean vs. median** by hand on five hospital stays `[2, 3, 3, 4, 38]` days:
- **Mean:** $(2 + 3 + 3 + 4 + 38) / 5 = 50 / 5 = \mathbf{10}$ days! (Pulled sky-high by that one 38-day stay!)
- **Median:** Line them up in order and pick the exact middle number → **`3`** days!

```{admonition} Predict before you run
:class: tip
We will plot histograms of **BMI** (fairly balanced) and **1-year diabetes progression** (a long right tail of higher scores). In the right-skewed progression chart, will the **mean** sit to the **right** (higher) or to the **left** (lower) of the **median**?
```

```{code-cell} python
:label: ch13-hist-fig
import matplotlib.pyplot as plt

for col in ["bmi", "progression"]:
    print(f"{col:<12} | mean = {df[col].mean():5.1f} | median = {df[col].median():5.1f} | "
          f"spread (std) = {df[col].std():4.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))

for ax, col, title, color in [
    (ax1, "bmi", "A. BMI (fairly balanced)", "#0072B2"),
    (ax2, "progression", "B. Diabetes progression (right-skewed)", "#009E73"),
]:
    ax.hist(df[col], bins=20, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(df[col].mean(), color="#D55E00", lw=2.2, label=f"Mean ({df[col].mean():.1f})")
    ax.axvline(df[col].median(), color="#111111", lw=2.2, ls="--", label=f"Median ({df[col].median():.1f})")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Number of patients")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

fig.tight_layout()
plt.show()
```

In Panel B, the long tail of high progression scores pulls the **mean (`152.1`)** noticeably above the **median (`140.5`)** — the classic signature of a **right-skewed** distribution. Notice we also printed `std` (**standard deviation**), which measures roughly how spread out patients are around the mean.

:::{margin}
**Don't blindly delete outliers!** A body temperature of `980°F` is a typo (fix it). A 40-day ICU stay is a **real critically ill human being** — deleting extreme patients makes an AI blind to the sickest cases!
:::

## Spotting a misleading chart: the chopped axis

Numbers aren't the only way people get misled — charts can trick your eyes too! Look at the **exact same two numbers** (`81%` vs. `84%` recovery) drawn two ways:

```{code-cell} python
:label: ch13-truncated-fig
clinics = ["Standard Care", "New Treatment"]
recovery = [81, 84]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

ax1.bar(clinics, recovery, color=["#0072B2", "#009E73"], width=0.5)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Patients recovered (%)")
ax1.set_title("Honest bar chart (starts at 0%)")

ax2.bar(clinics, recovery, color=["#0072B2", "#D55E00"], width=0.5)
ax2.set_ylim(80, 85)
ax2.set_ylabel("Patients recovered (%)")
ax2.set_title("Misleading bar chart (chopped at 80%!)")

fig.tight_layout()
plt.show()
```

On the right, chopping the vertical axis at `80%` makes a modest 3-percentage-point difference look **four times taller**! Whenever a bar chart looks dramatic in a news story or ad, check **where the vertical axis starts**.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.3.1 (Modify) — Inspect blood pressure (`bp`)
:class: hint
Print the mean, median, min, and max of `df["bp"]` (blood pressure) and draw its histogram. Is blood pressure heavily skewed or fairly balanced?
```

```{admonition} Solution
:class: dropdown
```python
print(f"bp: mean={df['bp'].mean():.1f}, median={df['bp'].median():.1f}, "
      f"min={df['bp'].min():.1f}, max={df['bp'].max():.1f}")
plt.figure(figsize=(6, 3.5))
plt.hist(df["bp"], bins=20, color="#0072B2", edgecolor="white")
plt.xlabel("Blood pressure"); plt.ylabel("Patients")
plt.show()
```
The mean (`94.6`) and median (`93.0`) sit very close together — blood pressure in this group is fairly balanced!
```

```{admonition} Exercise 1.3.2 (Make) — One bad typo breaks the mean
:class: hint
Start with the five hospital stays `stays = [2, 3, 3, 4, 5]`. Print their mean and median using `pd.Series(stays)`. Now replace `5` with a data-entry typo `50` and re-check both numbers. Which one barely flinched?
```

```{admonition} Solution
:class: dropdown
```python
s1 = pd.Series([2, 3, 3, 4, 5])
s2 = pd.Series([2, 3, 3, 4, 50])
print(f"Clean : mean = {s1.mean():.1f}, median = {s1.median():.1f}")
print(f"Typo  : mean = {s2.mean():.1f}, median = {s2.median():.1f}")
```
One typo rockets the mean from `3.4` to `12.4`, while the median stays completely steady at `3.0`!
```

```{admonition} Exercise 1.3.3 (Critical Reasoning) — Which average do you give the hospital director?
:class: hint
Return to our opening knee-surgery clinic (`median = 3.2 days`, `mean = 5.4 days`).
1. Which number gives a **patient** a better sense of their own likely stay?
2. Which number does a **hospital manager** need to budget total bed-days for the next 1,000 surgeries?
```

```{admonition} Solution
:class: dropdown
1. **The patient** wants the **median (`3.2` days)** — half of patients leave in about 3 days or less.
2. **The manager** needs the **mean (`5.4` days)** — because total bed-days for 1,000 patients is $1{,}000 \times 5.4 = 5{,}400$ bed-days! Using the median (`3,200`) would leave the hospital badly short on beds because the rare long stays take up huge bed capacity.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Always inspect a dataset's rows, ranges, and **histograms** before modeling.
- On **skewed** data, extreme values pull the **mean** toward the long tail while the **median** holds the middle.
- Never delete extreme values blindly — separate data-entry typos from genuinely sick patients.
- Always check whether a bar chart's vertical axis starts at **zero**.
```

*Next up:* Now that we can read one column honestly, what happens when we compare *two* columns — like a medicine's dose and whether patients got better? Meet the biggest trap in health science in [](1_4-correlation-and-confounding.md)!
