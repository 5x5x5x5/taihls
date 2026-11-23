---
title: 'Chapter 1.2 — Looking Before You Leap: Exploring Real Clinical Data'
short_title: '1.2 · Exploring clinical data'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.2 — Looking Before You Leap: Exploring Real Clinical Data

*Estimated time: ~35 minutes · Prerequisites: [](1_1-models-and-outbreaks.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- load a real clinical dataset into a `pandas` table and audit its shape, features, and summary statistics;
- distinguish **mean vs. median** and **standard deviation vs. interquartile range (IQR)** on **skewed** medical data;
- recognize how a handful of extreme **outliers** can distort averages — and why you should never delete them blindly;
- spot misleading medical charts caused by a **truncated axis** or hidden sample sizes.
```

## Start with a case: the "average" hospital stay

Suppose an insurance administrator asks: *"How long does a typical patient stay in our hospital after surgery?"* One analyst computes the **mean** and reports **6.8 days**. Another computes the **median** and reports **3.0 days**. Both ran their code on the exact same spreadsheet. Who is telling the truth?

Both — and neither tells the whole story on its own. Most surgical patients recover smoothly and go home in 2 to 4 days, while a small minority suffer complications and spend 30 to 60 days in intensive care. Those rare, prolonged stays drag the mean upward, far away from what a typical patient experiences.

Before any scientist trains an AI model, they slow down and **look at the data**. Rushing straight to modeling without checking shapes, skew, outliers, and units is the #1 way beginners get fooled.

## Loading a real clinical cohort

We'll start with a classic real medical dataset built right into `scikit-learn`: **442 patients with diabetes**, each measured on ten baseline clinical variables (age, sex, body mass index, blood pressure, and six blood-serum measurements) along with a quantitative score of **disease progression one year later**.

```{admonition} Definition — dataset, row, and feature
:class: important
A **dataset** is a structured table where each **row** represents one observation (here, one patient) and each **column** represents one {term}`feature` — a single measurement recorded for every observation.
```

```{code-cell} python
:label: ch12-load-diabetes
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

raw = load_diabetes(as_frame=True, scaled=False)   # keep real clinical units!
df = raw.frame.rename(columns={"target": "progression"})

print(f"Cohort size: {df.shape[0]} patients, {df.shape[1]} columns")
df[["age", "bmi", "bp", "s5", "progression"]].head(6)
```

:::{margin}
Setting `scaled=False` in `load_diabetes` gives us **actual clinical units** (age in years, BMI in kg/m², blood pressure in mmHg) instead of pre-standardized numbers — far easier to sanity-check!
:::

Inspecting those first six rows immediately pays off: `age` sits in the 40s–60s, `bmi` sits in the 20s–30s, `bp` sits around 70–110 mmHg, and `progression` ranges from double to triple digits. Checking physical plausibility ("Are any ages negative? Is anyone's BMI 400?") is your very first line of defense against corrupted clinical records.

## Summarizing center and spread

A single table of 442 numbers is too big for human eyes to absorb. We summarize a numerical column's {term}`distribution` with two complementary questions:

1. **Where is the center?** (What is a "typical" patient?)
2. **How wide is the spread?** (Are patients tightly clustered or wildly varied?)

::::{tab-set}

:::{tab-item} Mean & Standard Deviation (Sensitive to tails)
- **Mean:** Add every value and divide by $n$:
  $$\bar{x} = \frac{x_1 + x_2 + \dots + x_n}{n}$$ (eq-12-mean)
- **Standard deviation (SD):** Roughly the typical distance of a patient from the mean. Great for symmetric bell-shaped data, but easily inflated by extreme outliers.
:::

:::{tab-item} Median & IQR (Resistant to outliers)
- **Median:** Sort all patients from lowest to highest and pick the exact middle value (50th percentile).
- **Interquartile range (IQR):** The span of the middle 50% of patients — from the 25th percentile ($Q_1$) to the 75th percentile ($Q_3$): $\text{IQR} = Q_3 - Q_1$. Even if the single sickest patient stays 1,000 days, the median and IQR barely budge!
:::

::::

Let's compare a **nearly symmetric** clinical feature (`bmi`) against a **right-skewed** clinical outcome (`progression`), alongside a realistic **hospital length-of-stay** sample with ICU outliers!

```{admonition} Predict before you run
:class: tip
For a heavily **right-skewed** distribution (most patients have low values, a few have very high values), will the **mean** sit *above* or *below* the **median**? What happens if we accidentally type one extra zero on a single patient's hospital stay (turning `45` days into `450` days) — which summary shifts dramatically and which stays unchanged?
```

```{code-cell} python
:label: ch12-skew-stats
rng = np.random.default_rng(42)
# Realistic post-surgery length of stay (days): most leave in 2-5 days, a few need ICU
los = np.concatenate([
    rng.gamma(shape=3.0, scale=1.1, size=280),   # routine recoveries
    rng.uniform(18, 55, size=20),                # prolonged ICU complications
])

def summarize(name, values):
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    print(f"{name:<22} | mean = {np.mean(values):5.1f} (SD {np.std(values):4.1f}) "
          f"| median = {med:5.1f} (IQR {q3 - q1:4.1f})")

summarize("Diabetes BMI (kg/m²)", df["bmi"])
summarize("Diabetes progression", df["progression"])
summarize("Hospital stay (days)", los)

# Now introduce ONE data-entry typo (10x mistake on a single patient)
los_typo = los.copy()
los_typo[0] = 450.0
summarize("Stay + 1 typo (450d)", los_typo)
```

Look at those four rows — the story jumps off the screen:
- On **BMI**, mean ($26.4$) and median ($25.7$) are very close: the distribution is fairly balanced.
- On **diabetes progression**, a right tail pulls the mean ($152.1$) 11.6 points above the median ($140.5$).
- On **hospital length of stay**, the 20 ICU patients pull the mean ($5.4$ days) nearly **70% higher** than the median ($3.2$ days) — and just **one single typo** (`450` days) rockets the mean up by $1.5$ days and triples the standard deviation, while the median and IQR don't move by a single tenth!

Let's inspect all three distributions visually.

```{code-cell} python
:label: ch12-dist-fig
import matplotlib.pyplot as plt

BLUE, VERMILION, TEAL, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
panels = [
    (df["bmi"], "A. BMI (nearly symmetric)", "Body mass index (kg/m²)", BLUE),
    (df["progression"], "B. Diabetes progression (mild skew)", "1-year progression score", TEAL),
    (los, "C. Hospital stay (strong right skew)", "Length of stay (days)", AMBER),
]

for ax, (vals, title, xlabel, color) in zip(axes, panels):
    ax.hist(vals, bins=22, color=color, alpha=0.8, edgecolor="white")
    m, med = np.mean(vals), np.median(vals)
    ax.axvline(m, color=VERMILION, lw=2.2, ls="-", label=f"Mean ({m:.1f})")
    ax.axvline(med, color="#111111", lw=2.2, ls="--", label=f"Median ({med:.1f})")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Patients")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

## Never blindly delete outliers!

A tempting beginner mistake is to say: *"Those 20 long hospital stays are outliers — let's delete them to make the chart clean!"* **Stop and ask first: is an extreme value an *error* or a *real patient*?**

- A recorded body temperature of `398°C` or a `450`-day stay on a 30-day audit is a **data-entry error** (fix or flag it).
- A 42-day ICU stay or a blood glucose of `480 mg/dL` is a **critically ill human being**. Deleting real extreme patients blinds an AI model to the exact high-risk cases medicine cares about most!

## Visual integrity: how charts lie without changing a single number

Even honest summaries can mislead when plotted carelessly. The classic clinical-marketing trick is the **truncated y-axis**: cutting off the bottom of a bar chart so a tiny 3-percentage-point difference looks like a miracle cure.

```{code-cell} python
:label: ch12-truncated-axis
treatments = ["Standard care", "New Drug X"]
recovery = [81, 84]    # 3-percentage-point difference

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

# Left: honest baseline at zero
bars1 = ax1.bar(treatments, recovery, color=[BLUE, TEAL], width=0.5)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Patients recovered (%)")
ax1.set_title("Honest view: y-axis starts at 0%")
for b, val in zip(bars1, recovery):
    ax1.text(b.get_x() + b.get_width() / 2, val + 2, f"{val}%", ha="center", fontweight="bold")
ax1.grid(axis="y", alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)

# Right: truncated axis exaggerates a 3% gap into a 4x taller bar
bars2 = ax2.bar(treatments, recovery, color=[BLUE, VERMILION], width=0.5)
ax2.set_ylim(80, 85)
ax2.set_ylabel("Patients recovered (%)")
ax2.set_title("Misleading view: y-axis chopped at 80%")
for b, val in zip(bars2, recovery):
    ax2.text(b.get_x() + b.get_width() / 2, val + 0.15, f"{val}%", ha="center", fontweight="bold")
ax2.grid(axis="y", alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

In [](#ch12-truncated-axis), both panels show the exact same numbers (`81%` vs. `84%`), yet the right-hand bar is **four times taller** than the left-hand bar! Bar length encodes quantity — so bar charts must *always* start at zero. (Line charts tracking small changes in body temperature, like $36.8^\circ\text{C}$ to $38.5^\circ\text{C}$, *can* zoom in as long as the y-axis is clearly labeled and the reader isn't tricked by bar area.)

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.2.1 (Modify) — Inspect blood pressure and cholesterol
:class: hint
Using `df` from [](#ch12-load-diabetes), call `summarize()` on `"bp"` (blood pressure) and `"s1"` (total serum cholesterol) and plot their histograms side by side. Which one is more skewed, and are there any unusually extreme patients you would double-check?
```

```{admonition} Solution
:class: dropdown
```python
summarize("Blood pressure (mmHg)", df["bp"])
summarize("Total cholesterol (s1)", df["s1"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.hist(df["bp"], bins=20, color=BLUE, edgecolor="white")
ax1.set_title("Blood pressure (mmHg)")
ax2.hist(df["s1"], bins=20, color=TEAL, edgecolor="white")
ax2.set_title("Total cholesterol (s1)")
fig.tight_layout()
```

Both `bp` and `s1` are fairly symmetric (mean and median sit within 1–2 units of each other), though `s1` has a slight right tail reaching ~300 mg/dL — elevated, yet completely plausible clinically for patients with metabolic disease, so those values should be kept, not deleted.
```

```{admonition} Exercise 1.2.2 (Make) — Building an automated data-quality check
:class: hint
Write a 5-line check on `df` that prints (1) how many missing values (`NaN`) each column has, and (2) the minimum and maximum of `"age"`, `"bmi"`, and `"bp"`. Why should every clinical AI pipeline run a check like this *before* training?
```

```{admonition} Solution
:class: dropdown
```python
print("Missing values total:", df.isna().sum().sum())
for col in ["age", "bmi", "bp"]:
    print(f"{col:>4}: min = {df[col].min():.1f}, max = {df[col].max():.1f}")
```

In this clean teaching dataset there are `0` missing values, and every range is clinically plausible (`age` 19–79, `bmi` 18.0–42.2, `bp` 62–133 mmHg). Real hospital electronic health records routinely encode missing values as `-999` or `0` — catching those before training prevents models from learning nonsense rules like *"blood pressure = 0 predicts discharge!"*
```

```{admonition} Exercise 1.2.3 (Critical Reasoning) — Mean vs. median in hospital planning
:class: hint
Return to our surgical length-of-stay example (median 3.2 days, mean 5.4 days). Imagine two people ask for a single number:
1. A patient asking *"How many days will I probably spend in the hospital?"*
2. A hospital director budgeting *total bed-days* needed for the next 1,000 surgeries.

Which summary (mean or median) should you hand to each person, and why?
```

```{admonition} Solution
:class: dropdown
1. **Give the patient the median (plus IQR):** *"Half of patients go home within 3.3 days, and most stay between 2 and 5 days — though a small fraction with complications stay much longer."* That reflects the typical personal experience.
2. **Give the hospital director the mean:** Total bed-days for 1,000 patients equals $1{,}000 \times \text{mean} \approx 5{,}400$ bed-days! Budgeting using the median ($3{,}300$ bed-days) would leave the ICU severely understaffed because the rare long-stay patients consume a huge share of total beds.

Different decisions genuinely require different statistics — there is no single "best" average without knowing the question.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Always inspect clinical data visually and numerically (units, ranges, missingness, shape) *before* modeling.
- On **skewed** distributions, the **mean** and **SD** are pulled toward the long tail, whereas the **median** and **IQR** capture the typical patient robustly.
- Distinguish **data errors** from **real extreme patients** — never delete outliers blindly.
- Check chart axes (especially bar charts starting above zero) before believing a dramatic visual gap.
```

*Next up:* Now that we know how to look at one column honestly, what happens when we compare *two* columns — like a drug dose and a patient's outcome? Prepare for the single most famous trap in all of medical data science in [](1_3-correlation-confounding-causation.md).
