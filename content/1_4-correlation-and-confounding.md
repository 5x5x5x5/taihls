---
title: 'Chapter 1.4 — Correlation Is Not Causation: Confounders and Clinical Trials'
short_title: '1.4 · Correlation & causation'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.4 — Correlation Is Not Causation: Confounders and Clinical Trials

*Week 3 · ~40 minutes · Prerequisites: [](1_3-looking-at-clinical-data.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- read a **correlation** number (between $-1$ and $+1$) on a scatter plot;
- explain the difference between **association** and **causation**;
- spot a hidden **confounder** and reverse a misleading trend (**Simpson's paradox**) by **stratifying**;
- explain why **randomized controlled trials (RCTs)** are medicine's gold standard for testing treatments.
```

## Start with a case: the helpful medicine that looked harmful

A hospital checks last year's records for a pain-and-inflammation medicine and finds an alarming chart: **patients who received higher doses ended up sicker a week later!**

Should doctors ban the medicine immediately?

Wait — ask one detective question: **who got the higher doses in the first place?** Doctors don't hand out medicine at random! They give *higher doses* to patients who arrived *much sicker*. And patients who arrive severely ill naturally have tougher recoveries than mild patients, even when the medicine truly helps them.

## Drawing the hidden cause

Let's draw the arrows of cause and effect:

```{mermaid}
flowchart TD
    Z["How Sick Patient Arrived<br/>(Hidden Confounder)"]
    X["Medicine Dose"]
    Y["Sickness 1 Week Later<br/>(Higher = Worse)"]
    Z -- "Sicker patients get higher doses" --> X
    Z -- "Sicker patients end up worse" --> Y
    X -- "Medicine actually HELPS (-)" --> Y
```

```{admonition} Definition — association, causation, and confounder
:class: important
- {term}`Association` means two measurements tend to move together (measured by {term}`Correlation`, a number from $-1$ to $+1$).
- {term}`Causation` means actively changing one thing *makes* the other change.
- A {term}`Confounder` is a hidden third factor that drives *both* things you are comparing — capable of creating a fake link or flipping a real one completely backward!
```

## Check a 6-patient example by hand!

Look at six patients split into **Mild** and **Severe** arrivals:
- **Mild patients:** Dose `5 mg → sickness 15`, Dose `10 mg → sickness 10` *(higher dose helped: dropped from 15 to 10!)*
- **Severe patients:** Dose `25 mg → sickness 65`, Dose `30 mg → sickness 60` *(higher dose helped: dropped from 65 to 60!)*

Yet if you ignore Mild vs. Severe and lump all four patients onto one chart:
- Low doses (`5–10 mg`) had sickness `10–15`, while high doses (`25–30 mg`) had sickness `60–65`!
**Pooled together, the medicine looks harmful — even though it helped inside EVERY group!** That famous flip is called {term}`Simpson's paradox`.

## Seeing Simpson's paradox in Python

Let's simulate 300 hospital patients where the hidden truth is that **more medicine reduces sickness** (`-1.2 * dose`), while starting severity pushes *both* dose and sickness up. Then we fix the illusion by **stratifying** ({term}`Stratification`) — comparing mild patients only to mild patients, and severe only to severe!

```{admonition} Predict before you run
:class: tip
In the code below, the medicine genuinely helps (`-1.2 * dose`). Will the overall **raw correlation** between `dose` and `sickness` be **positive** (looks harmful) or **negative** (looks helpful)? What will happen inside each severity group?
```

```{code-cell} python
:label: ch14-simpson
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n = 300

# Hidden confounder: 1 = mild, 2 = moderate, 3 = severe
severity = rng.choice([1, 2, 3], size=n)

# Sicker patients are given higher doses
dose = 10 * severity + rng.normal(0, 3.8, size=n)

# True outcome (higher = worse): severity hurts (+28), medicine HELPS (-1.2)
sickness = 28 * severity - 1.2 * dose + rng.normal(0, 4.5, size=n)

r_all = np.corrcoef(dose, sickness)[0, 1]
print(f"All patients lumped together : correlation = {r_all:+.2f}  (looks HARMFUL!)")

for level, label in [(1, "Mild"), (2, "Moderate"), (3, "Severe")]:
    mask = severity == level
    r_group = np.corrcoef(dose[mask], sickness[mask])[0, 1]
    print(f"  {label:<8} patients only     : correlation = {r_group:+.2f}  (clearly HELPS!)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)

ax1.scatter(dose, sickness, s=18, color="#4D4D4D", alpha=0.6)
ax1.set_title(f"A. Pooled together (r = {r_all:+.2f})")
ax1.set_xlabel("Medicine dose (mg)")
ax1.set_ylabel("Sickness after 1 week (higher = worse)")
ax1.grid(alpha=0.25)

colors = {1: "#009E73", 2: "#E69F00", 3: "#0072B2"}
names  = {1: "Mild", 2: "Moderate", 3: "Severe"}
for level in [1, 2, 3]:
    m = severity == level
    ax2.scatter(dose[m], sickness[m], s=18, color=colors[level], alpha=0.7, label=names[level])
ax2.set_title("B. Stratified by severity (trend slopes DOWN in every group!)")
ax2.set_xlabel("Medicine dose (mg)")
ax2.legend(frameon=False)
ax2.grid(alpha=0.25)

fig.tight_layout()
plt.show()
```

Look at Panel A vs. Panel B: the exact same dots tell two completely opposite stories once you color them by severity!

## How medicine beats confounders: Randomized Controlled Trials

Stratifying works *if* you measured the confounder. What if there's an unmeasured confounder (like hidden genetics or diet)? Medicine solves that with a {term}`Randomized controlled trial (RCT)`: assign doses **by chance** (like rolling dice), so mild and severe patients get low and high doses equally — breaking the arrow from `Severity` to `Dose`!

```{code-cell} python
:label: ch14-rct
# Same patients, same biology — ONLY change: doses assigned at random!
random_dose = rng.uniform(5, 35, size=n)
rct_sickness = 28 * severity - 1.2 * random_dose + rng.normal(0, 4.5, size=n)

print(f"Randomized trial raw correlation: {np.corrcoef(random_dose, rct_sickness)[0, 1]:+.2f}  (correctly negative!)")
```

With randomization, the raw correlation immediately turns **negative** (helpful) without even needing to know each patient's severity!

```{admonition} Going deeper — collider bias (when filtering creates a fake link!)
:class: dropdown
Stratifying on a *confounder* fixes bias, but filtering on an *effect* of two unrelated things (a **collider**) can **invent** a fake link! For example, suppose heart trouble and broken legs are totally unrelated in the general public — yet if you study *only patients admitted to the hospital*, knowing someone does *not* have a broken leg makes it much more likely they are there for heart trouble! That creates a fake negative correlation inside hospital-only data (**Berkson's paradox**).
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.4.1 (Modify) — What if severity mattered less?
:class: hint
In [](#ch14-simpson), change `28 * severity` to `8 * severity` and re-compute `r_all`. Does the raw correlation still flip positive, or does the medicine's true helpful effect show through?
```

```{admonition} Solution
:class: dropdown
```python
sickness_mild_conf = 8 * severity - 1.2 * dose + rng.normal(0, 4.5, size=n)
print(f"With weaker confounder (8x): raw r = {np.corrcoef(dose, sickness_mild_conf)[0, 1]:+.2f}")
```
With a much weaker confounder (`8` instead of `28`), the pooled correlation drops near zero or turns slightly negative — notice that even when a confounder doesn't *fully flip* the sign, it still quietly hides a real treatment benefit!
```

```{admonition} Exercise 1.4.2 (Make) — Check a real correlation in the diabetes table
:class: hint
Using `load_diabetes(as_frame=True, scaled=False)` from Chapter 1.3, compute the correlation between `"bmi"` and `"target"` (progression) using `df["bmi"].corr(df["target"])`. Is the association positive, negative, or near zero?
```

```{admonition} Solution
:class: dropdown
```python
import pandas as pd
from sklearn.datasets import load_diabetes
ddf = load_diabetes(as_frame=True, scaled=False).frame
print(f"BMI vs. diabetes progression: r = {ddf['bmi'].corr(ddf['target']):+.2f}")
```
`r ≈ +0.59` — a fairly strong positive association (higher BMI tends to go along with faster progression), though this observational dataset alone cannot prove how much is caused by BMI vs. genetics, diet, or exercise.
```

```{admonition} Exercise 1.4.3 (Critical Reasoning) — Spot the confounder in a headline
:class: hint
A news article says: *"Large health study finds people who take daily aspirin have twice as many heart attacks as people who never take aspirin!"* Explain in two sentences what hidden confounder is fooling that headline.
```

```{admonition} Solution
:class: dropdown
**Pre-existing heart risk** is the confounder: doctors prescribe daily aspirin specifically to older or high-risk patients who are already far more likely to have a heart attack! Healthy teenagers rarely take daily aspirin.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Correlation** ($-1$ to $+1$) measures whether two variables move together — **association is not causation**.
- A **confounder** drives both sides of a comparison and can even reverse the true trend (**Simpson's paradox**).
- **Stratifying** compares like with like; **randomized controlled trials (RCTs)** break confounding altogether.
- Whenever you read *"X is linked to Y,"* ask: **what hidden third thing could cause both?**
```

*Next up:* Even when a medical test has zero confounding, percentages can still play huge tricks on our brains — especially for rare diseases. Let's master base rates and sample-size noise in [](1_5-base-rates-and-uncertainty.md)!
