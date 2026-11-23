---
title: 'Chapter 1.3 — Association, Confounding, and Causation'
short_title: '1.3 · Association vs. causation'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.3 — Association, Confounding, and Causation

*Estimated time: ~40 minutes · Prerequisites: [](1_2-exploring-clinical-data.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- compute and interpret the **correlation coefficient** ($r$) between two clinical variables;
- draw a simple **causal diagram** to spot a **confounder** behind a misleading pattern;
- demonstrate **Simpson's paradox** in code and reverse an apparent treatment harm by **stratifying**;
- explain why **randomized controlled trials (RCTs)** break confounding — and recognize **collider bias** in hospital-only datasets.
```

## Start with a case: the "deadly" medicine that actually saves lives

A hospital analyzes its electronic health records for a new anti-inflammatory drug. The data science team plots each patient's **drug dose** against their **7-day sickness score** (higher = worse). The correlation is glaringly positive: **patients who got higher doses did dramatically worse**.

Should the hospital ban the drug immediately?

Not so fast. Ask the single most important question in all of health data science:

> **Why did some patients get higher doses in the first place?**

In real hospitals, doctors do not hand out drugs by rolling dice. They give *higher doses* to patients who arrived *much sicker*! And sicker patients naturally have worse outcomes than mild patients, even when the drug genuinely helps them. In epidemiology, this trap is called **confounding by indication** — and it fools naive AI models every single day.

## Drawing the hidden web: causal diagrams

Before writing a line of code, let's draw the causal arrows. An arrow $A \to B$ means *"changing $A$ directly causes $B$ to change."*

```{mermaid}
flowchart TD
    Z["Baseline Severity (Confounder)<br/>How sick the patient arrived"]
    X["Drug Dose (Treatment)"]
    Y["7-Day Outcome (Higher = Worse)"]
    Z -- "Sicker patients get higher doses (+)" --> X
    Z -- "Sicker patients have worse outcomes (+)" --> Y
    X -- "Drug truly helps (-)" --> Y
```

```{admonition} Definition — association, causation, and confounder
:class: important
- {term}`Association` means two variables tend to move together in data.
- {term}`Causation` means actively changing one variable makes the other change.
- A {term}`confounder` is a common cause that drives *both* the treatment and the outcome ($X \leftarrow Z \to Y$). It creates a backdoor path that can invent, mask, or completely **reverse** the apparent relationship between treatment and outcome.
```

Let's measure straight-line association with the {term}`correlation coefficient` $r$, which ranges from $-1$ (perfect downward line) through $0$ (no straight-line trend) to $+1$ (perfect upward line).

## Lab 1: Building Simpson's paradox from the inside

Because we simulate the hospital records ourselves, we know the ground truth: every extra unit of drug *reduces* sickness by `1.2` points (`-1.2 * dose`). Watch how baseline severity completely reverses that truth in the raw chart!

```{admonition} Predict before you run
:class: tip
In the code below, the true drug effect is **helpful** (`-1.2 * dose`, lowering the bad outcome). Yet severity pushes *both* `dose` up (`+10 * severity`) and `outcome` up (`+28 * severity`). Will the raw pooled correlation between `dose` and `outcome` be **positive** (looks harmful) or **negative** (looks helpful)? What happens when we split patients into mild, moderate, and severe groups?
```

```{code-cell} python
:label: ch13-simpson-sim
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n = 360

# Hidden confounder: baseline severity (1 = mild, 2 = moderate, 3 = severe)
severity = rng.choice([1, 2, 3], size=n)

# Observational hospital practice: sicker patients receive higher doses
dose = 10 * severity + rng.normal(0, 3.8, size=n)

# Ground truth: severity worsens outcome (+28), drug HELPS outcome (-1.2)
outcome = 28 * severity - 1.2 * dose + rng.normal(0, 4.5, size=n)

r_raw = np.corrcoef(dose, outcome)[0, 1]
print(f"Pooled (unstratified) correlation r = {r_raw:+.2f}  --> drug looks HARMFUL!")

for tier, name in [(1, "Mild"), (2, "Moderate"), (3, "Severe")]:
    mask = severity == tier
    r_sub = np.corrcoef(dose[mask], outcome[mask])[0, 1]
    print(f"  Inside {name:<8} tier only:       r = {r_sub:+.2f}  --> drug clearly HELPS!")
```

Look at that dramatic sign flip: pooled across the whole hospital, $r = +0.71$ (higher dose = worse outcome). Yet inside **every single severity tier**, $r$ flips to **$-0.64$ to $-0.74$** (higher dose = *better* outcome)!

```{admonition} Definition — stratification and Simpson's paradox
:class: important
To **stratify** ({term}`Stratification`) is to split the dataset into groups that share the same level of a confounder and examine the relationship *within* each group. When the pooled trend disappears or flips direction inside every subgroup, you are witnessing {term}`Simpson's paradox`.
```

Let's plot the pooled view side-by-side with the stratified view so this reversal burns into visual memory forever.

```{code-cell} python
:label: ch13-simpson-fig
BLUE, VERMILION, TEAL, AMBER, SLATE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#4D4D4D"

def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    return xs, m * xs + b

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

# Panel A: Naive pooled view
ax1.scatter(dose, outcome, s=20, color=SLATE, alpha=0.55)
xs, ys = fit_line(dose, outcome)
ax1.plot(xs, ys, color=VERMILION, lw=2.5, label=f"Overall trend (r = {r_raw:+.2f})")
ax1.set_title("A. Pooled observational data (misleading!)")
ax1.set_xlabel("Drug dose (mg)")
ax1.set_ylabel("7-day sickness score (higher = worse)")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)

# Panel B: Stratified by baseline severity
palette = {1: TEAL, 2: AMBER, 3: BLUE}
labels = {1: "Mild", 2: "Moderate", 3: "Severe"}
for tier in (1, 2, 3):
    m = severity == tier
    r_sub = np.corrcoef(dose[m], outcome[m])[0, 1]
    ax2.scatter(dose[m], outcome[m], s=20, color=palette[tier], alpha=0.6)
    xs, ys = fit_line(dose[m], outcome[m])
    ax2.plot(xs, ys, color=palette[tier], lw=2.5, label=f"{labels[tier]} (r = {r_sub:+.2f})")

ax2.set_title("B. Stratified by severity (Simpson's paradox revealed)")
ax2.set_xlabel("Drug dose (mg)")
ax2.legend(frameon=False, loc="upper left")
ax2.grid(alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

:::{margin}
**Why AI needs causal thinking:** A standard machine-learning model trained only on Panel A would happily predict *"giving a higher dose predicts death"* — and if someone used that model to recommend treatments, it could recommend withholding a life-saving drug!
:::

## Lab 2: Breaking the confounder with randomization

Stratification works *if* we measured every confounder (`severity`). But what about unmeasured confounders — hidden genetics, nutrition, stress, or frailty that aren't even columns in our table?

Medicine's gold-standard weapon against *both* measured and unmeasured confounders is the {term}`Randomized controlled trial (RCT)`: assign the dose **at random**, completely independent of how sick the patient arrived. In our causal diagram, randomization physically deletes the arrow `Severity → Dose`.

```{code-cell} python
:label: ch13-rct-sim
# Same patients, same biological truth — ONLY change: dose is assigned at random!
dose_rct = rng.uniform(5, 35, size=n)
outcome_rct = 28 * severity - 1.2 * dose_rct + rng.normal(0, 4.5, size=n)

r_rct = np.corrcoef(dose_rct, outcome_rct)[0, 1]
slope_rct, _ = np.polyfit(dose_rct, outcome_rct, 1)

print(f"Observational pooled slope: {np.polyfit(dose, outcome, 1)[0]:+.2f} pts/mg (WRONG SIGN!)")
print(f"Randomized trial slope:     {slope_rct:+.2f} pts/mg (True causal effect = -1.20)")
print(f"Randomized trial r:         {r_rct:+.2f}")
```

Without even knowing each patient's severity, the randomized trial immediately recovers the true negative slope (around `-1.2` points per mg)! Randomization balanced mild, moderate, and severe patients equally across low and high doses.

## Watch out for a sneaky cousin: collider bias

Stratifying on a *confounder* fixes bias. Surprisingly, stratifying on the **wrong** kind of variable — a {term}`collider`, which is *caused by* both variables ($X \to C \leftarrow Y$) — actually **creates** a fake correlation out of thin air!

```{mermaid}
flowchart LR
    X["Disease A<br/>(e.g., Heart condition)"] --> C["Hospitalized (Collider)<br/>Only studying inpatients!"]
    Y["Disease B<br/>(e.g., Broken leg)"] --> C
```

Suppose in the general public, having a heart condition and breaking your leg are completely independent ($r \approx 0$). Now suppose you study *only patients currently admitted to the hospital*. If a hospitalized patient does *not* have a broken leg, they are much more likely to have a heart condition (otherwise why are they in the hospital?). Inside the hospital-only dataset, **broken legs suddenly look "protective" against heart disease** — a classic clinical trap known as **Berkson's paradox**!

```{code-cell} python
:label: ch13-collider
# Two completely independent traits in the general population (n = 5,000)
pop_n = 5_000
heart_risk = rng.normal(0, 1, size=pop_n)
injury_risk = rng.normal(0, 1, size=pop_n)

# Either high heart risk OR high injury risk sends you to the hospital
hospitalized = (heart_risk > 1.3) | (injury_risk > 1.3)

print(f"General population correlation : r = {np.corrcoef(heart_risk, injury_risk)[0, 1]:+.2f}")
print(f"Hospitalized-only correlation  : r = "
      f"{np.corrcoef(heart_risk[hospitalized], injury_risk[hospitalized])[0, 1]:+.2f}  <-- FAKE link!")
```

Notice that! Two completely independent numbers suddenly show a strong negative correlation (`r = −0.64`) purely because we only collected data from hospitalized patients. Whenever an AI dataset comes exclusively from emergency rooms, specialist clinics, or paid health apps, ask: *who wasn't selected into this table?*

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.3.1 (Modify) — How strong must the confounder be to flip the sign?
:class: hint
In [](#ch13-simpson-sim), the severity coefficient in `outcome` was `28` (`outcome = 28 * severity - 1.2 * dose + ...`). Change `28` to `5`, `12`, and `20` and compute the pooled correlation `r_raw` each time. Roughly where does the sign flip from helpful (negative) to misleadingly harmful (positive)?
```

```{admonition} Solution
:class: dropdown
```python
for sev_effect in (5, 12, 20, 28):
    out = sev_effect * severity - 1.2 * dose + rng.normal(0, 4.5, size=n)
    print(f"severity effect = {sev_effect:>2}: pooled r = {np.corrcoef(dose, out)[0, 1]:+.2f}")
```

At `sev_effect = 5`, the pooled correlation is still slightly negative (the drug's helpful effect outweighs the weak confounding). By `12`, it is nearly zero (confounding *hides* a real drug benefit!). By `20` and `28`, it flips strongly positive (Simpson's paradox). Notice that even before a full sign reversal, confounding still silently distorts the size of the estimated effect!
```

```{admonition} Exercise 1.3.2 (Make) — Checking a real correlation in the diabetes data
:class: hint
Load `load_diabetes(as_frame=True, scaled=False)` from Chapter 1.2 and compute the correlation between `"bmi"` and `"target"` (disease progression), and between `"age"` and `"target"`. Plot a scatter plot of `"bmi"` vs. `"target"`. Does a strong positive correlation prove that lowering BMI alone will reduce the progression score by that exact slope? Why or why not?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.datasets import load_diabetes
ddf = load_diabetes(as_frame=True, scaled=False).frame
print(f"BMI vs progression: r = {ddf['bmi'].corr(ddf['target']):+.2f}")
print(f"Age vs progression: r = {ddf['age'].corr(ddf['target']):+.2f}")
```

`bmi` has a moderate-to-strong positive correlation ($r \approx +0.59$) with 1-year progression, much stronger than `age` ($r \approx +0.19$). However, this is an observational cohort, not an RCT: diet, physical activity, genetics, medication adherence, and socioeconomic access could confound both BMI and progression. The correlation is strong evidence for *predicting* risk, not proof of the exact *causal* effect of an intervention.
```

```{admonition} Exercise 1.3.3 (Critical Reasoning) — Spotting the confounder in health headlines
:class: hint
Read this real-world style headline: *"Study of 200,000 health records finds patients who take daily aspirin have a higher rate of heart attacks than people who never take aspirin."*
1. Draw (in words or arrows) the confounder behind this paradox.
2. Why would training a hospital AI model on raw observational data without knowing this causal structure be dangerous?
```

```{admonition} Solution
:class: dropdown
1. **Confounder = Pre-existing cardiovascular risk** (`High baseline heart risk → Doctor prescribes daily aspirin` AND `High baseline heart risk → Higher rate of heart attacks`). People without heart risk rarely take preventive aspirin!
2. A purely predictive AI learns *association*, not *causation*. It correctly notes that aspirin-takers have more heart attacks — which is fine for estimating *risk*, but disastrous if misread as *"stop taking aspirin to lower heart-attack risk!"* Separating **prediction** from **intervention** is a golden rule of clinical AI.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Correlation ($r$)** measures straight-line association between $-1$ and $+1$, never causation by itself.
- A **confounder** drives both treatment and outcome ($X \leftarrow Z \to Y$) and can reverse an entire conclusion (**Simpson's paradox**).
- **Stratification** controls for measured confounders; **randomized controlled trials (RCTs)** break both measured and unmeasured confounding.
- Watch out for **collider bias**: filtering a dataset to a selected group (like hospitalized patients) can fabricate fake relationships.
```

*Next up:* Even when a test or clinical study has zero confounding, our intuition about *chances* and rare diseases is famously unreliable. Let's sharpen our probabilistic toolkit in [](1_4-probability-and-base-rates.md).
