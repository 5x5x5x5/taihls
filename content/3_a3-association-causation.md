---
title: 'Chapter A3 — Do two things go together?'
short_title: 'A3 · Association vs. causation'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter A3 — Do two things go together?

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- measure how strongly two columns move together with a **correlation** coefficient;
- explain the difference between **association** and **causation**;
- recognise a **confounder** — a hidden cause lurking behind a misleading pattern;
- **stratify** a dataset to check whether an association survives once you account for
  that hidden cause.
```

## Start with a story

A hospital reviews its records and notices something alarming: patients who received
*more* of a certain drug tended to do *worse*. The chart is stark — higher dose, worse
outcome, a clear upward line. Should the hospital stop using the drug?

Hold on. There is a trap here, and learning to see it is the single most important
habit in this entire book. Two things "going together" in the data does **not** mean
one caused the other. Let's build exactly this situation from scratch so we can see
the trick from the inside — and then expose it.

```{admonition} Definition — association vs. causation
:class: important
An **association** means two things tend to occur together (when one is high, the
other tends to be high or low too). **Causation** means changing one actually *makes*
the other change. Association is easy to measure; causation is what we usually care
about — and they are not the same thing.
```

## Building a deceptive dataset

Here is the hidden truth we'll bake in. Patients arrive at three **severity** tiers —
mild, moderate, severe. How sick a patient *starts out* secretly drives two things at
once:

1. **the dose they receive** — sicker patients are given *more* drug, and
2. **their outcome** — sicker patients end up *worse*, no matter what.

Meanwhile the drug itself genuinely **helps**: holding severity fixed, more drug leads
to a *better* outcome. Severity is the lurking villain of this story.

```{admonition} Definition — confounder
:class: important
A **confounder** is a hidden variable that influences *both* of the two things you are
comparing. It can manufacture an association between them — or hide a real one — that
has nothing to do with one causing the other.
```

```{code-cell} python
:label: make-data
import numpy as np

np.random.seed(0)
n = 300

# The hidden confounder: baseline severity (1 = mild, 2 = moderate, 3 = severe).
severity = np.random.choice([1, 2, 3], size=n)

# Sicker patients get MORE drug (plus some patient-to-patient variation).
dose = 10 * severity + np.random.normal(0, 4, n)

# Outcome (higher = worse). Severity pushes it UP a lot; the drug pushes it DOWN.
outcome = 25 * severity - 1.0 * dose + np.random.normal(0, 5, n)

print(f"{n} patients across {len(np.unique(severity))} severity tiers")
```

Notice the line `-1.0 * dose`: in the data-generating truth, more drug *lowers* the
(bad) outcome. The drug helps. Keep that fact in your pocket.

## Measuring how things go together

To put a single number on "do these move together?", we use the **correlation
coefficient**.

```{admonition} Definition — correlation
:class: important
The **correlation coefficient** is a number between $-1$ and $+1$ that summarises how
two columns move together. Near $+1$: when one goes up, the other goes up. Near $-1$:
when one goes up, the other goes down. Near $0$: no straight-line relationship.
```

Let's look at the raw picture the hospital saw — dose against outcome, ignoring
severity entirely.

```{code-cell} python
:label: raw-corr
import matplotlib.pyplot as plt

r_raw = np.corrcoef(dose, outcome)[0, 1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(dose, outcome, s=14, color="#7f8c8d", alpha=0.7)
ax.set_xlabel("drug dose")
ax.set_ylabel("outcome (higher = worse)")
ax.set_title(f"Raw view: correlation = {r_raw:+.2f}")
fig
```

```{code-cell} python
:label: print-raw
print(f"raw correlation (dose vs. outcome): {r_raw:+.2f}")
```

The raw correlation is about **+0.65** — strongly positive. Taken at face value, more
drug looks clearly *harmful*. This is the chart that nearly got the drug banned. But we
built this data, so we *know* the drug helps. What went wrong?

## Stratify to reveal the truth

The fix is to compare like with like. Instead of mixing all severities together, we
**stratify** — split the patients into their severity tiers and look at the dose /
outcome relationship *within* each tier, where severity can no longer vary.

```{admonition} Definition — stratify
:class: important
To **stratify** is to split your data into groups that share the same value of a
confounder, then study the relationship *inside* each group. This stops the confounder
from moving around and muddying the comparison.
```

```{code-cell} python
:label: stratified-corr
fig, ax = plt.subplots(figsize=(6, 4))
colors = {1: "#27ae60", 2: "#f39c12", 3: "#c0392b"}
names = {1: "mild", 2: "moderate", 3: "severe"}

for tier in (1, 2, 3):
    mask = severity == tier
    r = np.corrcoef(dose[mask], outcome[mask])[0, 1]
    ax.scatter(dose[mask], outcome[mask], s=14, alpha=0.7,
               color=colors[tier], label=f"{names[tier]}: r = {r:+.2f}")
    print(f"{names[tier]:>8} tier: correlation = {r:+.2f}")

ax.set_xlabel("drug dose")
ax.set_ylabel("outcome (higher = worse)")
ax.set_title("Stratified by severity")
ax.legend(fontsize=8)
fig
```

There it is — the punchline. Overall the correlation was **+0.65** (drug looks
harmful), but *inside every single severity tier* the correlation flips to roughly
**−0.6** (drug looks helpful). The same data, the same drug — and the conclusion
**reverses** the moment we stop comparing severe patients to mild ones.

What had really happened: severe patients both got more drug *and* did worse — but for
reasons of their severity, not the drug. Severity, the confounder, faked a harmful
association out of a drug that actually helps. This flip is famous enough to have a
name — **Simpson's paradox** — and once you've seen it, you'll never trust a raw
scatter plot again.

```{tip}
This is the question to ask of *every* "X is linked to Y" headline you ever read:
**what else might cause both?** Coffee linked to heart disease? Maybe smokers drink
more coffee. The confounder is almost always the real story.
```

## Exercise

```{admonition} Exercise A3.1 — Kill the confounder
:class: hint
Suppose the hospital had instead **assigned the dose at random**, with no regard for
how sick each patient was. Change the `dose` line so the dose no longer depends on
`severity` (e.g. `dose = np.random.uniform(0, 40, n)`), keep everything else the same,
and re-compute the raw correlation. What sign is it now, and why?
```

```{admonition} Solution
:class: dropdown
```python
np.random.seed(0)
severity = np.random.choice([1, 2, 3], size=n)
dose = np.random.uniform(0, 40, n)              # dose now independent of severity
outcome = 25 * severity - 1.0 * dose + np.random.normal(0, 5, n)
print(f"raw correlation now: {np.corrcoef(dose, outcome)[0, 1]:+.2f}")
```

The raw correlation is now **negative** — the drug's true helpful effect shows up
directly, with no stratifying needed. Because the dose no longer depends on severity,
severity can't confound the comparison: random assignment **breaks the link** between
the confounder and the treatment. That is exactly why **randomised** experiments are
the gold standard for separating association from causation.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **correlation** coefficient (between $-1$ and $+1$) measures how two columns move
  together — but **association is not causation**.
- A **confounder** is a hidden variable affecting *both* sides of a comparison; it can
  invent, hide, or even *reverse* an apparent relationship.
- **Stratifying** by the confounder — comparing like with like — can flip the
  conclusion entirely (**Simpson's paradox**).
- Before believing "X causes Y," always ask **what else could cause both?** Randomly
  assigning the treatment is the cleanest way to rule confounders out.
```
