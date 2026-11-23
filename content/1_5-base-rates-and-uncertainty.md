---
title: 'Chapter 1.5 — Reading a Medical Test: Base Rates and Small-Sample Noise'
short_title: '1.5 · Base rates & sample noise'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.5 — Reading a Medical Test: Base Rates and Small-Sample Noise

*Week 3 · ~40 minutes · Prerequisites: [](1_4-correlation-and-confounding.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- use **natural frequencies** (counting real people out of 10,000) to avoid the **base-rate fallacy**;
- show why a **second confirmatory test** turns a noisy positive screen into high certainty;
- explain why a result from a **small sample** (25 patients) bounces around far more than a large study (400 patients).
```

## Start with a case: the scary "95% accurate" test result

Imagine a town screens **10,000 people** for a rare genetic condition that affects **1 in 200 people** (`0.5%`). The lab test is genuinely very good:
- If you **have** the condition, it catches you **95%** of the time ({term}`Sensitivity`).
- If you are **healthy**, it clears you **95%** of the time ({term}`Specificity`).

You take the test and it comes back **positive**. What are the chances you actually have the condition?

Most people — including many doctors in classic studies [@gigerenzer2011natural] — guess *"about 90% or 95%!"* The true answer is **under 9%**! Over 9 out of 10 positive results are false alarms.

## Count real people, not percentages (natural frequencies)

Why do percentages fool our brains? Because we forget the {term}`Base rate` — how rare the condition was *before* taking the test. The trick is called {term}`Natural frequencies`: forget percentages and just trace **10,000 real people** step by step [@gigerenzer2011natural]!

```{mermaid}
flowchart TD
    Pop["10,000 People Screened"] --> Sick["50 Truly Have It (0.5%)"]
    Pop --> Healthy["9,950 Truly Healthy (99.5%)"]
    Sick --> TP["48 Test Positive (Real cases)"]
    Sick --> FN["2 Test Negative (Missed)"]
    Healthy --> FP["498 Test Positive (False alarms!)"]
    Healthy --> TN["9,452 Test Negative (Cleared)"]
```

Stare at those two "Test Positive" boxes: **48** real cases buried inside **498** false alarms (`48 + 498 = 546` total positive results)! So out of everyone who gets a positive result, only $48 / 546 \approx \mathbf{8.8\%}$ actually have the condition. That illusion is the {term}`Base-rate fallacy`.

## How doctors fix it: a second confirmatory test!

If one positive screen leaves you at only ~8.8%, how do hospitals ever diagnose rare conditions? **They run a second, independent test on everyone who tested positive!** Now the starting group is no longer 0.5% sick — it's **8.8% sick**!

```{admonition} Predict before you run
:class: tip
Suppose the 546 people who tested positive on Screen 1 now take a second, independent confirmatory test (95% sensitive, 98% specific). If *both* tests come back positive, roughly what percentage chance is it that you truly have the condition — 25%, 50%, or **over 80%**?
```

```{code-cell} python
:label: ch15-base-rate
import matplotlib.pyplot as plt

def chance_really_sick(prevalence, sensitivity, specificity):
    true_pos  = prevalence * sensitivity
    false_pos = (1 - prevalence) * (1 - specificity)
    return true_pos / (true_pos + false_pos)

# Stage 1: screening 10,000 people at 0.5% base rate
p1 = chance_really_sick(prevalence=0.005, sensitivity=0.95, specificity=0.95)
# Stage 2: confirmatory test on the positive pool (now starting at p1!)
p2 = chance_really_sick(prevalence=p1,    sensitivity=0.95, specificity=0.98)

print(f"Before any test   : {0.005:.1%} chance")
print(f"After Screen 1 (+): {p1:.1%} chance  (still mostly false alarms!)")
print(f"After Test 2   (+): {p2:.1%} chance  (now strong evidence!)")

# Plot how PPV climbs as a condition becomes more common
rates = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.40]
ppvs  = [chance_really_sick(r, 0.95, 0.95) * 100 for r in rates]

plt.figure(figsize=(7.0, 3.8))
plt.plot([r * 100 for r in rates], ppvs, "o-", color="#0072B2", lw=2.3)
plt.xlabel("How common the condition is (% of population)")
plt.ylabel("Chance a positive result is real (%)")
plt.title("The exact same 95% test means wildly different things by rarity")
plt.grid(alpha=0.25)
plt.show()
```

Jumping from **0.5% → 8.7% → 81.9%** explains a real medical rule you've probably heard: *never panic over a single routine screening flag — wait for the confirmatory test!*

## Small-sample noise: why a 25-patient headline can mislead

One more vital habit before we start building AI models in Unit 2: **always check how many patients were tested!**

Suppose a new diagnostic app is truly **80% accurate** on the whole population. If you test it on just **25 random patients**, luck alone can make it look like 92% — or 64%! Test it on **400 patients**, and luck averages out. Let's simulate running 1,000 small studies vs. 1,000 large studies!

```{code-cell} python
:label: ch15-sample-size
import numpy as np

rng = np.random.default_rng(0)

# Simulate 1,000 trials of an AI whose TRUE accuracy is 80%
small_studies = [rng.choice([0, 1], size=25,  p=[0.2, 0.8]).mean() * 100 for _ in range(1000)]
large_studies = [rng.choice([0, 1], size=400, p=[0.2, 0.8]).mean() * 100 for _ in range(1000)]

print(f"Small studies (n =  25): typical range = {np.percentile(small_studies, 2.5):.0f}% to {np.percentile(small_studies, 97.5):.0f}%")
print(f"Large studies (n = 400): typical range = {np.percentile(large_studies, 2.5):.0f}% to {np.percentile(large_studies, 97.5):.0f}%")

plt.figure(figsize=(7.2, 3.8))
plt.hist(small_studies, bins=16, alpha=0.45, color="#D55E00", label="Small pilot study (n = 25)")
plt.hist(large_studies, bins=22, alpha=0.75, color="#0072B2", label="Large study (n = 400)")
plt.axvline(80, color="#111111", ls="--", lw=1.6, label="True accuracy (80%)")
plt.xlabel("Measured study accuracy (%)")
plt.ylabel("Number of simulated studies")
plt.title("Same 80% AI — small studies bounce wildly, large studies stay sharp")
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.show()
```

In a 25-patient pilot study, a plain 80% tool can easily score **92% or 96%** purely by luck! Whenever a headline boasts *"New AI hits 94% accuracy!"*, immediately ask: **out of how many patients?**

```{admonition} Going deeper — Bayes' theorem & the bootstrap
:class: dropdown
1. **Bayes' theorem:** Our `chance_really_sick` function is literally Bayes' rule written in natural frequencies:
   $$P(\text{Sick} \mid +) = \frac{P(+\mid \text{Sick})\,P(\text{Sick})}{P(+\mid \text{Sick})\,P(\text{Sick}) + P(+\mid \text{Healthy})\,P(\text{Healthy})}$$
2. **The {term}`Bootstrap`:** In real studies we don't know the true accuracy (80%), so scientists estimate that orange/blue spread by randomly resampling their own $n$ patients *with replacement* 1,000 times — producing a **95% confidence interval**!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.5.1 (Modify) — When the condition is common
:class: hint
Call `chance_really_sick(prevalence=0.30, sensitivity=0.95, specificity=0.95)` for a *common* winter illness (30% of clinic visitors have it). How high is the chance a positive result is real now?
```

```{admonition} Solution
:class: dropdown
```python
print(f"At 30% prevalence: {chance_really_sick(0.30, 0.95, 0.95):.1%}")
```
It jumps to **89.1%**! When the illness is common, true positives vastly outnumber false alarms — the exact same test becomes far more decisive.
```

```{admonition} Exercise 1.5.2 (Make) — Boosting sensitivity vs. specificity on a rare disease
:class: hint
At `prevalence = 0.005` (1 in 200), compare:
- Test A: `sensitivity = 0.99, specificity = 0.95`
- Test B: `sensitivity = 0.95, specificity = 0.99`

Which upgrade raises `chance_really_sick` much more, and why?
```

```{admonition} Solution
:class: dropdown
```python
print(f"Test A (99% sens, 95% spec): {chance_really_sick(0.005, 0.99, 0.95):.1%}")
print(f"Test B (95% sens, 99% spec): {chance_really_sick(0.005, 0.95, 0.99):.1%}")
```
**Test B (`32.3%` vs. `9.0%`)!** Because 99.5% of screened people are healthy, cutting false alarms from 5% down to 1% slashes false positives by **5×**!
```

```{admonition} Exercise 1.5.3 (Critical Reasoning) — Judging a glowing startup claim
:class: hint
A startup posts on social media: *"Our AI detected cancer with 95% accuracy in a clinical test of 20 patients!"* Give **two** sharp questions from Unit 1 that you would ask before trusting that post.
```

```{admonition} Solution
:class: dropdown
1. **Sample size:** 20 patients is tiny — getting just 2 more cases wrong drops accuracy from 95% to 85%! Where is a larger study?
2. **Base rate & error breakdown:** How many of the 20 actually had cancer? If only 1 of 20 had cancer, always saying "healthy" gets 95% accuracy while catching **zero** cancers! (We'll unpack this exact trap in Unit 2!)
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Natural frequencies** ("out of 10,000 people…") prevent the **base-rate fallacy** when reading rare-disease tests [@gigerenzer2011natural].
- **Two-stage testing** (screen + confirmatory test) turns a noisy initial alert into strong evidence.
- **Small samples are noisy:** always check sample size before trusting a bold percentage.
```

*End of Unit 1!* You now think like a sharp data detective. In **Unit 2**, we start teaching computers to make medical predictions — building our very first breast-tumor classifier from scratch in [](2_1-nearest-neighbors.md)!
