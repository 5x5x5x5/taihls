---
title: 'Chapter 1.4 — Thinking in Chances: Base Rates, Diagnostic Tests, and Uncertainty'
short_title: '1.4 · Probability & base rates'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 1.4 — Thinking in Chances: Base Rates, Diagnostic Tests, and Uncertainty

*Estimated time: ~40 minutes · Prerequisites: [](1_3-correlation-confounding-causation.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- use **natural frequencies** to calculate what a positive diagnostic test actually means;
- explain the **base-rate fallacy** and show how **sequential confirmatory testing** overcomes it;
- quantify **sampling uncertainty** using a 10-line computational **bootstrap** confidence interval;
- judge why small pilot studies produce noisy medical headlines.
```

## Start with a case: the "95% accurate" screening scare

Suppose a clinic rolls out a rapid screening test for a rare genetic condition that affects **1 in 200 people** ($0.5\%$ of the population). The brochure proudly reports two numbers:
- Among people who truly have the condition, the test catches **95%** of them ({term}`Sensitivity (Recall)`).
- Among healthy people, the test correctly clears **95%** of them ({term}`Specificity`).

You take the test as part of a routine checkup, and it comes back **positive**. How likely are you to truly have the condition?

When psychologist Gerd Gigerenzer posed puzzles like this to experienced physicians, most answered *"around 90% to 95%"* [@gigerenzer2011natural]. The true answer is **less than 9%** — over **9 out of 10** positive results are false alarms!

Why does human intuition miss by an order of magnitude? Because percentages hide the {term}`Base rate` (how rare the condition was *before* the test). The antidote is gloriously simple: **stop juggling percentages and count real people.**

## Natural frequencies: counting a town of 10,000

```{admonition} Definition — base rate and natural frequencies
:class: important
The {term}`Base rate` (or *prevalence*) is the fraction of a population that truly has a condition before any test is run. {term}`Natural frequencies` express probabilities as concrete counts out of a round population (such as 10,000 people), turning tricky conditional probability into ordinary arithmetic [@gigerenzer2011natural].
```

Trace a town of 10,000 people through the screening tree:

```{mermaid}
flowchart TD
    Pop["10,000 Screened People"] --> Sick["50 Truly Have Condition (0.5%)"]
    Pop --> Healthy["9,950 Truly Healthy (99.5%)"]
    Sick --> TP["48 Test Positive (True Positives)<br/>95% of 50"]
    Sick --> FN["2 Test Negative (False Negatives)<br/>5% missed"]
    Healthy --> FP["498 Test Positive (False Positives)<br/>5% false alarm rate"]
    Healthy --> TN["9,452 Test Negative (True Negatives)<br/>95% cleared"]
```

Stare at the two "Test Positive" boxes in that tree: **48** real cases, drowned inside **498** false alarms! Even a modest 5% false-alarm rate applied to a huge crowd of 9,950 healthy people generates ten times as many positives as the rare sick group. That shock is the {term}`Base-rate fallacy`.

## Lab 1: Base rates and the power of a second test

If a single positive screen leaves you with only an ~8.8% chance of truly having the condition, how does medicine ever diagnose rare diseases? **Two-stage (sequential) testing!** After a positive screen, the patient moves into a much higher-prevalence pool (~8.8% instead of 0.5%) and takes an independent confirmatory test (often with even higher specificity).

```{admonition} Predict before you run
:class: tip
Suppose everyone who tests positive on Screen 1 (where your chance of truly being sick is just **8.7%**) now takes an independent **Confirmatory Test 2** (95% sensitive, 98% specific). If *both* tests come back positive, roughly how high does your chance of truly having the condition jump — 25%, 50%, or over 80%?
```

```{code-cell} python
:label: ch14-natural-freq
import numpy as np

def test_update(prevalence, sensitivity, specificity):
    """Given a starting prevalence (base rate), return the post-test probability
    of truly having the condition after a positive test result."""
    tp = prevalence * sensitivity
    fp = (1 - prevalence) * (1 - specificity)
    return tp / (tp + fp)

p0 = 0.005                                          # 1 in 200 people (0.5%)
p1 = test_update(p0, sensitivity=0.95, specificity=0.95)
p2 = test_update(p1, sensitivity=0.95, specificity=0.98)

# Show concrete natural frequencies out of 10,000 people for Stage 1
N = 10_000
sick = round(N * p0)
healthy = N - sick
tp1 = round(sick * 0.95)
fp1 = round(healthy * 0.05)

print(f"Out of {N:,} people: {sick} truly sick, {healthy:,} truly healthy")
print(f"Stage 1 positives : {tp1} true positives + {fp1} false positives -> PPV = {p1:.1%}")
print(f"After Stage 2 (+) : chance truly sick jumps from {p1:.1%} -> {p2:.1%}!")
```

:::{margin}
This exact two-step update *is* **Bayes' theorem** in disguise: yesterday's posterior probability (`p1 = 8.7%`) becomes today's new base rate for the second test!
:::

From **0.5%** → **8.7%** → **81.9%** in two tests! Notice how much cleaner medicine becomes when you see screening as a funnel: Stage 1 casts a wide, inexpensive net to catch almost every sick person, and Stage 2 filters out the false alarms before anyone undergoes risky treatment.

Let's visualize how dramatically a test's **Positive Predictive Value** (the chance a positive result is real) depends on the underlying base rate.

```{code-cell} python
:label: ch14-baserate-fig
import matplotlib.pyplot as plt

BLUE, VERMILION, TEAL, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"

prevalences = np.logspace(-3.3, -0.3, 200)   # 0.05% to 50%
ppv_95 = [test_update(p, 0.95, 0.95) * 100 for p in prevalences]
ppv_99 = [test_update(p, 0.95, 0.99) * 100 for p in prevalences]
ppv_2step = [test_update(test_update(p, 0.95, 0.95), 0.95, 0.98) * 100 for p in prevalences]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(prevalences * 100, ppv_95, lw=2.3, color=VERMILION, label="Single test (95% spec)")
ax.plot(prevalences * 100, ppv_99, lw=2.3, color=BLUE, label="Single test (99% spec)")
ax.plot(prevalences * 100, ppv_2step, lw=2.5, color=TEAL, ls="--",
        label="Two-stage screen + confirm")
ax.axvline(0.5, color="#4D4D4D", ls=":", lw=1.2)
ax.text(0.55, 22, "0.5% rare disease\n(1 in 200)", fontsize=8.5, color="#4D4D4D")
ax.set_xscale("log")
ax.set_xlabel("Disease base rate / prevalence (%, log scale)")
ax.set_ylabel("Chance a positive result is real (PPV, %)")
ax.set_title("The exact same test means wildly different things as rarity changes")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.25, which="both")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

## Lab 2: Measuring uncertainty with the bootstrap

So far we treated numbers like "95% sensitivity" or "84% recovery" as exact truths. In reality, every medical number comes from a **finite sample of patients**. If a clinic tests an AI tool on just **25 patients** and gets 22 right ($88\%$), how confident should we be that its real accuracy on the next 10,000 patients is truly around 88%?

Traditional statistics classes spend months deriving formulas for error bars. With Python, we have a universal, formula-free powerhouse called the {term}`Bootstrap`:

1. Treat your sample of $n$ patients as a miniature stand-in for the population.
2. Draw $n$ patients **with replacement** (some patients get picked twice, some zero times) and recompute the accuracy.
3. Repeat 2,000 times and check where the middle 95% of resampled scores land! That range is your **95% bootstrap confidence interval**.

```{admonition} Predict before you run
:class: tip
Compare two studies that *both* report **88% accuracy**: Study A tested **25 patients** (22/25 correct) and Study B tested **400 patients** (352/400 correct). Roughly how much narrower will Study B's 95% confidence interval be — half as wide, a quarter as wide, or identical?
```

```{code-cell} python
:label: ch14-bootstrap
rng = np.random.default_rng(0)

def bootstrap_accuracy(n_correct, n_total, n_boot=3000):
    outcomes = np.array([1] * n_correct + [0] * (n_total - n_correct))
    # Resample with replacement n_boot times
    draws = rng.choice(outcomes, size=(n_boot, n_total), replace=True)
    accs = draws.mean(axis=1)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return accs, lo, hi

accs_small, lo_s, hi_s = bootstrap_accuracy(22, 25)      # 88% on 25 patients
accs_large, lo_l, hi_l = bootstrap_accuracy(352, 400)    # 88% on 400 patients

print(f"Pilot study (n =  25): 88.0%  |  95% CI = [{lo_s:.1%}, {hi_s:.1%}]  "
      f"(width = {hi_s - lo_s:.1%})")
print(f"Large study (n = 400): 88.0%  |  95% CI = [{lo_l:.1%}, {hi_l:.1%}]  "
      f"(width = {hi_l - lo_l:.1%})")

fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.hist(accs_small * 100, bins=18, density=True, alpha=0.45, color=VERMILION,
        label=f"Small study (n=25): 95% CI [{lo_s:.0%}, {hi_s:.0%}]")
ax.hist(accs_large * 100, bins=24, density=True, alpha=0.75, color=BLUE,
        label=f"Large study (n=400): 95% CI [{lo_l:.0%}, {hi_l:.0%}]")
ax.axvline(88, color="#111111", ls="--", lw=1.6, label="Observed score (88%)")
ax.set_xlabel("Bootstrapped accuracy (%)")
ax.set_ylabel("Relative likelihood")
ax.set_title("Same 88% headline — wildly different certainty")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

Notice the square-root rule in action: multiplying sample size by **16×** (from 25 to 400) shrank the uncertainty interval by **$\sqrt{16} = 4\times$** (from a wide 24-point spread `[76%, 100%]` down to a crisp 6-point spread `[85%, 91%]`)! In a 25-patient pilot study, a model that looks "88% accurate" could easily be only 76% accurate in reality — barely better than existing care. Always ask: *"88% out of how many patients, and what is the confidence interval?"*

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 1.4.1 (Modify) — Sensitivity vs. specificity on a rare disease
:class: hint
In [](#ch14-natural-freq), start from `prevalence = 0.005` (1 in 200) and a test with 95% sensitivity and 95% specificity (PPV = 8.7%). Compare two upgrades costing the same amount:
- **Upgrade A:** Raise **sensitivity** from 95% to **99%** (keep specificity 95%).
- **Upgrade B:** Raise **specificity** from 95% to **99%** (keep sensitivity 95%).

Which upgrade boosts PPV more dramatically, and why?
```

```{admonition} Solution
:class: dropdown
```python
print(f"Upgrade A (99% sens, 95% spec): PPV = {test_update(0.005, 0.99, 0.95):.1%}")
print(f"Upgrade B (95% sens, 99% spec): PPV = {test_update(0.005, 0.95, 0.99):.1%}")
```

**Upgrade B wins by a landslide** (PPV leaps to **32.3%**, whereas Upgrade A barely moves from 8.7% to **9.0%**)! For a rare disease, almost everyone tested is healthy — so slashing the false-positive rate from 5% to 1% cuts false alarms by **5×**, dramatically raising the reliability of a positive result.
```

```{admonition} Exercise 1.4.2 (Make) — Bootstrap the median diabetes progression
:class: hint
Using `load_diabetes(as_frame=True, scaled=False)` from Chapter 1.2, write a 6-line bootstrap loop (2,000 resamples) to compute a **95% confidence interval for the median** 1-year progression score across the 442 patients.
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.datasets import load_diabetes
prog = load_diabetes(as_frame=True, scaled=False).frame["target"].to_numpy()
rng_ex = np.random.default_rng(1)
boot_medians = [np.median(rng_ex.choice(prog, size=len(prog), replace=True))
                for _ in range(2000)]
lo, hi = np.percentile(boot_medians, [2.5, 97.5])
print(f"Observed median = {np.median(prog):.1f}  |  95% CI = [{lo:.1f}, {hi:.1f}]")
```

Notice how effortless the bootstrap is: exact classical formulas for the confidence interval of a *median* are messy, yet resampling computes a rock-solid interval (roughly `[131, 152]`) in four lines of code!
```

```{admonition} Exercise 1.4.3 (Critical Reasoning) — Why not screen everyone for every rare disease?
:class: hint
A consumer wellness company launches a whole-body AI scan that screens healthy 20-year-olds for 40 rare cancers at once, boasting "96% specificity per condition." Explain in 2–3 sentences why this service could cause widespread anxiety and unnecessary biopsies even if the AI works exactly as advertised.
```

```{admonition} Solution
:class: dropdown
Two compounding traps are at work:
1. **Base-rate fallacy:** For healthy 20-year-olds, the prevalence of each rare cancer is tiny (often < 1 in 10,000), so even a 4% false-positive rate generates hundreds of false alarms for every true case.
2. **Multiple testing:** Running 40 separate screens with a 4% false-alarm rate each means the probability of getting **zero** false alarms is only $0.96^{40} \approx 19.5\%$ — meaning over **80% of completely healthy customers** will receive at least one alarming "abnormal" finding!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Never interpret a positive test without the **base rate**; converting percentages into **natural frequencies** prevents the **base-rate fallacy**.
- **Two-stage testing** (sensitive screen → specific confirmation) turns a noisy initial alert into high diagnostic certainty.
- Every accuracy number from a finite study carries **sampling uncertainty**; the **bootstrap** gives an intuitive 95% confidence interval in a few lines of Python.
- Small pilot studies have wide error bars — look for sample sizes and confidence intervals before trusting headlines.
```

*End of Part 1!* You now think like a skeptical biomedical scientist: modeling assumptions, data distributions, confounding vs. causation, base rates, and uncertainty. In **Part 2**, we start teaching computers to make clinical decisions — beginning with our first real breast-cancer classifier in [](2_1-nearest-neighbors-and-features.md).
