---
title: 'Chapter 5.4 — Knowing When to Say "I Don’t Know": Humans in the Loop'
short_title: '5.4 · Humans in the loop'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 5.4 — Knowing When to Say "I Don't Know": Humans in the Loop

*Week 14 · ~40 minutes · Prerequisites: [](2_4-mistakes-and-thresholds.md), [](5_1-fairness-and-bias.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/05-fairness-and-abstention-audit.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why a safe medical AI should sometimes **abstain** ("I'm not sure") and hand hard cases to a human expert;
- use predicted confidence scores to plot the trade-off between **coverage** and **accuracy**;
- check whether abstention defers one patient subgroup unfairly;
- describe two real human-psychology traps: **automation bias** and **alert fatigue**.
```

## Start with a case: the humblest AI in the clinic

Imagine a radiology AI reviewing hundreds of breast biopsies every morning. On 85% of scans, the answer is crystal clear — the model is more than **95% confident**. On the tricky 15%, the cells look borderline — confidence hovers near **55%**, barely better than a coin flip.

A careless program forces itself to guess on every single scan. A smart {term}`Human-in-the-loop` system handles the clear routine cases and says **"I'm unsure — send this one straight to a specialist!"** on the borderline ones [@rajpurkar2022ai].

```{admonition} Definition — abstention (selective prediction)
:class: important
{term}`Abstention` lets an AI skip automatic decisions whenever its confidence falls below a chosen cutoff, handing those uncertain cases to a human. **Coverage** is the share of cases the AI handles automatically; **auto-accuracy** is how often it gets those kept cases right.
```

## Watch abstention eliminate almost every mistake!

Let's train our breast-cancer classifier and let it **abstain** whenever its confidence (`max(P(benign), P(malignant))`) drops below **`92%`**!

```{admonition} Predict before you run
:class: tip
When forced to decide **100%** of held-out biopsies, our model gets **`95.3%`** right (`8` mistakes out of `171`). If we let it defer the most uncertain **~13%** of cases to a pathologist, how high do you think accuracy on the remaining auto-decided cases will jump?
```

```{code-cell} python
:label: ch54-abstention
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
X, y = cancer.data, (cancer.target == 0).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)
model = LogisticRegression(max_iter=1000).fit(sc.transform(X_tr), y_tr)

probs = model.predict_proba(sc.transform(X_te))
preds = probs.argmax(axis=1)
conf  = probs.max(axis=1)                      # how confident the AI is (0.50 to 1.00)
full_acc = (preds == y_te).mean()

# Test one clear operating cutoff: 92% confidence
keep = conf >= 0.92
print(f"Decide everything (100% coverage) : accuracy = {full_acc:.1%} ({(preds != y_te).sum()} mistakes)")
print(f"Abstain below 92% confidence      : coverage = {keep.mean():.1%} | "
      f"auto-accuracy = {(preds[keep] == y_te[keep]).mean():.1%} | "
      f"sent to human = {(~keep).sum()}/{len(y_te)} cases")

# Check whether benign and malignant cases are deferred at fair rates
for lbl, name in [(0, "Benign"), (1, "Malignant")]:
    m = y_te == lbl
    print(f"  {name:<10}: sent to human = {(~keep[m]).mean():.1%}")

# Sweep confidence cutoffs from 50% to 99%
cutoffs = np.linspace(0.50, 0.99, 40)
covs = [(conf >= t).mean() * 100 for t in cutoffs]
accs = [(preds[conf >= t] == y_te[conf >= t]).mean() * 100 for t in cutoffs]

plt.figure(figsize=(7.2, 3.9))
plt.plot(cutoffs * 100, covs, color="#0072B2", lw=2.3, label="Coverage (% auto-decided)")
plt.plot(cutoffs * 100, accs, color="#009E73", lw=2.4, label="Accuracy on auto-decided cases")
plt.axhline(full_acc * 100, color="#D55E00", ls="--", label=f"Decide everything ({full_acc:.1%})")
plt.xlabel("Minimum confidence required to auto-decide (%)")
plt.ylabel("Percentage (%)")
plt.ylim(55, 102)
plt.title("Defer the hardest ~14% to specialists → 99.3% accuracy on the rest!")
plt.legend(frameon=False, loc="lower left", fontsize=8.5)
plt.grid(alpha=0.25)
plt.show()
```

Deferring just `23` tricky biopsies (`13.5%`) cut mistakes from **8 down to 1** (`99.3%` accuracy!) while lifting 86.5% of the routine workload!

## Two human traps: over-trusting vs. tuning out

Adding a human to the loop isn't magic — psychology matters too!
- **{term}`Automation bias`:** Just like drivers who blindly follow a GPS onto a closed road, tired doctors at hour 11 of a hospital shift sometimes accept a wrong AI suggestion even when they would have caught the mistake on their own!
- **Alert fatigue:** If an alarm buzzes with too many false positives (remember Chapter 1.5!), exhausted nurses start clicking *"Dismiss"* automatically — risking missing a real emergency!

```{admonition} Going deeper — full clinical audit notebook on Colab
:class: dropdown
Want to run a combined calibration, subgroup-fairness, and abstention audit on real clinical diabetes data? Try [Companion Colab Notebook 05](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/05-fairness-and-abstention-audit.ipynb) (runs instantaneously on CPU!).
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 5.4.1 (Modify) — Pick a cutoff for 100% auto-accuracy
:class: hint
Using `cutoffs`, `accs`, and `covs` in [](#ch54-abstention), find the **lowest** confidence cutoff that hits **100.0%** accuracy on auto-decided cases. What share of biopsies (`coverage`) does the model still handle automatically there?
```

```{admonition} Solution
:class: dropdown
```python
for t, acc, cov in zip(cutoffs, accs, covs):
    if acc >= 99.99:
        print(f"Lowest cutoff for 100% auto-acc: {t:.2f} -> coverage = {cov:.1%}")
        break
```
Around `t = 0.94`, the model achieves **100% accuracy** on roughly **84%** of all biopsies — sending the trickiest 16% straight to a pathologist!
```

```{admonition} Exercise 5.4.2 (Make) — Check confidence on hits vs. mistakes
:class: hint
Compare `np.median(conf[preds == y_te])` (typical confidence when right) against `np.median(conf[preds != y_te])` (typical confidence when wrong). Why does abstention *fail* if a poorly calibrated model is 99% confident even on its mistakes?
```

```{admonition} Solution
:class: dropdown
```python
print(f"Median confidence when RIGHT : {np.median(conf[preds == y_te]):.1%}")
print(f"Median confidence when WRONG : {np.median(conf[preds != y_te]):.1%}")
```
When right, median confidence is near `100%`; when wrong, it sits much lower (`~77%`). If a model were overconfident on its own errors, thresholding confidence couldn't tell uncertain cases apart from easy ones!
```

```{admonition} Exercise 5.4.3 (Critical Reasoning) — When abstention can accidentally worsen fairness
:class: hint
Suppose an AI was trained mostly on Group A (Chapter 5.1), so it feels much *less confident* on Group B and defers **60%** of Group B patients to a slow waiting list while auto-clearing **95%** of Group A instantly. Explain why abstention needs a **per-group coverage audit**!
```

```{admonition} Solution
:class: dropdown
If deferred patients wait weeks longer for care, deferring Group B at 12× the rate of Group A delays their treatment — turning a safety feature into a new source of unequal care! Always check **both accuracy AND deferral rate per group**.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Abstention** defers low-confidence cases to humans, sharply raising reliability on the auto-decided bulk.
- Safe **human-in-the-loop** workflows guard against **automation bias** (over-trusting AI) and **alert fatigue** (too many false alarms).
- Always audit deferral rates **per subgroup** so safety filters stay fair.
```

*Next up:* How does a promising Python prototype actually earn permission to enter real hospitals? Let's trace clinical trials, safety rules, and your portable **8-Question Critical Thinking Checklist** in [](5_5-from-lab-to-hospital.md)!
