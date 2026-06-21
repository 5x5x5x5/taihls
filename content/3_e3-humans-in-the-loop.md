---
title: 'Chapter E3 — Keeping humans in the loop'
short_title: 'E3 · Humans in the loop'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter E3 — Keeping humans in the loop

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/5x5x5x5/taihls/HEAD)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why a model should sometimes say **"I'm not sure"** instead of guessing;
- use a classifier's **confidence** (`predict_proba`) to **abstain** on hard cases;
- plot the trade-off between **coverage** (how much it decides) and **accuracy**;
- apply a reusable **checklist** for judging any AI claim you meet in the wild.
```

## Start with a story

A radiology AI flags chest X-rays as "normal" or "needs review." On most images it is
confident and correct. But on a blurry, oddly positioned scan it is barely more sure
than a coin flip — and it guesses anyway, marking a subtle pneumonia as "normal."

The fix is not a smarter model. It is a humbler one. A good system knows the
*difference* between a case it understands and a case it doesn't, and on the hard
cases it **defers to a human** rather than gambling. Tools like this work best as
partners to clinicians, not replacements [@rajpurkar2022ai]. Let's name the idea.

```{admonition} Definition — human-in-the-loop
:class: important
A **human-in-the-loop** system lets a model handle the cases it is confident about,
while routing the uncertain or high-stakes cases to a **person** for the final call.
The model does the easy bulk; humans catch what the model would have gotten wrong.
```

## Confidence: how sure is the model?

Most classifiers can report not just an answer but a *probability* for each class.
The size of that probability is a rough **confidence** score.

```{admonition} Definition — confidence (and calibration)
:class: important
A model's **confidence** is the probability it assigns to its chosen answer (e.g.
`0.97` means "97% sure"). A model is **well-calibrated** if, among all the times it
says "90% sure," it really is right about 90% of the time. We won't fix calibration
here — we just use confidence to decide *when to ask a human*.
```

Let's train a classifier on the breast-cancer dataset from [](1_b1-knn-tumor.md) and
look at its confidences.

```{code-cell} python
:label: e3-train
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

model = LogisticRegression(max_iter=5000).fit(X_train, y_train)

proba = model.predict_proba(X_test)   # probability of each class
pred = model.predict(X_test)          # the chosen answer
confidence = proba.max(axis=1)        # how sure it is of that answer

baseline = (pred == y_test).mean()
print(f"If the model decides EVERY case, accuracy = {baseline:.1%}")
print(f"Confidence ranges from {confidence.min():.0%} (a near coin-flip) "
      f"to {confidence.max():.0%} (almost certain).")
```

Deciding everything gives about **96%** accuracy. Respectable — but the misses are
hiding among the low-confidence cases. What if we let the model *opt out* of those?

## Selective prediction: decide only when sure

The rule is simple. Pick a **confidence threshold**. If the model's confidence is
above it, accept the model's answer (**auto-decide**). Otherwise, **abstain** and send
the case to a human.

```{admonition} Definition — abstention (selective prediction)
:class: important
**Abstention**, or **selective prediction**, is when a model declines to answer cases
below a confidence threshold, leaving them for a human. **Coverage** is the fraction
of cases it *does* decide; the rest are deferred.
```

```{code-cell} python
:label: e3-threshold
threshold = 0.95
auto = confidence >= threshold          # cases the model is confident enough to keep

coverage = auto.mean()
auto_accuracy = (pred[auto] == y_test[auto]).mean()
deferred = (~auto).sum()

print(f"Threshold = {threshold:.0%}")
print(f"Coverage      : {coverage:.0%}  (model auto-decides this share)")
print(f"Auto-accuracy : {auto_accuracy:.0%}  (accuracy on the cases it kept)")
print(f"Deferred to a human: {deferred} of {len(y_test)} cases")
```

At a 95% threshold the model auto-decides about **80%** of cases and is **100%
correct** on them, handing the trickiest ~20% to a clinician. We traded a little
*coverage* for a lot of *reliability* — exactly the bargain a high-stakes setting
wants.

## The coverage–accuracy trade-off

Raising the threshold makes the model pickier: it decides fewer cases (coverage drops)
but is righter on the ones it keeps (accuracy rises). Let's sweep the threshold and
watch both curves.

```{code-cell} python
:label: e3-fig
import matplotlib.pyplot as plt

thresholds = np.linspace(0.50, 0.99, 40)
coverages, accuracies = [], []
for t in thresholds:
    keep = confidence >= t
    coverages.append(keep.mean())
    accuracies.append((pred[keep] == y_test[keep]).mean())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, coverages, marker="o", markersize=3,
        color="#2980b9", label="coverage (share auto-decided)")
ax.plot(thresholds, accuracies, marker="s", markersize=3,
        color="#27ae60", label="accuracy on auto-decided cases")
ax.axhline(baseline, color="#c0392b", linestyle="--", linewidth=1,
           label=f"accuracy if deciding everything ({baseline:.0%})")
ax.set_xlabel("confidence threshold")
ax.set_ylabel("fraction")
ax.set_title("Defer the hard cases, get the easy ones right")
ax.legend(loc="lower left", fontsize=8)
fig
```

In [](#e3-fig) the two lines pull apart: as we move right, the green accuracy line
climbs above the red "decide-everything" baseline while the blue coverage line slides
down. There is no single "best" threshold — it depends on how costly a mistake is
versus how much human time you have. That choice is a *human* decision about an AI
system, which is the whole point.

```{tip}
A neat way to read [](#e3-fig): pick the accuracy you *need* (say, 99%), then read off
how much human help it costs. Here, demanding ~99% auto-accuracy still lets the model
handle roughly 9 in 10 cases on its own — a big workload lifted, with the riskiest
tenth kept under human eyes.
```

## A checklist for any AI claim

You will meet AI claims everywhere — in product demos, news, and research. Here is a
reusable checklist that pulls together the whole of Part E.

```{admonition} Checklist — questioning an AI claim
:class: important
Before you trust an AI result, ask:
1. **Accuracy for whom?** Is the score broken down by group, or only an average that
   could hide a gap? (See [](1_e1-bias-in-medical-ai.md).)
2. **What data trained it?** Who is over- or under-represented? Could there be
   **distribution shift** between training and use?
3. **What is the baseline?** Beating a coin flip — or even "always guess the common
   class" — is a low bar. (See [](1_b1-knn-tumor.md).)
4. **How was it tested?** On *held-out* data the model never saw, or on the same data
   it trained on?
5. **What happens when it's wrong?** Who is harmed, and is there a **human in the
   loop** for high-stakes or low-confidence cases?
6. **Was the data used responsibly?** Consent, privacy, de-identification. (See
   [](2_e2-privacy-consent.md).)
7. **Can it say "I don't know"?** Or is it forced to guess on every input?
```

## Exercises

```{admonition} Exercise E3.1 — Pick a threshold for 99% accuracy
:class: hint
Suppose the clinic insists the AI be **at least 99% accurate** on whatever it
auto-decides. Using the loop in [](#e3-fig), find the *lowest* threshold that achieves
this, and report the coverage there. (Lower thresholds keep more cases, so we want the
smallest threshold that still clears 99%.)
```

```{admonition} Solution
:class: dropdown
```python
for t, acc, cov in zip(thresholds, accuracies, coverages):
    if acc >= 0.99:
        print(f"threshold {t:.2f}: accuracy {acc:.0%}, coverage {cov:.0%}")
        break
```

You should find a threshold around **0.79** already reaches 99% auto-accuracy while
still covering about **91%** of cases. In other words, insisting on near-perfect
auto-decisions costs the clinic only about 1 case in 10 sent to a human — a cheap
price for the safety it buys.
```

```{admonition} Exercise E3.2 — When abstention backfires
:class: hint
Abstention is not magic. Describe a situation where deferring "uncertain" cases to
humans could make a *fairness* problem (Chapter E1) **worse**, not better.
```

```{admonition} Solution
:class: dropdown
If the model is systematically *less confident* for an under-represented group, then
abstention will defer that group's cases far more often. The humans then carry the
extra load for that group — and if the humans are also rushed or biased, the group is
doubly disadvantaged. Worse, the deferred cases might quietly get *less* care, not
more. Abstention helps only if the deferred cases genuinely get good human attention,
and if you **check coverage and accuracy per group**, exactly as in E1 — never just
overall.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A trustworthy model can **abstain**: it declines low-**confidence** cases instead of
  guessing, and hands them to a **human in the loop**.
- There is a **coverage–accuracy trade-off**: a higher threshold decides fewer cases
  but gets more of them right. The right threshold is a human judgement about
  stakes and resources.
- Confidence is only useful if the model is reasonably **calibrated** — and abstention
  must be checked **per group**, or it can deepen the biases of Chapter E1.
- The **checklist** is your portable tool: question accuracy-for-whom, the data,
  baselines, testing, failure modes, privacy, and whether the system can say "I don't
  know." Put it to work in the [](4_e4-capstone.md).
```
