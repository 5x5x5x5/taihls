---
title: 'Chapter B3 — The cost of being wrong'
short_title: 'B3 · Sensitivity & specificity'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter B3 — The cost of being wrong

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- read a **confusion matrix** and name its four cells;
- define **sensitivity (recall)**, **specificity**, and **precision (PPV)**;
- distinguish a **false positive** from a **false negative** and weigh their costs;
- explain the **base-rate fallacy**: why a good test still raises many false alarms
  when a disease is rare.
```

## Start with a story

In [](2_b2-train-test.md) we learned to grade a model honestly with a single number:
accuracy. But "right 95% of the time" can hide a deadly detail. Imagine a screening
test for a serious cancer. Being wrong by *calling a healthy person sick* means an
anxious week and an extra biopsy. Being wrong by *calling a sick person healthy*
means a tumor goes untreated. Those two mistakes are not equal — yet plain accuracy
lumps them together. To run a medical test responsibly, we have to count our mistakes
*by type*. Let's build the tool that does it.

## Two questions, four answers

Every prediction our classifier makes is either *positive* (we say "disease") or
*negative* (we say "healthy"), and reality is also one of those two. That gives four
possible outcomes, which we lay out in a small table.

```{admonition} Definition — confusion matrix
:class: important
A **confusion matrix** is a 2×2 table that counts how predictions line up with the
truth: **true positives** (sick, correctly flagged), **true negatives** (healthy,
correctly cleared), **false positives** (healthy, wrongly flagged), and **false
negatives** (sick, wrongly cleared).
```

The two kinds of error have names worth burning into memory.

```{admonition} Definition — false positive and false negative
:class: important
A **false positive (FP)** is a false alarm: the test says *disease* but the person
is healthy. A **false negative (FN)** is a missed case: the test says *healthy* but
the person is sick. In screening, an FN is usually the more dangerous of the two.
```

Let's build the matrix on our breast-cancer data. We keep the convention from B1:
the *positive* class is **malignant** (the thing we want to catch), so we relabel for
clarity and train a simple classifier.

```{code-cell} python
:label: b3-confusion
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X = data.data
# Make "malignant" the positive (1) class: catching cancer is the job.
y = (data.target == 0).astype(int)        # 1 = malignant, 0 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
pred = clf.predict(X_test)

# Order the matrix as [[TN, FP], [FN, TP]] by listing negatives first.
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
print(f"True negatives  (healthy, cleared):   {tn}")
print(f"False positives (healthy, flagged):   {fp}")
print(f"False negatives (sick, missed):       {fn}")
print(f"True positives  (sick, caught):       {tp}")
```

## Three numbers that matter more than accuracy

From those four counts we build three rates. Each answers a different real-world
question.

```{admonition} Definition — sensitivity (recall)
:class: important
**Sensitivity**, also called **recall**, is the fraction of truly sick people the
test catches: $\text{TP} / (\text{TP} + \text{FN})$. High sensitivity means few
missed cases — the priority when a miss is dangerous.
```

```{admonition} Definition — specificity
:class: important
**Specificity** is the fraction of truly healthy people the test correctly clears:
$\text{TN} / (\text{TN} + \text{FP})$. High specificity means few false alarms.
```

```{admonition} Definition — precision (PPV)
:class: important
**Precision**, also called **positive predictive value (PPV)**, is the fraction of
people the test *flagged* who are actually sick: $\text{TP} / (\text{TP} + \text{FP})$.
It answers the patient's question: "the test came back positive — what's the chance I
really have it?"
```

In symbols, with TP, TN, FP, FN as the four counts:

$$ \text{sensitivity} = \frac{TP}{TP + FN}, \quad
   \text{specificity} = \frac{TN}{TN + FP}, \quad
   \text{precision} = \frac{TP}{TP + FP} $$ (eq-rates)

```{code-cell} python
:label: b3-rates
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision   = tp / (tp + fp)

print(f"Sensitivity (caught {tp} of {tp + fn} cancers): {sensitivity:.1%}")
print(f"Specificity (cleared {tn} of {tn + fp} healthy): {specificity:.1%}")
print(f"Precision   (of {tp + fp} flagged, {tp} real):  {precision:.1%}")
```

These look reassuring on a test set that is roughly one-third cancer. But real
screening populations look nothing like that — and that changes everything.

## The base-rate fallacy

Here is the trap that fools doctors and patients alike. Population screening is run on
people who are *mostly healthy*. Suppose only **1 in 200** screened people actually
has the cancer. Take the very same test — same sensitivity, same specificity — and ask
the patient's question again: if your result is positive, how worried should you be?

```{admonition} Definition — base-rate fallacy
:class: important
The **base-rate fallacy** is the mistake of judging a positive result by the test's
accuracy while ignoring how *rare* the disease is to begin with. When the disease is
rare, even a small false-alarm rate produces far more false positives than true ones.
```

Let's compute it directly. We imagine screening 100,000 people at a 0.5% prevalence,
keeping our test's measured sensitivity and specificity.

```{code-cell} python
:label: b3-baserate
population = 100_000
prevalence = 0.005                              # 1 in 200 truly has the cancer

truly_sick    = population * prevalence
truly_healthy = population - truly_sick

caught       = truly_sick * sensitivity                 # true positives
false_alarms = truly_healthy * (1 - specificity)        # false positives

ppv = caught / (caught + false_alarms)
print(f"Of {population:,} people, ~{truly_sick:,.0f} truly have the cancer.")
print(f"The test flags ~{caught + false_alarms:,.0f} people as positive:")
print(f"  - {caught:,.0f} real cancers caught")
print(f"  - {false_alarms:,.0f} false alarms")
print(f"So a positive result is real only {ppv:.1%} of the time (PPV).")
```

Read that last line twice. The *same* test that looked excellent now means a positive
result is right only a small fraction of the time — most people who get bad news are
in fact healthy. Nothing about the test got worse; the **base rate** did all the
damage. This is exactly why a positive screening result leads to a confirmatory test
rather than treatment.

```{code-cell} python
:label: b3-fig
import matplotlib.pyplot as plt

prevalences = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
ppvs = []
for p in prevalences:
    sick, healthy = population * p, population * (1 - p)
    tp_p = sick * sensitivity
    fp_p = healthy * (1 - specificity)
    ppvs.append(tp_p / (tp_p + fp_p))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([p * 100 for p in prevalences], [v * 100 for v in ppvs],
        marker="o", color="#8e44ad")
ax.set_xlabel("disease prevalence (% of screened people)")
ax.set_ylabel("precision / PPV (%)")
ax.set_title("A positive result means less when disease is rarer")
fig
```

[](#b3-fig) shows the punchline as a curve: as the disease gets rarer (moving left),
the chance that a positive result is real collapses — even though the test never
changed.

## Exercises

```{admonition} Exercise B3.1 — Which mistake would you rather make?
:class: hint
For a cancer screen, would you rather tune the test toward higher **sensitivity** or
higher **specificity** if you had to sacrifice one? Justify your answer in terms of
**false negatives** and **false positives** and their human costs.
```

```{admonition} Solution
:class: dropdown
For a serious cancer most people would favour **high sensitivity**: a false negative
means a missed, untreated tumor, while a false positive "only" leads to a follow-up
test and some anxiety. The standard design is a *sensitive* screen (catch nearly
everyone) followed by a *specific* confirmatory test (weed out the false alarms). The
right balance always depends on the relative costs — there is no universally correct
answer, only a justified one.
```

```{admonition} Exercise B3.2 — Rarer still
:class: hint
Repeat the calculation in [](#b3-baserate) for a very rare disease with prevalence
`0.0005` (1 in 2,000). What happens to the PPV, and what does this imply about
screening the *general* public for rare diseases?
```

```{admonition} Solution
:class: dropdown
```python
p = 0.0005
sick, healthy = 100_000 * p, 100_000 * (1 - p)
tp_p, fp_p = sick * sensitivity, healthy * (1 - specificity)
print(f"{tp_p / (tp_p + fp_p):.1%}")
```
The PPV drops to only a few percent: almost everyone flagged is in fact healthy. This
is why blanket screening of the whole population for rare conditions can do more harm
(needless biopsies, anxiety, cost) than good, and why screening is usually targeted at
higher-risk groups where the base rate — and therefore the PPV — is higher.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **confusion matrix** splits errors into **false positives** (false alarms) and
  **false negatives** (missed cases) — costs that are rarely equal in medicine.
- **Sensitivity** asks "of the sick, how many did we catch?"; **specificity** asks "of
  the healthy, how many did we clear?"; **precision/PPV** asks "of those we flagged,
  how many are really sick?".
- The **base-rate fallacy**: when a disease is rare, even an excellent test yields
  mostly false alarms, so a positive result can be unlikely to be real.
- Choosing how to trade these off is a *values* decision, not just a technical one —
  which makes visualizing where the model draws its line, in
  [](4_b4-decision-boundaries.md), so useful.
```
