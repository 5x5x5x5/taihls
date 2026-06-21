---
title: 'Chapter E1 — When AI gets it wrong for some people'
short_title: 'E1 · Bias in medical AI'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter E1 — When AI gets it wrong for some people

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what **algorithmic bias** means and why a single accuracy number can hide it;
- name two common sources of bias: **representation/sampling bias** and **distribution shift**;
- build a small simulation where a model is accurate for one group and unreliable for another;
- measure the **per-group accuracy gap** and show how fixing the data shrinks it.
```

## Start with a story

A hospital buys an AI tool that screens patients for an at-risk condition. The
vendor's brochure says it is **92% accurate** — better than the busy clinic could
manage by hand. The hospital switches it on.

Months later, a nurse notices something troubling. The tool seems to catch
at-risk patients in one group of people just fine, but keeps clearing patients in
another group who later turn out to be sick. The overall accuracy on the brochure
was real — but it was an *average*, and the average quietly hid that the tool was
working well for some people and badly for others.

This is not a made-up worry. A widely cited study found that a real algorithm used
on millions of patients systematically under-estimated the needs of Black patients
relative to equally sick white patients [@obermeyer2019bias]. And skin-cancer
classifiers trained mostly on lighter skin can perform worse on darker skin, simply
because darker skin was scarce in the training images [@esteva2017skin]. Let's give
the problem a name.

```{admonition} Definition — algorithmic bias
:class: important
**Algorithmic bias** is when a model's mistakes fall *unequally* on different groups
of people — it is more accurate, or more useful, for some groups than for others.
A high *overall* score can coexist with poor performance for a specific group.
```

## Where does the bias come from?

A model only knows what its training data showed it. Two problems with that data
account for a huge share of real-world bias.

```{admonition} Definition — representation (sampling) bias
:class: important
**Representation bias** (also called **sampling bias**) happens when some group is
**under-represented** in the training data. The model sees too few examples of that
group to learn its pattern well, so it leans on the majority's pattern instead.
```

```{admonition} Definition — distribution shift
:class: important
**Distribution shift** happens when the data a model is *used* on differs from the
data it was *trained* on — for example, the relationship between a measurement and
the outcome is not the same in every group, or changes over time. What the model
learned no longer fits.
```

In our simulation below, **both** problems appear at once: one group is rare in the
training data *and* the signal that predicts risk shows up in a different
measurement for that group. That combination is exactly what makes bias so easy to
miss.

## Build the data and train a model

We'll invent two patient groups, **A** and **B**. Each patient has two screening
measurements (think of two different lab tests). In **group A** the risk signal
lives in test 0; in **group B** it lives in test 1 instead. Crucially, our training
set is **90% group A and only 10% group B** — group B is under-represented.

```{code-cell} python
:label: e1-data
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
rng = np.random.default_rng(0)

def make_patients(n, signal_feature):
    """n patients, half at-risk (1), half healthy (0).
    The risk signal lives in `signal_feature` (test 0 or test 1);
    the other test is just noise."""
    n_pos = n // 2
    X = rng.normal(0, 1.0, size=(n, 2))         # two screening tests
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    X[:n_pos, signal_feature] += 1.5            # at-risk: signal test runs high
    X[n_pos:, signal_feature] -= 1.5            # healthy: signal test runs low
    return X, y

# Training set: lots of group A, very little group B.
XA_tr, yA_tr = make_patients(900, signal_feature=0)   # group A: signal in test 0
XB_tr, yB_tr = make_patients(100, signal_feature=1)   # group B: signal in test 1
X_tr = np.vstack([XA_tr, XB_tr])
y_tr = np.concatenate([yA_tr, yB_tr])

model = LogisticRegression().fit(X_tr, y_tr)
print(f"Trained on {len(yA_tr)} group-A and {len(yB_tr)} group-B patients.")
```

The model learns whatever pattern dominates the training data — and that pattern
comes overwhelmingly from group A.

## Measure the gap

Now we test the model on a **balanced** set: 500 fresh patients from each group.
The trick is to *not* stop at the overall score. We compute accuracy **separately
for each group**.

```{admonition} Definition — per-group accuracy gap
:class: important
The **per-group accuracy gap** is the difference between a model's accuracy on one
group and its accuracy on another. A gap near zero means the model treats both
groups equally well; a large gap is the fingerprint of algorithmic bias.
```

```{code-cell} python
:label: e1-gap
XA_te, yA_te = make_patients(500, signal_feature=0)
XB_te, yB_te = make_patients(500, signal_feature=1)

accA = model.score(XA_te, yA_te)
accB = model.score(XB_te, yB_te)
overall = model.score(np.vstack([XA_te, XB_te]),
                      np.concatenate([yA_te, yB_te]))

print(f"Overall accuracy : {overall:.1%}")
print(f"Group A accuracy : {accA:.1%}")
print(f"Group B accuracy : {accB:.1%}")
print(f"Accuracy gap     : {accA - accB:.1%}")
```

Look at what the overall number hid. The model is right about **93%** of the time
for group A but only **70%** for group B — a **23-point gap**. A clinician seeing
only the "overall accuracy" would never suspect that group B is being failed.

## See it

A picture makes the gap unmistakable.

```{code-cell} python
:label: e1-fig
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
groups = ["Overall", "Group A", "Group B"]
scores = [overall, accA, accB]
colors = ["#7f8c8d", "#2980b9", "#c0392b"]
bars = ax.bar(groups, scores, color=colors)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="coin-flip (50%)")
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("One model, very different experiences")
for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.0%}",
            ha="center", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
fig
```

In [](#e1-fig) the gray "Overall" bar looks reassuring, the blue group-A bar looks
great, and the red group-B bar tells the real story. The average was never wrong —
it was just the wrong question.

## Can we fix it?

The bias here was driven by **representation bias**: group B was too rare in
training. So let's give the model a *balanced* diet — 500 patients from each group —
and re-test on the very same test set.

```{code-cell} python
:label: e1-fix
XA_b, yA_b = make_patients(500, signal_feature=0)
XB_b, yB_b = make_patients(500, signal_feature=1)
X_b = np.vstack([XA_b, XB_b])
y_b = np.concatenate([yA_b, yB_b])

fair_model = LogisticRegression().fit(X_b, y_b)
accA2 = fair_model.score(XA_te, yA_te)
accB2 = fair_model.score(XB_te, yB_te)
print(f"Group A accuracy : {accA2:.1%}")
print(f"Group B accuracy : {accB2:.1%}")
print(f"Accuracy gap     : {accA2 - accB2:.1%}")
```

With balanced data the gap nearly vanishes (group A around **84%**, group B around
**88%** — a gap of only a few points, and the model is now *slightly better* for
the group it used to fail). Group A's score even dropped a little: the unbalanced
model had been spending all its effort pleasing the majority. Fairness sometimes
means trading a sliver of majority performance for a fairer deal overall — a
genuine decision, not a free lunch.

```{tip}
Auditing for bias is a habit, not a one-time check. Whenever you see a single
accuracy figure, ask: *accuracy for whom?* Splitting the score by group — as in
[](#e1-gap) — is often all it takes to surface a problem the average was hiding.
```

## Exercises

```{admonition} Exercise E1.1 — Make the imbalance worse
:class: hint
In [](#e1-data), change the group-B training size from `100` to `20` (and group A to
`980`, keeping the total at 1000). Re-run the gap calculation in [](#e1-gap). Does
the gap grow, shrink, or stay the same? Why?
```

```{admonition} Solution
:class: dropdown
```python
XA_tr, yA_tr = make_patients(980, signal_feature=0)
XB_tr, yB_tr = make_patients(20,  signal_feature=1)
model = LogisticRegression().fit(np.vstack([XA_tr, XB_tr]),
                                 np.concatenate([yA_tr, yB_tr]))
print(f"Group B accuracy: {model.score(XB_te, yB_te):.1%}")
```

Group B accuracy falls even further (toward a coin flip). With fewer group-B
examples, the model has even less reason to learn group B's pattern, so it leans
harder on group A's — a textbook case of **representation bias** getting worse as
representation gets thinner.
```

```{admonition} Exercise E1.2 — Same numbers, different signal
:class: hint
Suppose group B's signal lived in **test 0**, the *same* place as group A's
(change `signal_feature=1` to `signal_feature=0` for group B everywhere). Predict
what happens to the gap *before* you run it, then check. What does this tell you
about which kind of bias is doing the damage?
```

```{admonition} Solution
:class: dropdown
With both groups using the same signal test, there is no **distribution shift**
between them — only the representation imbalance remains. The model trained mostly
on group A now generalizes to group B almost perfectly, and the gap nearly
disappears even *without* rebalancing. This isolates the lesson: in our original
setup it was the **combination** of under-representation *and* a group-specific
signal that produced the large gap. Real systems often suffer from both at once,
which is what makes them so hard to debug from an average alone.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- **Algorithmic bias** means a model's errors land unequally across groups; a strong
  *overall* score can hide a group it serves badly.
- Two big causes are **representation (sampling) bias** — a group is too rare in the
  training data — and **distribution shift** — the pattern differs between training
  and use.
- The fix starts with *measurement*: always split accuracy by group to expose the
  **per-group gap**, never trust the average alone.
- Improving fairness (here, by balancing the data) can shrink the gap dramatically,
  but may cost a little majority-group performance — a real trade-off to decide
  openly. We'll keep humans in that decision loop in [](3_e3-humans-in-the-loop.md).
```
