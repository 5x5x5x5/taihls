---
title: 'Chapter A4 — Reading a medical statistic'
short_title: 'A4 · Reading statistics'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter A4 — Reading a medical statistic

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **base rate** is and why it changes the meaning of a test result;
- use **natural frequencies** to work out how many positive results are *real*;
- define **false positive**, **sensitivity**, and **specificity** in plain words;
- spot a misleading chart drawn with a **truncated axis**.
```

## Start with a story

You take a screening test for a rare disease. The test is described as "90% accurate."
It comes back **positive**. How worried should you be?

Most people — including, studies show, many doctors — answer "about 90% likely to be
sick." The real answer can be astonishingly different: with the numbers below, a
positive result means you are only about **8%** likely to actually have the disease.
The reason is one idea that almost everyone forgets, and this chapter is about not
forgetting it.

```{admonition} Definition — base rate
:class: important
The **base rate** (or **prevalence**) of a condition is how common it is *before* any
test — the fraction of people who have it to begin with. A rare disease has a low base
rate, and that single fact can overwhelm even a good test.
```

## Counting people, not probabilities

The cleanest way to think about tests is to forget percentages and instead imagine a
crowd of real people. This trick is called using **natural frequencies**: just count.

Let's set up our example test. We'll define its two quality numbers as we go.

```{admonition} Definition — sensitivity & specificity
:class: important
**Sensitivity** is the fraction of *sick* people the test correctly flags as positive.
**Specificity** is the fraction of *healthy* people the test correctly clears as
negative. A **false positive** is a healthy person the test wrongly flags as positive.
(We go much deeper on these in [](3_b3-sensitivity-specificity.md).)
```

```{code-cell} python
:label: natural-frequencies
import numpy as np

people = 10_000        # imagine a town of 10,000
prevalence = 0.01      # base rate: 1% truly have the disease
sensitivity = 0.90     # catches 90% of sick people
specificity = 0.90     # correctly clears 90% of healthy people

sick = round(people * prevalence)          # truly sick
healthy = people - sick                    # truly healthy

true_positives = round(sick * sensitivity)        # sick AND test positive
false_positives = round(healthy * (1 - specificity))  # healthy BUT test positive

print(f"Out of {people:,} people:")
print(f"  truly sick:    {sick:>5}")
print(f"  truly healthy: {healthy:>5}")
print(f"  test positives from the sick (true positives):    {true_positives:>4}")
print(f"  test positives from the healthy (false positives): {false_positives:>4}")
```

Stare at those last two lines. Among the sick, the test correctly flags **90** people.
But among the *much larger* healthy crowd, even a small 10% error rate produces **990**
false alarms. The healthy group is so big that its handful-of-percent mistakes
outnumber all the true cases nearly eleven to one.

## So what does a positive actually mean?

Of everyone who tests positive, the share who are *truly sick* is just the true
positives divided by all the positives.

```{code-cell} python
:label: ppv
all_positives = true_positives + false_positives
chance_really_sick = true_positives / all_positives

print(f"total positive results: {all_positives}")
print(f"chance a positive person is truly sick: {chance_really_sick:.1%}")
```

A positive result means only about an **8%** chance of really being sick — not 90%.
Nothing is wrong with the test; the **base rate** is doing the damage. When a disease
is rare, most positive results are false alarms, simply because there are so many more
healthy people to misclassify. This is why a single positive screen usually leads to a
*second*, more specific test rather than straight to treatment.

```{tip}
The next time you read "this test is 95% accurate," ask the missing question: *how
common is the condition?* Without the base rate, an accuracy number tells you almost
nothing about what a positive result means.
```

## A second way to mislead: the chart

Numbers can deceive; so can pictures. One of the most common tricks is the **truncated
axis** — starting the vertical axis somewhere above zero so that tiny differences look
enormous. Let's draw the *same two numbers* twice.

```{code-cell} python
:label: truncated-axis
import matplotlib.pyplot as plt

clinics = ["Clinic A", "Clinic B"]
recovery = [81, 84]   # % of patients who recovered — a genuine but small gap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# LEFT: honest axis, starting at 0.
ax1.bar(clinics, recovery, color="#2980b9")
ax1.set_ylim(0, 100)
ax1.set_ylabel("recovery rate (%)")
ax1.set_title("Honest: axis starts at 0")

# RIGHT: truncated axis, starting at 80 — same data, dramatic look.
ax2.bar(clinics, recovery, color="#c0392b")
ax2.set_ylim(80, 85)
ax2.set_title("Misleading: axis starts at 80")

fig
```

Both panels of [](#truncated-axis) plot the *identical* numbers — 81% and 84%. On the
left the bars are nearly the same height, honestly showing a small 3-point gap. On the
right, with the axis chopped to start at 80, Clinic B's bar towers over Clinic A's,
screaming a difference that barely exists. Whenever a bar chart impresses you, the
first thing to check is **where the axis starts**.

## Exercise

```{admonition} Exercise A4.1 — When the disease is common
:class: hint
Re-run the natural-frequencies calculation, but for a *common* condition with a base
rate of **30%** (`prevalence = 0.30`), keeping sensitivity and specificity at 90%.
What is the chance now that a positive person is truly sick? Why did it change so much?
```

```{admonition} Solution
:class: dropdown
```python
people, prevalence, sensitivity, specificity = 10_000, 0.30, 0.90, 0.90
sick = round(people * prevalence)
healthy = people - sick
true_positives = round(sick * sensitivity)
false_positives = round(healthy * (1 - specificity))
chance = true_positives / (true_positives + false_positives)
print(f"chance a positive person is truly sick: {chance:.1%}")
```

Now the chance is about **79%** — far higher than the 8% we got before, even though the
test itself is unchanged. The only difference is the **base rate**: when the condition
is common, the sick group is large and the true positives easily outnumber the false
alarms. The exact same test means very different things in a rare disease versus a
common one.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A test result is meaningless without the **base rate** — how common the condition is
  to begin with.
- For a **rare** disease, **false positives** can swamp true positives, so a positive
  result may still mean a low chance of really being sick.
- Counting **natural frequencies** ("out of 10,000 people…") makes this clear far more
  reliably than juggling percentages.
- Charts mislead too: a **truncated axis** can blow a tiny difference up into a dramatic
  one. Always check where the axis starts.
```
