---
title: 'Chapter A1 — How fast does an infection grow?'
short_title: 'A1 · Exponential growth'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter A1 — How fast does an infection grow?

<!-- Launch badges. JupyterLite (in-page) is the real zero-install path and is
     enabled site-wide in myst.yml; these badges are extra on-ramps.
     Note: the Colab badge works for .ipynb files; this book's chapters are .md
     code-cell files, so Colab is best-effort — JupyterLite or Binder is preferred. -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/5x5x5x5/taihls/HEAD)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- describe what **exponential growth** means in your own words;
- predict how a population changes when it **doubles** at a steady rate;
- run and modify a small simulation of a growing colony of cells;
- explain why "the number was small yesterday" can be dangerously misleading.
```

## Start with a story

Imagine a single bacterium lands on a warm, nutrient-rich surface — say, an
uncleaned kitchen counter. Under good conditions some bacteria can **split in two
about once an hour**. One becomes two, two become four, four become eight.

This feels harmless at first. After three hours there are only eight cells; you'd
never notice them. But this kind of growth has a sting in its tail, and seeing
*why* is the whole point of this chapter.

Let's define our one new term before we go further.

```{admonition} Definition — exponential growth
:class: important
**Exponential growth** is what happens when a quantity increases by the *same
factor* over each equal step of time (here: ×2 every hour), rather than by the
same *amount*. The bigger it gets, the faster it grows.
```

## From the story to a formula

If we start with $N_0$ cells and the population doubles every hour, then after
$t$ hours the number of cells $N(t)$ is:

$$ N(t) = N_0 \times 2^{t} $$ (eq-growth)

That's it — no calculus, just repeated multiplication. The "$2$" is the doubling
factor and the "$t$" (the exponent) is how many hours have passed. Equation
[](#eq-growth) is the engine behind everything below.

## Watch it happen in code

Let's start one cell and let it double every hour for half a day. We'll build the
list of hourly counts with a simple loop.

```{code-cell} python
:label: growth-sim
N0 = 1                 # we start with a single cell
hours = 12             # simulate half a day

counts = [N0]
for hour in range(1, hours + 1):
    counts.append(counts[-1] * 2)   # the population doubles each hour

print(f"Hour 0:  {counts[0]:>10,} cells")
print(f"Hour {hours}: {counts[-1]:>10,} cells")
```

One cell became thousands in half a day. Notice how *little* happened in the first
few hours and how *much* happened in the last few — that's the signature of
exponential growth. Let's make that visible.

```{code-cell} python
:label: growth-fig
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(hours + 1), counts, marker="o", color="#c0392b")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Number of cells")
ax.set_title("One cell, doubling every hour")
fig
```

The curve in [](#growth-fig) is nearly flat, then shoots upward. This is why an
infection (or a viral video, or compound interest) can seem like "nothing is
happening" right up until it's everywhere. The early small numbers hid how fast
the *rate* was building.

```{tip}
Try it yourself: in the cell labelled [](#growth-sim), change `N0 = 1` to
`N0 = 10` and re-run. Every count is 10× bigger — but the *shape* of the curve is
identical. Exponential growth doesn't care where you start.
```

## Exercise

```{admonition} Exercise A1.1 — Antibiotics fight back
:class: hint
Suppose a doctor starts an antibiotic that **kills half the cells every hour**,
while the bacteria keep trying to double. Starting from `N0 = 1000`, what happens
to the population over 12 hours? Write the loop, then describe the result in one
sentence.

*Hint: doubling is `× 2`; killing half is `× 0.5`. What is `2 × 0.5`?*
```

```{admonition} Solution
:class: dropdown
Doubling and then halving multiplies the count by `2 × 0.5 = 1`, so the population
stays flat:

```python
N0 = 1000
counts = [N0]
for hour in range(1, 13):
    counts.append(counts[-1] * 2 * 0.5)   # double, then lose half
print(counts[-1])   # → 1000.0, unchanged
```

**In one sentence:** when the kill rate exactly matches the growth rate, the
infection neither grows nor shrinks — and a *slightly* stronger drug (say, killing
60% per hour) would tip the balance toward shrinking.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- **Exponential growth** multiplies by a fixed factor each step; small beginnings
  explode surprisingly fast.
- The model is just $N(t) = N_0 \times 2^{t}$ — repeated multiplication, no
  advanced math.
- A loop in a few lines of Python reproduces and visualizes the whole story.
- Whether a population grows or shrinks depends on the *balance* of competing
  rates (growth vs. death) — a theme we'll meet again throughout the book.
```
