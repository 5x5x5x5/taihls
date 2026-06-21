---
title: 'Chapter C1 — Fitting the best line'
short_title: 'C1 · Fitting a line'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter C1 — Fitting the best line

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **model**, a **parameter**, and a **loss** are, in your own words;
- describe **mean squared error** as a single number that says "how wrong" a line is;
- fit a line to data by **trying many slopes and keeping the best** one;
- read a loss-vs-slope curve and find the bottom of the valley.
```

## Start with a story

A pharmacologist gives volunteers different **doses** of a new drug and measures each
person's **response** — say, how much their blood pressure drops. They suspect that
bigger doses give bigger responses, roughly along a straight line. But the data are
messy: people differ, measurements wobble. They want one clean line through the cloud
of points that captures the trend, so they can predict the response at a dose nobody
tried yet.

How does a computer find that "best" line? The surprising answer is that it does almost
exactly what you would do by eye — try a line, ask "how wrong is it?", nudge it, and
keep the version that's least wrong. To make that precise we need three words.

```{admonition} Definition — model
:class: important
A **model** is a rule with adjustable knobs that turns an input (a dose) into a
prediction (a response). Here our model is a straight line through the origin:
prediction $=$ slope $\times$ dose.
```

```{admonition} Definition — parameter
:class: important
A **parameter** is one of the adjustable knobs inside a model. Our line has a single
parameter, its **slope** — how steeply the response rises as the dose grows. Choosing a
model means choosing values for its parameters.
```

## How wrong is a line? The idea of loss

Suppose we pick a slope and draw the line. For each volunteer we can compare the line's
prediction to what actually happened. The gap between them is the **error** for that
person. A good line makes all those gaps small at once.

To turn "all the gaps" into a single score we **square** each gap (so positive and
negative errors don't cancel, and big misses are penalized hard) and take the average.
That average is the **loss**.

```{admonition} Definition — loss
:class: important
A **loss** is a single number that measures how badly a model fits the data: the lower,
the better. Fitting a model means searching for the parameters that make the loss as
small as possible.
```

```{admonition} Definition — mean squared error (MSE)
:class: note
The **mean squared error** is one common loss: for each data point, take the difference
between the prediction and the true value, square it, and average over all points.
"Squared" punishes large misses; "mean" keeps the score from growing just because we
have more data.
```

In algebra, if point $i$ has dose $x_i$ and true response $y_i$, and our line predicts
$\text{slope} \times x_i$, the loss over $n$ points is

$$ \text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\bigl(\text{slope}\times x_i - y_i\bigr)^2 . $$ (eq-mse)

No calculus — just subtract, square, and average. Equation [](#eq-mse) is the entire
scoreboard for the rest of this chapter.

## Make some dose–response data

Let's invent a tidy little dataset: a true underlying slope, plus the kind of random
wobble real measurements always carry. Seeding the random numbers means everyone gets
the same picture.

```{code-cell} python
:label: c1-make-data
import numpy as np

np.random.seed(0)
true_slope = 2.0                      # the hidden truth we hope to recover
doses = np.linspace(0, 10, 40)        # 40 doses from 0 to 10 mg
noise = np.random.normal(0, 2.0, size=doses.shape)
responses = true_slope * doses + noise   # response = 2 x dose, plus wobble

print(f"{len(doses)} volunteers, doses {doses.min():.0f}–{doses.max():.0f} mg")
print(f"First three responses: {responses[:3].round(2)}")
```

## Scoring one line

Before searching, let's write the loss from [](#eq-mse) as a tiny function and test it
on a deliberately bad guess.

```{code-cell} python
:label: loss-fn
def mse(slope, x, y):
    predictions = slope * x
    return np.mean((predictions - y) ** 2)   # average squared gap

print(f"Loss if slope = 0.5 (too shallow): {mse(0.5, doses, responses):.2f}")
print(f"Loss if slope = 2.0 (the truth):   {mse(2.0, doses, responses):.2f}")
```

The truthful slope already scores far lower. That single number is all our search needs.

## Fitting by trying many slopes

Here is the whole idea, stripped bare: sweep a range of candidate slopes, score each one
with the loss, and keep the slope with the smallest loss. No fancy math — just patience
and a loop.

```{code-cell} python
:label: brute-force
candidate_slopes = np.linspace(0, 4, 401)     # try 401 slopes from 0 to 4
losses = np.array([mse(s, doses, responses) for s in candidate_slopes])

best_index = np.argmin(losses)                # where is the loss smallest?
best_slope = candidate_slopes[best_index]
print(f"Best slope found: {best_slope:.2f}  (hidden truth was {true_slope})")
print(f"Lowest loss: {losses[best_index]:.2f}")
```

We recovered a slope very close to the true `2.0`, just by checking which candidate was
least wrong. The data's wobble means we won't land exactly on `2.0`, and that's honest —
the best line for *this* noisy sample isn't quite the hidden truth.

## Seeing the fit and the valley

Two pictures tell the story. On the left, the data with our best line laid over it. On
the right, the loss for every candidate slope — a **valley** whose lowest point is the
slope we chose.

```{code-cell} python
:label: fit-fig
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: data + best-fit line.
ax1.scatter(doses, responses, s=18, color="#2980b9", alpha=0.7, label="volunteers")
ax1.plot(doses, best_slope * doses, color="#c0392b", lw=2,
         label=f"best line (slope {best_slope:.2f})")
ax1.set_xlabel("dose (mg)")
ax1.set_ylabel("response (mmHg drop)")
ax1.set_title("Data and the best-fit line")
ax1.legend()

# Right: the loss-vs-slope valley.
ax2.plot(candidate_slopes, losses, color="#16a085")
ax2.scatter(best_slope, losses[best_index], s=90, color="#c0392b", zorder=5,
            label="lowest loss")
ax2.set_xlabel("candidate slope")
ax2.set_ylabel("loss (MSE)")
ax2.set_title("How wrong is each slope?")
ax2.legend()

fig
```

In the right-hand panel of [](#fit-fig) the loss is huge for slopes that are too small or
too large, and dips to a single low point in between. Fitting a model is nothing more
than finding the bottom of that valley. Brute force works fine here because we have only
**one** parameter; in [](4_c4-how-models-learn.md) we'll meet a smarter way to roll
downhill that scales to models with millions of knobs.

## Exercises

```{admonition} Exercise C1.1 — A finer search
:class: hint
Our search stepped through slopes in jumps of `0.01`. Re-run the brute-force search with
`np.linspace(0, 4, 41)` (steps of `0.1`) instead. Does the best slope change much? What
does this tell you about how fine your search needs to be?
```

```{admonition} Solution
:class: dropdown
```python
coarse = np.linspace(0, 4, 41)
coarse_losses = np.array([mse(s, doses, responses) for s in coarse])
print(coarse[np.argmin(coarse_losses)])   # → 2.0
```
The coarse search lands on `2.0`, essentially the same answer. The valley is wide and
smooth near the bottom, so a rough search gets you almost there; a finer search only
polishes the last decimal. Knowing how fine to search is a recurring practical question.
```

```{admonition} Exercise C1.2 — More noise, fuzzier answer
:class: hint
Change the noise level from `2.0` to `6.0` in the data-making cell and re-run everything.
Does the best slope still come out near `2.0`? What happens to the lowest loss, and why?
```

```{admonition} Solution
:class: dropdown
With noisier data the best slope still hovers around `2.0` but drifts a little further
from it on any given run, and the **lowest loss rises** — because even the perfect line
can't pass through points that have been scattered more widely. Loss measures leftover
wobble we can't explain, not just a bad choice of slope. More noise means a fuzzier
estimate of the underlying trend.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **model** is a rule with adjustable **parameters**; our line had one, its slope.
- A **loss** scores how wrong a model is in a single number; **mean squared error**
  averages the squared gaps between predictions and truth.
- We can fit a one-parameter model by **trying many values and keeping the lowest loss**.
- The loss-vs-parameter curve is a **valley**; fitting means finding its bottom — the
  central idea behind the learning we'll do for the rest of Part C.
```
