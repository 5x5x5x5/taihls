---
title: 'Chapter C4 — How models learn'
short_title: 'C4 · Gradient descent'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter C4 — How models learn

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **gradient descent** as rolling downhill on a loss surface;
- define **learning rate** and **epoch** in plain words;
- watch a parameter step its way to the bottom of a loss valley in code;
- recognize **overfitting** by comparing training and validation error as a model grows.
```

## Start with a story

In [](1_c1-fitting-a-line.md) we fit a line by *trying every slope* and keeping the lowest
loss. That brute-force sweep works for one knob. But a neural network like the one in
[](2_c2-what-is-a-neural-net.md) can have thousands or millions of knobs — you could never
try every combination. We need a way to find the bottom of the loss valley **without
checking every point in it**.

The trick is the one a ball uses to find the bottom of a bowl: it doesn't survey the whole
bowl, it just rolls downhill from wherever it happens to be. That is the single idea behind
how almost every modern AI model learns.

## Rolling downhill

```{admonition} Definition — gradient descent
:class: important
**Gradient descent** is a way to minimize a loss by repeatedly taking a small step in the
direction that goes *downhill* on the loss curve. Stand on the slope, feel which way is down,
take a step, and repeat. You don't need to see the whole valley — only which way is down
right here.
```

How do we tell which way is downhill without calculus? We simply **peek**: nudge the
parameter a tiny bit and see whether the loss went up or down. If stepping right lowered the
loss, the valley is to the right, so we move right; if it raised the loss, we move left. That
"check both sides and lean toward the lower one" is gradient descent in spirit, and it needs
nothing more than evaluating the loss we already built.

```{admonition} Definition — learning rate
:class: important
The **learning rate** is how big a step we take each time. Too small and learning crawls;
too big and we leap clear over the bottom and bounce around — or fly out of the valley
entirely. Choosing it well is one of the most important practical decisions in training.
```

```{admonition} Definition — epoch
:class: note
An **epoch** is one full pass of the learning process over all the training data. Training
usually takes many epochs — many downhill steps — before the loss settles near the bottom.
```

## Rebuild the loss valley

Let's reuse the dose–response setup from [](1_c1-fitting-a-line.md): one parameter (the
slope) and the same mean-squared-error loss. This gives us a clean valley to roll down.

```{code-cell} python
:label: rebuild-loss
import numpy as np

np.random.seed(0)
true_slope = 2.0
doses = np.linspace(0, 10, 40)
responses = true_slope * doses + np.random.normal(0, 2.0, size=doses.shape)

def loss(slope):
    return np.mean((slope * doses - responses) ** 2)   # mean squared error

print(f"Loss at slope 0.0: {loss(0.0):.1f}")
print(f"Loss at slope 2.0: {loss(2.0):.1f}")
```

## Stepping downhill, no calculus

Here is gradient descent by peeking. From a deliberately bad starting slope, we measure the
loss a hair to the left and a hair to the right, work out which way is downhill, and take a
step proportional to how steep the slope is. We record every stop along the way.

```{code-cell} python
:label: descent
slope = 0.0              # start far from the answer
learning_rate = 0.01     # how big each step is
peek = 1e-3              # tiny nudge used to feel the downhill direction

path = [slope]
for epoch in range(60):
    # Feel the slope of the loss: how much does loss change per unit of slope?
    downhill = (loss(slope + peek) - loss(slope - peek)) / (2 * peek)
    slope = slope - learning_rate * downhill   # step downhill
    path.append(slope)

print(f"Started at slope 0.00, loss {loss(0.0):.1f}")
print(f"Ended at slope {slope:.2f}, loss {loss(slope):.1f}")
print(f"(The hidden truth was {true_slope}.)")
```

Starting from a useless slope of `0`, the parameter walked itself almost exactly to the true
value — never once trying every slope, only ever stepping downhill from where it stood. The
steps are big when the valley wall is steep and shrink as the ground flattens near the
bottom, so it slows down and settles instead of overshooting.

## Watch the descent path

Let's lay the descent steps on top of the loss valley so you can see the ball rolling in.

```{code-cell} python
:label: descent-fig
import matplotlib.pyplot as plt

slopes = np.linspace(0, 4, 200)
losses = [loss(s) for s in slopes]
path = np.array(path)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(slopes, losses, color="#16a085", label="loss valley")
ax.plot(path, [loss(s) for s in path], "o-", color="#c0392b",
        markersize=4, lw=1, label="descent steps")
ax.scatter(path[0], loss(path[0]), s=90, color="black", zorder=5, label="start")
ax.set_xlabel("slope (the parameter)")
ax.set_ylabel("loss (MSE)")
ax.set_title("Gradient descent rolls to the bottom")
ax.legend()
fig
```

The red dots in [](#descent-fig) crowd together near the bottom: far up the wall the loss is
steep, so the steps are long; as the ground levels off the steps shrink and the parameter
eases into the lowest point. That is gradient descent — and it works just the same with a
million parameters, each feeling its own downhill direction at once.

```{tip}
Try changing `learning_rate = 0.01` to `0.05` (bigger steps) and re-running. The descent
reaches the bottom in far fewer steps. Now try `0.3` — too greedy, and the steps overshoot
and bounce across the valley instead of settling. The learning rate is a balancing act.
```

## When learning goes too far: overfitting

Gradient descent makes the **training** loss small. But small training loss is not the goal —
the goal is to do well on **new** data the model has never seen. A model with too many knobs
can drive its training loss to almost nothing by memorizing the quirks and noise of the
training set, then fail on anything new. That trap is **overfitting**.

```{admonition} Definition — overfitting
:class: important
**Overfitting** is when a model fits its training data so closely that it captures the random
noise as well as the real pattern, and so performs worse on fresh data. It has memorized the
answers instead of learning the lesson.
```

To see it, we fit polynomial curves of growing flexibility to a small wiggly dataset, and
track the error on a **held-out validation set** the model never trained on.

```{code-cell} python
:label: overfit
from numpy.polynomial import polynomial as P

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-3, 3, size=30))
true_curve = np.sin(x)                      # the real pattern
data = true_curve + rng.normal(0, 0.3, size=x.shape)   # noisy observations

# Split into training and validation halves.
x_tr, y_tr = x[::2], data[::2]
x_val, y_val = x[1::2], data[1::2]

degrees = range(1, 16)                       # model complexity: line up to wiggly curve
train_err, val_err = [], []
for d in degrees:
    coeffs = np.polyfit(x_tr, y_tr, d)       # fit a curve of degree d
    train_err.append(np.mean((np.polyval(coeffs, x_tr) - y_tr) ** 2))
    val_err.append(np.mean((np.polyval(coeffs, x_val) - y_val) ** 2))

best = degrees[int(np.argmin(val_err))]
print(f"Lowest validation error at degree {best}.")
print(f"Training error keeps falling: {train_err[0]:.2f} → {train_err[-1]:.4f}")
```

The training error keeps dropping as we add flexibility — the model fits the training points
ever more tightly. But the validation error tells the real story.

```{code-cell} python
:label: overfit-fig
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(list(degrees), train_err, "o-", color="#2980b9", label="training error")
ax.plot(list(degrees), val_err, "o-", color="#c0392b", label="validation error")
ax.axvline(best, color="gray", ls="--", lw=1, label=f"best (degree {best})")
ax.set_xlabel("model complexity (polynomial degree)")
ax.set_ylabel("error (MSE)")
ax.set_title("Overfitting: training keeps improving, validation doesn't")
ax.legend()
ax.set_ylim(0, max(val_err[:8]) * 1.2)
fig
```

In [](#overfit-fig) the two curves part ways. Training error slides toward zero, but
validation error bottoms out at a modest complexity and then **climbs** as the model starts
memorizing noise. The best model is the one that does best on data it hasn't seen — not the
one that fits the training set hardest. Spotting and avoiding this gap is one of the most
important skills in all of machine learning.

## Exercises

```{admonition} Exercise C4.1 — A learning rate too big
:class: hint
In the `descent` cell, set `learning_rate = 0.5` and re-run, printing `path[:8]`. Describe
what the slope values do. Why does too large a learning rate fail to find the bottom?
```

```{admonition} Solution
:class: dropdown
With a learning rate of `0.5` the steps are so large that each one overshoots the bottom and
lands further up the *opposite* wall; the slope values swing back and forth with growing
size and the loss explodes instead of shrinking. A step proportional to a steep slope, made
too big, leaps clear over the valley — which is exactly why a runaway learning rate is one of
the first things to check when training "blows up."
```

```{admonition} Exercise C4.2 — Where would you stop?
:class: hint
Looking at the overfitting plot, suppose you could only pick one polynomial degree to deploy
on future patients. Which would you choose, and why is it *not* the degree with the lowest
training error?
```

```{admonition} Solution
:class: dropdown
You would pick the degree with the lowest **validation** error (around degree 3–5), not the
highest degree. The high-degree model has the lowest *training* error but the worst
validation error — it has memorized the noise. On future patients you only ever see "new"
data, so the validation score is the honest estimate of how the model will actually perform.
Choosing complexity this way is the heart of the train/validation discipline introduced in
[](2_b2-train-test.md).
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- **Gradient descent** finds the bottom of a loss valley by repeatedly stepping downhill —
  no need to try every parameter value, which is what makes huge models trainable.
- We found "downhill" simply by **peeking**: nudge the parameter and see which way lowers the
  loss; no calculus required.
- The **learning rate** sets the step size — too small is slow, too big overshoots — and an
  **epoch** is one full downhill pass over the data.
- Driving training loss to zero invites **overfitting**; the model that wins is the one with
  the lowest error on **held-out** data, not the tightest fit to what it has already seen.
```
