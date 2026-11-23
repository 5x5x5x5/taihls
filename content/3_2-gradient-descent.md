---
title: 'Chapter 3.2 — Rolling Downhill: How AI Learns with Gradient Descent'
short_title: '3.2 · Gradient descent'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.2 — Rolling Downhill: How AI Learns with Gradient Descent

*Week 7 · ~40 minutes · Prerequisites: [](3_1-fitting-a-line-and-loss.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **gradient descent** as rolling downhill on a loss valley without seeing the whole map;
- feel which way is downhill **without calculus** by peeking slightly left and right;
- code a 10-line gradient-descent loop and watch a bad guess walk straight to the best slope;
- diagnose what happens when the **learning rate** is too small, well-tuned, or too large.
```

## Start with a case: finding the bottom of a fog-covered bowl

Imagine standing blindfolded on the wall of a smooth, U-shaped valley (like Panel B of Chapter 3.1). You cannot see the whole valley, and checking every foot of ground would take forever. How do you still reach the bottom?

Easy! **Tap your foot one tiny inch to the right and one tiny inch to the left.**
- If the ground goes **down** to the right, take a step **right**!
- If the ground is **steep**, take a bigger step; as the ground flattens out near the bottom, take smaller steps until you settle smoothly!

That single idea — {term}`Gradient descent` — is how almost every modern AI model learns, from small medical predictors to giant language models.

## Feel the downhill slope by hand (no calculus!)

To measure how steeply the loss rises or falls at our current `slope`, we just **peek** a tiny distance `peek = 0.001` to both sides and divide the height change by the horizontal width (`rise / run` from algebra!):

$$ \text{steepness} = \frac{\text{loss}(\text{slope} + 0.001) - \text{loss}(\text{slope} - 0.001)}{2 \times 0.001} $$ (eq-32-peek)

Check the sign!
- Suppose `slope = 0.0` (way up the **left** wall of the valley). Stepping slightly right (`+0.001`) lowers the loss, so `steepness` is **negative** (say, `-150`).
- Then we update: `slope = slope - learning_rate * steepness`. Subtracting a *negative* number **adds** to `slope` — moving us **right, downhill** toward `2.2`!
- If we ever stood on the **right** wall (`slope = 4.0`), `steepness` is **positive**, so subtracting moves us **left, downhill**!

```{admonition} Definition — gradient descent, learning rate, and epoch
:class: important
- {term}`Gradient descent` minimizes loss by repeatedly feeling which way is downhill and taking a step in that direction.
- The {term}`Learning rate` sets how big each step is: too small crawls, while too big leaps right over the valley floor!
- Each pass of this update step over the data is one {term}`Epoch`.
```

## Watch gradient descent roll to the bottom!

Let's start from a terrible initial guess (`slope = 0.0`) on our exact blood-pressure data from Chapter 3.1 and watch 25 steps of gradient descent walk straight to `2.2`!

```{admonition} Predict before you run
:class: tip
Starting from `slope = 0.0`, we will test three **learning rates**: `0.001` (tiny), `0.01` (smooth), and `0.032` (huge!). Which one will walk steadily to the bottom, which will barely move in 25 steps, and which will **bounce wildly out of the valley**?
```

```{code-cell} python
:label: ch32-gd-1d
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
doses = np.linspace(0, 10, 40)
drops = 2.2 * doses + rng.normal(0, 1.8, size=len(doses))

def mse_loss(slope):
    return float(np.mean((slope * doses - drops) ** 2))

def roll_downhill(learning_rate, epochs=25, peek=0.001):
    slope = 0.0                     # terrible starting guess!
    path = [slope]
    for _ in range(epochs):
        steepness = (mse_loss(slope + peek) - mse_loss(slope - peek)) / (2 * peek)
        slope = slope - learning_rate * steepness
        path.append(slope)
    return path

smooth_path = roll_downhill(learning_rate=0.01, epochs=25)
print(f"Start (epoch  0): slope = {smooth_path[0]:.2f}, loss = {mse_loss(smooth_path[0]):6.1f}")
print(f"End   (epoch 25): slope = {smooth_path[-1]:.2f}, loss = {mse_loss(smooth_path[-1]):6.2f}")

# Plot Panel A: Footsteps rolling down the valley | Panel B: Comparing 3 learning rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

xs = np.linspace(-0.3, 4.5, 200)
ax1.plot(xs, [mse_loss(s) for s in xs], color="#009E73", lw=2.2, label="Loss valley")
ax1.plot(smooth_path, [mse_loss(s) for s in smooth_path], "o-", color="#D55E00", ms=5, lw=1.5, label="Gradient descent steps")
ax1.scatter(smooth_path[0], mse_loss(smooth_path[0]), s=90, color="#111111", zorder=5, label="Start (0.0)")
ax1.set_xlabel("Slope"); ax1.set_ylabel("Loss (MSE)")
ax1.set_title("A. Footsteps rolling to the bottom (lr = 0.01)")
ax1.legend(frameon=False, fontsize=8.5); ax1.grid(alpha=0.25)

for lr, col, name in [
    (0.001, "#E69F00", "lr = 0.001 (too slow)"),
    (0.010, "#0072B2", "lr = 0.010 (smooth!)"),
    (0.032, "#D55E00", "lr = 0.032 (explodes!)"),
]:
    p = roll_downhill(learning_rate=lr, epochs=15)
    ax2.plot([min(mse_loss(s), 400) for s in p], "o-", ms=3.5, lw=2, color=col, label=name)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (capped at 400)")
ax2.set_title("B. Learning rate controls stability")
ax2.legend(frameon=False, fontsize=8.5); ax2.grid(alpha=0.25)

fig.tight_layout()
plt.show()
```

Look at those red dots in Panel A! High up on the steep left wall, gradient descent takes huge confident leaps; as the valley flattens out near `2.2`, the steps shrink gently to a stop. And in Panel B, making the step size just ~3× too big (`0.032`) leaps clear over the valley floor and rockets upward!

:::{margin}
**What is "backpropagation"?** Peeking left and right twice per knob works effortlessly for 1 or 10 knobs. Real neural networks have millions of knobs, so deep-learning libraries like PyTorch use a fast algebra shortcut called **backpropagation** to compute those *exact same downhill steepnesses* for all million knobs at once!
:::

```{admonition} Going deeper — rolling downhill with TWO knobs (slope + intercept)
:class: dropdown
What if our line doesn't pass through zero (`predicted = w * dose + b`)? The loop is identical — we just peek on `w` AND peek on `b` in every epoch!

```python
w, b = 0.0, 0.0
for _ in range(300):
    gw = (np.mean(((w + 1e-3)*doses + b - drops)**2) - np.mean(((w - 1e-3)*doses + b - drops)**2)) / 2e-3
    gb = (np.mean((w*doses + (b + 1e-3) - drops)**2) - np.mean((w*doses + (b - 1e-3) - drops)**2)) / 2e-3
    w -= 0.02 * gw
    b -= 0.02 * gb
```
Both knobs roll down the 2D bowl together until they settle at the best line!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.2.1 (Modify) — Start on the opposite wall
:class: hint
Modify `roll_downhill` so it starts way up the **right** wall at `slope = 4.5` instead of `0.0` (keeping `learning_rate = 0.01`). Does it still roll back down to `~2.2`? What is the sign of `steepness` on the very first step?
```

```{admonition} Solution
:class: dropdown
```python
slope = 4.5
for step in range(25):
    steepness = (mse_loss(slope + 0.001) - mse_loss(slope - 0.001)) / 0.002
    if step == 0:
        print(f"First step steepness = {steepness:+.1f} (positive -> moves LEFT!)")
    slope = slope - 0.01 * steepness
print(f"Final slope from 4.5: {slope:.2f}")
```
Yes! On the right wall, `steepness` is positive (`+156`), so subtracting moves `slope` smoothly leftward back down to `~2.2`.
```

```{admonition} Exercise 3.2.2 (Make) — Find the danger boundary
:class: hint
Call `roll_downhill(learning_rate=lr, epochs=25)` for `lr` in `[0.015, 0.025, 0.029, 0.031]` and print the final slope and loss for each. Roughly where does the learning rate cross from settling down to bouncing out?
```

```{admonition} Solution
:class: dropdown
```python
for lr in [0.015, 0.025, 0.029, 0.031]:
    p = roll_downhill(learning_rate=lr, epochs=25)
    print(f"lr = {lr:.3f}: final slope = {p[-1]:8.2f}, final loss = {mse_loss(p[-1]):9.2f}")
```
Around `lr = 0.029`, steps start bouncing back and forth across the valley; by `0.031`, each bounce lands higher up the opposite wall and explodes!
```

```{admonition} Exercise 3.2.3 (Critical Reasoning) — What if a loss landscape has multiple valleys?
:class: hint
Our straight-line MSE curve had only **one** smooth bowl, so every starting point reached the same bottom. Imagine a rugged mountain range with a shallow little puddle high up and a deep canyon farther down. Why might a neural network sometimes get stuck in a mediocre spot depending on where its random starting weights began?
```

```{admonition} Solution
:class: dropdown
If gradient descent only feels the ground right under its feet, it stops at the bottom of whichever valley it rolls into first (a **local minimum**)! That is why real AI engineers try smart random starts and gentle random mini-batch jiggles to help training bounce out of shallow puddles into deeper, better valleys.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Gradient descent** finds the lowest loss by repeatedly feeling the downhill slope and taking a step — scaling to models with millions of parameters.
- You can feel downhill **without calculus** simply by peeking slightly left and right (`rise / run`).
- The **learning rate** must sit in the Goldilocks zone: too small crawls, too big bounces out of the valley.
```

*Next up:* We know how models learn their knobs. Now let's wire multiple artificial neurons together into a real **neural network** that can bend around curved biology in [](3_3-inside-a-neural-network.md)!
