---
title: 'Chapter 3.1 — Measuring Mistakes: Fitting a Line with Loss'
short_title: '3.1 · Fitting a line & loss'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.1 — Measuring Mistakes: Fitting a Line with Loss

*Week 7 · ~35 minutes · Prerequisites: Unit 2*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **model**, a **parameter**, and a **loss function** are in plain language;
- calculate **Mean Squared Error (MSE)** by hand on three volunteers before writing code;
- fit a line to drug dose–response data by scoring many candidate slopes;
- read a U-shaped **loss valley** and find its lowest point.
```

## Start with a case: how much does a blood-pressure medicine help?

A pharmacy research team tests a new blood-pressure medicine on 40 adult volunteers. Each person takes a daily dose between `0` and `10 mg`, and doctors measure how many points their blood pressure drops (measured in **mmHg**, the standard number on a doctor's arm cuff).

People vary naturally, so the dots form a fuzzy diagonal cloud rather than a razor-sharp line. We want one simple line through the cloud:

$$ \text{predicted drop} = \text{slope} \times \text{dose} $$ (eq-31-line)

Here our rule has just **one adjustable knob** ({term}`Parameter`): the **slope** (how many blood-pressure points drop for each additional milligram). How does a computer decide which slope is "best"?

It gives every candidate slope a **scoreboard** called a {term}`Loss function` — a single number measuring *how wrong* the line is (smaller = better!).

## Calculate Mean Squared Error (MSE) by hand first!

Suppose just **3 volunteers** took doses `1 mg`, `2 mg`, and `3 mg`, and their real blood-pressure drops were **`3`**, **`5`**, and **`6`** mmHg. Let's test a guess of **`slope = 2.0`** by hand:

| Volunteer | Dose | Predicted (`2 × dose`) | Actual drop | Error (`pred − actual`) | Squared error |
|---|---|---|---|---|---|
| #1 | 1 mg | 2 | 3 | `−1` | $(-1)^2 = \mathbf{1}$ |
| #2 | 2 mg | 4 | 5 | `−1` | $(-1)^2 = \mathbf{1}$ |
| #3 | 3 mg | 6 | 6 | `0` | $0^2 = \mathbf{0}$ |

Now take the **average** of the squared errors:

$$ \text{MSE} = \frac{1 + 1 + 0}{3} \approx \mathbf{0.67} $$

Why do we **square** each error before averaging? For two simple reasons:
1. Minus signs disappear (`-1` and `+1` can't cancel each other out and pretend the error was zero!).
2. Huge misses get punished much harder than tiny misses (`4² = 16`, whereas `1² = 1`).

```{admonition} Definition — parameter, loss function, and MSE
:class: important
- A {term}`Parameter` is an adjustable number inside a model (here, the line's slope).
- A {term}`Loss function` compresses all prediction mistakes into one "how-wrong" score (lower is better).
- {term}`Mean squared error (MSE)` is the average of the squared gaps between predicted and actual values.
```

## Trying many slopes and seeing the loss valley

Since we only have **one** knob (`slope`), we can simply try **401 candidate slopes** between `0.0` and `4.0`, compute the MSE for each, and pick the lowest!

```{admonition} Predict before you run
:class: tip
The hidden biological truth in our simulated trial is `true_slope = 2.2` mmHg per mg (plus realistic person-to-person wobble).
1. Will the MSE loss be **smaller** at `slope = 0.5` or at `slope = 2.2`?
2. When we plot **loss vs. slope** from `0` to `4`, what shape will the curve make — a straight line, a hill, or a U-shaped **valley**?
```

```{code-cell} python
:label: ch31-mse-sweep
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
true_slope = 2.2
doses = np.linspace(0, 10, 40)
drops = true_slope * doses + rng.normal(0, 1.8, size=len(doses))

def mse_loss(slope):
    predictions = slope * doses
    errors = predictions - drops
    return float(np.mean(errors ** 2))

print(f"Loss if slope = 0.5 (too flat) : {mse_loss(0.5):6.1f}")
print(f"Loss if slope = 2.2 (truth)    : {mse_loss(2.2):6.1f}")
print(f"Loss if slope = 4.0 (too steep): {mse_loss(4.0):6.1f}")

# Try 401 slopes from 0.00 to 4.00 and keep the lowest loss
candidates = np.linspace(0, 4, 401)
losses = [mse_loss(s) for s in candidates]
best_slope = candidates[int(np.argmin(losses))]
print(f"\nBest slope found: {best_slope:.2f} (lowest MSE = {min(losses):.2f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

ax1.scatter(doses, drops, s=24, color="#0072B2", alpha=0.75, label="Volunteers")
ax1.plot(doses, best_slope * doses, color="#D55E00", lw=2.5, label=f"Best line (slope = {best_slope:.2f})")
ax1.set_xlabel("Daily dose (mg)"); ax1.set_ylabel("Blood-pressure drop (mmHg)")
ax1.set_title("A. Data and best-fit line"); ax1.legend(frameon=False); ax1.grid(alpha=0.25)

ax2.plot(candidates, losses, color="#009E73", lw=2.4)
ax2.scatter(best_slope, min(losses), s=90, color="#D55E00", zorder=5, label=f"Bottom of valley ({best_slope:.2f})")
ax2.set_xlabel("Candidate slope"); ax2.set_ylabel("Loss (MSE)")
ax2.set_title("B. The loss valley"); ax2.legend(frameon=False); ax2.grid(alpha=0.25)

fig.tight_layout()
plt.show()
```

Look at Panel B! Fitting a model is literally **finding the bottom of a valley**. Notice also that even at the very bottom, the lowest loss isn't `0.00` — it's around `3.7`, which reflects the real person-to-person biological wobble that *no* straight line can eliminate.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.1.1 (Modify) — More person-to-person wobble
:class: hint
In [](#ch31-mse-sweep), change the biological wobble `1.8` in `rng.normal(0, 1.8, ...)` up to `5.0` and re-run. What happens to the lowest MSE at the bottom of the valley, and how close is `best_slope` to `2.2` now?
```

```{admonition} Solution
:class: dropdown
```python
drops_noisy = true_slope * doses + rng.normal(0, 5.0, size=len(doses))
losses_n = [float(np.mean((s * doses - drops_noisy) ** 2)) for s in candidates]
print(f"Noisier best slope: {candidates[int(np.argmin(losses_n))]:.2f} | lowest MSE = {min(losses_n):.1f}")
```
The valley floor rises sharply (roughly `~25` instead of `3.7`) because random spread around the line cannot be explained away, and the best slope drifts a bit farther from `2.2`!
```

```{admonition} Exercise 3.1.2 (Make) — Score our 3-volunteer hand example in Python
:class: hint
Check our hand calculation (`doses_3 = np.array([1, 2, 3])`, `drops_3 = np.array([3, 5, 6])`) by printing the MSE for `slope = 1.5`, `2.0`, and `2.5`. Which of those three wins?
```

```{admonition} Solution
:class: dropdown
```python
d3, y3 = np.array([1, 2, 3]), np.array([3, 5, 6])
for s in [1.5, 2.0, 2.5]:
    print(f"slope = {s:.1f} -> MSE = {np.mean((s * d3 - y3) ** 2):.2f}")
```
`slope = 2.0` gets `0.67`, beating both `1.5` (`3.58`) and `2.5` (`1.25`)!
```

```{admonition} Exercise 3.1.3 (Critical Reasoning) — Why brute force won't scale to AI
:class: hint
Testing 100 candidate values for **1 knob** takes 100 checks. Testing every combination of 100 values for **2 knobs** takes $100 \times 100 = 10{,}000$ checks. Roughly how many checks would brute force take for a tiny AI with just **10 knobs** — and why do we need a smarter search trick next chapter?
```

```{admonition} Solution
:class: dropdown
$100^{10} = 10^{20}$ (one hundred quintillion!) checks — even a supercomputer would take years, and real neural networks have **millions or billions** of knobs! We need a way to find the bottom of the valley *without* checking every spot — which is exactly what **gradient descent** does next.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **model** turns inputs into predictions using adjustable **parameters** (knobs).
- **Mean Squared Error (MSE)** scores how wrong a model is by squaring and averaging prediction errors.
- Plotting loss against a parameter creates a U-shaped **loss valley**; fitting a model means finding its bottom.
```

*Next up:* How do you find the bottom of a fog-covered valley **without checking every point**? Roll downhill with calculus-free **gradient descent** in [](3_2-gradient-descent.md)!
