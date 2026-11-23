---
title: 'Chapter 3.3 — Inside a Neural Network: Neurons, Layers, and Curves'
short_title: '3.3 · Neural networks'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.3 — Inside a Neural Network: Neurons, Layers, and Curves

*Week 8 · ~45 minutes · Prerequisites: [](2_5-lines-trees-and-forests.md), [](3_2-gradient-descent.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/01-pytorch-neural-net.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- trace **one artificial neuron** by hand (multiply inputs by weights, add a bias, apply a bend);
- explain why a nonlinear **activation function** (**ReLU** or **sigmoid**) is the secret that lets a **neural network** curve;
- prove on a two-drug biology problem that a straight line fails while a small trained neural network bends cleanly around the safe zone;
- explain what a **hidden layer** does in plain English.
```

## Start with a case: the curved safe drug window

To stop stubborn infections or resistant tumors, doctors sometimes combine **two medicines** (Drug A and Drug B). Often the safe-and-effective dose combination forms a **curved arch**:
- Too little of both drugs → the infection survives (**ineffective**).
- Too much of either drug → side effects become unsafe (**toxic**).
- Only inside a **curved band in the middle** is the pair **safe and effective**!

Can a single straight line (Logistic Regression from Chapter 2.5) cut out a curved arch? No! To draw curved boundaries, we connect tiny straight-line decision-makers into a {term}`Neural network`.

```{mermaid}
flowchart LR
    A["Input 1<br/>Drug A Dose"] --> H1["Hidden Neuron 1"]
    A --> H2["Hidden Neuron 2"]
    A --> H3["Hidden Neuron 3"]
    B["Input 2<br/>Drug B Dose"] --> H1
    B --> H2
    B --> H3
    H1 --> Out["Output Neuron<br/>Safe Window? (0%–100%)"]
    H2 --> Out
    H3 --> Out
```

## Trace ONE artificial neuron by hand!

An artificial neuron sounds mysterious, yet it is just **two lines of high-school algebra**:
1. **Weighted score:** Multiply each input by its importance (**weight**), add them up, and add a starting shift (**bias**):
   $$\text{score} = (w_1 \times x_1) + (w_2 \times x_2) + b$$
2. **Bend it!** Pass `score` through an {term}`Activation function`!

The two most famous bends in all of AI are surprisingly simple:
- **ReLU** (*"keep positive, zero out negative"*): $\text{ReLU}(z) = \max(0, z)$.
  - Try it by hand: $\text{ReLU}(3.5) = \mathbf{3.5}$, and $\text{ReLU}(-2.0) = \mathbf{0}$!
- **Sigmoid** (*"squash any number smoothly between `0` and `1`"*): great for the final output neuron so it reports a probability like `0.87` (`87%`)!

**Quick hand check:** Suppose a neuron has weights `w1 = 2`, `w2 = -1`, and bias `b = -1`. For patient `(x1 = 3, x2 = 2)`:
- Weighted score $= (2 \times 3) + (-1 \times 2) - 1 = 6 - 2 - 1 = \mathbf{3}$.
- After ReLU bend: $\max(0, 3) = \mathbf{3}$ (fires!).
- For patient `(x1 = 0, x2 = 2)`: score $= 0 - 2 - 1 = -3$ → after ReLU: **`0`** (stays quiet!).

:::{margin}
**Why do we NEED the bend?** If you take `2x + 1` and plug it into `3(2x + 1) - 4` without any bend, algebra simplifies it right back to `6x - 1` — just another straight line! The ReLU hinge is what lets stacked neurons **fold and bend** space.
:::

## Straight line vs. trained neural network on the drug window!

Let's generate 320 two-drug lab assays with a curved safe arch, then compare a **straight-line model** (`LogisticRegression`) against a small **ReLU neural network** (`MLPClassifier` with two hidden layers of 16 ReLU neurons — trained by gradient descent!).

```{admonition} Predict before you run
:class: tip
In this curved drug-window dataset, about **69%** of tested combinations fall outside the safe window (so always guessing "unsafe" gets 69%).
1. Will a straight-line classifier do much better than 69%?
2. What happens when we give a **neural network** hidden layers of bent ReLU neurons?
```

```{code-cell} python
:label: ch33-nn-drug-window
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Create the curved safe therapeutic window (320 lab assays)
rng = np.random.default_rng(7)
drug_a = rng.uniform(-2.0, 2.0, size=320)
drug_b = rng.uniform(-2.0, 2.0, size=320)
X = np.column_stack([drug_a, drug_b])

dist = np.sqrt(drug_a ** 2 + (drug_b + 0.35) ** 2)
in_window = (dist > 0.65) & (dist < 1.55) & (drug_b > -1.0)
y = np.where(rng.random(320) < 0.03, ~in_window, in_window).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

# 2. Train a straight line vs. a small ReLU neural network
line_model = LogisticRegression().fit(X_tr, y_tr)
nn_model   = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu",
                           learning_rate_init=0.01, max_iter=2000, random_state=0).fit(X_tr, y_tr)

print(f"Always-guess-unsafe baseline : {max(y_te.mean(), 1 - y_te.mean()):.1%}")
print(f"Straight-line model          : {line_model.score(X_te, y_te):.1%}  (cannot bend!)")
print(f"ReLU Neural Network          : {nn_model.score(X_te, y_te):.1%}  (curves cleanly around the arch!)")

# 3. Plot both decision boundaries side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True, sharey=True)
for ax, clf, title in [
    (ax1, line_model, f"A. Straight line ({line_model.score(X_te, y_te):.1%})"),
    (ax2, nn_model,   f"B. ReLU Neural Network ({nn_model.score(X_te, y_te):.1%})"),
]:
    DecisionBoundaryDisplay.from_estimator(clf, X_tr, response_method="predict_proba",
                                           cmap="RdBu_r", alpha=0.5, ax=ax)
    ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1], s=20, color="#0072B2", edgecolors="white", lw=0.4, label="Unsafe / Ineffective")
    ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1], s=20, color="#D55E00", edgecolors="white", lw=0.4, label="Safe therapeutic window")
    ax.set_title(title)
    ax.set_xlabel("Drug A dose (std)")

ax1.set_ylabel("Drug B dose (std)")
ax1.legend(frameon=False, loc="lower left", fontsize=8)
fig.tight_layout()
plt.show()
```

Look at Panel A vs. Panel B! The straight line is completely stuck at the `68.8%` guessing baseline, while stacked ReLU hinges — trained automatically by the exact gradient-descent principle from Chapter 3.2 — wrap smoothly around the curved safe arch and leap to **`92.7%`**!

```{admonition} Going deeper — build and train this in PyTorch!
:class: dropdown
In real AI labs, researchers build neural networks in **PyTorch**, where you can watch the training loss drop epoch by epoch on a GPU. Try training this exact drug-window network in [Companion Colab Notebook 01](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/01-pytorch-neural-net.ipynb)!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.3.1 (Modify) — Turn off the ReLU bend!
:class: hint
In `MLPClassifier(...)` inside [](#ch33-nn-drug-window), change `activation="relu"` to `activation="identity"` (meaning **no bend at all** — just straight multiplication and addition!) and re-run. What happens to the neural network's test accuracy and boundary shape?
```

```{admonition} Solution
:class: dropdown
```python
flat_nn = MLPClassifier(hidden_layer_sizes=(12,), activation="identity",
                        max_iter=1500, random_state=0).fit(X_tr, y_tr)
print(f"12-neuron network with NO bend: {flat_nn.score(X_te, y_te):.1%}")
```
Accuracy crashes right back to **`68.8%`** (identical to the straight line)! Without a nonlinear bend, all 12 hidden neurons collapse into a single straight cut.
```

```{admonition} Exercise 3.3.2 (Make) — Code one single neuron in 4 lines of Python
:class: hint
Write a tiny function `one_neuron(x1, x2, w1, w2, b)` that computes `score = w1 * x1 + w2 * x2 + b` and returns `max(0, score)` (ReLU!). Test it on our hand-worked points `(3, 2)` and `(0, 2)` with `w1=2, w2=-1, b=-1`.
```

```{admonition} Solution
:class: dropdown
```python
def one_neuron(x1, x2, w1, w2, b):
    score = w1 * x1 + w2 * x2 + b
    return max(0, score)

print("Point (3, 2):", one_neuron(3, 2, 2, -1, -1))   # -> 3
print("Point (0, 2):", one_neuron(0, 2, 2, -1, -1))   # -> 0
```
Every hidden neuron inside `MLPClassifier` above is doing literally those two lines!
```

```{admonition} Exercise 3.3.3 (Critical Reasoning) — More neurons isn't always better
:class: hint
Suppose you only have **50 patients** in a rare-disease study and you train a huge neural network with `hidden_layer_sizes=(200, 200)` (over 40,000 adjustable knobs!). Remembering Chapter 2.2, what is the big danger?
```

```{admonition} Solution
:class: dropdown
**Severe overfitting!** With 40,000 knobs for just 50 patients, the network can draw tiny private bubbles around every single training patient — hitting 100% training accuracy while failing badly on new patients.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- One artificial **neuron** does a weighted sum (`w1*x1 + w2*x2 + b`) followed by a nonlinear **activation** bend (**ReLU** or **sigmoid**).
- A **hidden layer** combines many bent neurons so a **neural network** can carve curved boundaries no straight line can draw.
- Without the nonlinear activation, extra layers add zero bending power.
```

*Next up:* Our drug network only had 2 inputs. What if the input is a **chest X-ray with a million pixels**? Let's write a sliding image filter by hand in [](3_4-how-computers-see-scans.md)!
