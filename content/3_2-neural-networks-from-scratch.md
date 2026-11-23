---
title: 'Chapter 3.2 — Inside a Neural Network: Neurons, Activations, and Learning Curves'
short_title: '3.2 · Neural networks'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.2 — Inside a Neural Network: Neurons, Activations, and Learning Curves

*Estimated time: ~50 minutes · Prerequisites: [](2_4-trees-ensembles-and-boundaries.md), [](3_1-loss-and-gradient-descent.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/01-pytorch-neural-net.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- trace a **forward pass** through a small neural network (inputs → weights & biases → **activation function** → output probability);
- explain *why* nonlinear activations (**ReLU** or **sigmoid**) are the exact ingredient that lets a network bend its decision boundary;
- **train a real neural network from scratch** in pure NumPy using the calculus-free downhill peeking loop from Chapter 3.1;
- visualize how individual hidden neurons act as simple "creases" that combine to surround a curved biological zone.
```

## Start with a case: the curved therapeutic window

In oncology and antimicrobial therapy, doctors frequently combine **two drugs** (Drug A and Drug B) to attack resistant cells. Biology is rarely straight-lined:
- If *both* doses are too low, the pathogen survives (**ineffective**).
- If *either* dose climbs too high — or if the combined total crosses a curved toxicity curve — the patient suffers severe side effects (**toxic**).
- Only inside a **curved crescent-shaped "therapeutic window"** is the combination both **safe and effective**!

No single straight line (logistic regression) can carve out a curved window like that. To separate curved biological spaces, we assemble tiny straight-line decision-makers into a {term}`Neural network`.

```{mermaid}
flowchart LR
    X1["Input 1<br/>Drug A Dose"] --> H1["Hidden Neuron 1<br/>(bent crease)"]
    X1 --> H2["Hidden Neuron 2<br/>(bent crease)"]
    X1 --> H3["Hidden Neuron 3<br/>(bent crease)"]
    X1 --> H4["Hidden Neuron 4<br/>(bent crease)"]
    X2["Input 2<br/>Drug B Dose"] --> H1
    X2 --> H2
    X2 --> H3
    X2 --> H4
    H1 --> Out["Output Neuron<br/>P(Safe & Effective)"]
    H2 --> Out
    H3 --> Out
    H4 --> Out
```

## Anatomy of an artificial neuron

Every circle in the diagram above performs the exact same two-step recipe:
1. **Weighted sum:** Multiply each input by its learnable **weight**, add them up, and add a learnable **bias** (a shift knob): $z = w_1 x_1 + w_2 x_2 + b$.
2. **Bend it!** Pass $z$ through a nonlinear {term}`Activation function`!

::::{tab-set}

:::{tab-item} Sigmoid — Smooth S-Curve (0 to 1)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Squashes any real number smoothly between `0` and `1` — perfect for an output neuron reporting a **probability**!
:::

:::{tab-item} ReLU — Rectified Linear Unit
$$\text{ReLU}(z) = \max(0, z)$$
Literally *"keep positive numbers, zero out negative numbers."* It looks almost comically simple — just one sharp hinge at zero — yet stacking ReLU hinges is the standard workhorse inside modern deep neural networks because it trains rapidly without flattening out!
:::

::::

:::{margin}
**Why is the bend mandatory?** If you multiply and add inputs (`Layer 1`), and then multiply and add *those* numbers (`Layer 2`) with no nonlinear activation in between, high-school algebra collapses the whole network back into a single straight line! The bend is what gives depth its power.
:::

## Lab 1: Creating the curved drug-synergy dataset

Let's simulate 320 patient dose-combination assays where the safe-and-effective therapeutic zone forms a curved arch in `(Drug A, Drug B)` space — and prove first that a straight-line classifier fails on it.

```{code-cell} python
:label: ch32-drug-window
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(7)
n = 320
# Standardized concentrations of Drug A and Drug B
drug_a = rng.uniform(-2.0, 2.0, size=n)
drug_b = rng.uniform(-2.0, 2.0, size=n)
X = np.column_stack([drug_a, drug_b])

# Safe & effective only inside a curved synergy ring (distance from optimal center)
radius = np.sqrt((drug_a) ** 2 + (drug_b + 0.35) ** 2)
effective_score = (radius > 0.65) & (radius < 1.55) & (drug_b > -1.0)
# Add slight biological assay noise (flip ~3% of labels)
flip = rng.random(n) < 0.03
y = np.where(flip, ~effective_score, effective_score).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

lin_clf = LogisticRegression().fit(X_tr, y_tr)
print(f"Majority baseline accuracy       : {max(y_te.mean(), 1 - y_te.mean()):.1%}")
print(f"Straight-line Logistic Regression: {lin_clf.score(X_te, y_te):.1%}  <-- cannot bend!")
```

A straight-line classifier barely beats guessing the majority class! Now let's build a **tiny 6-hidden-neuron neural network** (just **25 learnable numbers** total!) and **train it right here** using the exact same calculus-free peeking gradient descent from Chapter 3.1!

## Lab 2: Training a real neural network from scratch (zero calculus!)

Watch closely:
- Our network has **25 parameters**: `W1` ($2 \times 6 = 12$ weights), `b1` ($6$ biases), `W2` ($6$ weights), and `b2` ($1$ bias).
- We pack all 25 numbers into one flat vector `params`, define the forward pass (`tanh` — a centered S-shaped cousin of sigmoid — in the hidden layer, `sigmoid` at the output), and measure **binary cross-entropy loss** (the classification cousin of MSE that heavily punishes confident wrong probabilities!).
- In every epoch, our peeking loop nudges each of the 25 knobs slightly to feel its downhill slope and takes a step (adding a touch of **momentum** so the ball rolls smoothly through flat spots!).

```{admonition} Predict before you run
:class: tip
At Epoch 0, all 25 weights are random numbers, so accuracy will start near a coin flip (~50–65%). Using ONLY our calculus-free peeking loop on those 25 numbers, how high do you think held-out test accuracy will climb after 350 epochs — ~70%, ~75%, or **near 90%**?
```

```{code-cell} python
:label: ch32-train-scratch
H = 6   # 6 hidden tanh neurons -> 2*6 + 6 + 6 + 1 = 25 total parameters!

def unpack(params):
    W1 = params[0 : 2 * H].reshape(2, H)
    b1 = params[2 * H : 3 * H]
    W2 = params[3 * H : 4 * H]
    b2 = params[4 * H]
    return W1, b1, W2, b2

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

def forward(X_in, params):
    W1, b1, W2, b2 = unpack(params)
    hidden = np.tanh(X_in @ W1 + b1)               # Smooth nonlinear bend (-1..+1)
    probs = sigmoid(hidden @ W2 + b2)              # Output probability (0..1)
    return probs, hidden

def bce_loss(params, X_in, y_in):
    p, _ = forward(X_in, params)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_in * np.log(p) + (1 - y_in) * np.log(1 - p)))

# Initialize 25 random parameters and train via calculus-free peeking + momentum!
rng_nn = np.random.default_rng(42)
params = rng_nn.normal(0, 1.0, size=4 * H + 1)
velocity = np.zeros_like(params)

lr, momentum, peek, epochs = 0.40, 0.85, 1e-3, 350
tr_losses, te_losses = [], []

for ep in range(epochs):
    tr_losses.append(bce_loss(params, X_tr, y_tr))
    te_losses.append(bce_loss(params, X_te, y_te))

    # Feel the downhill slope for each of the 25 knobs
    grads = np.zeros_like(params)
    for j in range(len(params)):
        params[j] += peek
        loss_plus = bce_loss(params, X_tr, y_tr)
        params[j] -= 2 * peek
        loss_minus = bce_loss(params, X_tr, y_tr)
        params[j] += peek
        grads[j] = (loss_plus - loss_minus) / (2 * peek)

    velocity = momentum * velocity + lr * grads
    params -= velocity

p_te, _ = forward(X_te, params)
nn_acc = ((p_te >= 0.5).astype(int) == y_te).mean()
print(f"Start loss (epoch   0): {tr_losses[0]:.3f}")
print(f"Final loss (epoch {epochs}): {tr_losses[-1]:.3f}")
print(f"Trained 6-neuron network TEST accuracy: {nn_acc:.1%}  (vs. linear {lin_clf.score(X_te, y_te):.1%}!)")
```

**Nearly 88% test accuracy (an almost +19-point leap over the straight-line model!) — trained completely from scratch in 25 lines of NumPy with zero calculus and zero deep-learning libraries!**

## Lab 3: Opening the black box — what did the network actually draw?

Let's plot **(A)** the training & test learning curves sliding smoothly downhill, **(B)** the curved boundary our trained network carved around the therapeutic arch, and **(C)** a peek *inside* three of the individual hidden neurons so you can literally see how straight ReLU hinges add together to build a curved shape!

```{code-cell} python
:label: ch32-visual-inside
import matplotlib.pyplot as plt

BLUE, VERMILION, TEAL, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"

xx, yy = np.meshgrid(np.linspace(-2.1, 2.1, 220), np.linspace(-2.1, 2.1, 220))
grid = np.c_[xx.ravel(), yy.ravel()]
zz, hidden_grid = forward(grid, params)
zz = zz.reshape(xx.shape)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

# Panel A: Learning curves
axes[0].plot(tr_losses, lw=2.2, color=BLUE, label="Training loss")
axes[0].plot(te_losses, lw=2.2, color=VERMILION, ls="--", label="Test loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy loss")
axes[0].set_title("A. Watching the 25 weights learn")
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.25)
axes[0].spines[["top", "right"]].set_visible(False)

# Panel B: Final learned curved decision boundary
axes[1].contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.55)
axes[1].contour(xx, yy, zz, levels=[0.5], colors=["#111111"], linewidths=2.0)
for lbl, col, name in [(0, BLUE, "Ineffective/Toxic"), (1, VERMILION, "Therapeutic window")]:
    m = y_te == lbl
    axes[1].scatter(X_te[m, 0], X_te[m, 1], s=22, color=col, edgecolors="white", lw=0.4, label=name)
axes[1].set_xlabel("Drug A dose (standardized)")
axes[1].set_ylabel("Drug B dose (standardized)")
axes[1].set_title(f"B. Trained curved boundary (test = {nn_acc:.1%})")
axes[1].legend(frameon=False, loc="lower left", fontsize=8)
axes[1].spines[["top", "right"]].set_visible(False)

# Panel C: Inside the hidden layer — each neuron draws one smooth step fence (act = 0)!
W1, b1, W2, b2 = unpack(params)
axes[2].set_xlim(-2.1, 2.1)
axes[2].set_ylim(-2.1, 2.1)
for h_i in range(H):
    act = hidden_grid[:, h_i].reshape(xx.shape)
    color_h = TEAL if W2[h_i] >= 0 else AMBER
    axes[2].contour(xx, yy, act, levels=[0.0], colors=[color_h], linewidths=2.0, linestyles="-")
for lbl, col in [(0, BLUE), (1, VERMILION)]:
    m = y_te == lbl
    axes[2].scatter(X_te[m, 0], X_te[m, 1], s=14, color=col, alpha=0.35)
axes[2].plot([], [], color=TEAL, lw=2, label="Neuron pushes UP (+W2)")
axes[2].plot([], [], color=AMBER, lw=2, label="Neuron pushes DOWN (−W2)")
axes[2].set_xlabel("Drug A dose (standardized)")
axes[2].set_ylabel("Drug B dose (standardized)")
axes[2].set_title("C. Inside: 6 smooth fences enclose the arch!")
axes[2].legend(frameon=False, loc="lower left", fontsize=8)
axes[2].spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

**Stop and admire Panel C of [](#ch32-visual-inside)!** That is the deepest intuition in all of deep learning made visible in one picture:
- Each hidden neuron draws **one straight S-curve fence** across the plane.
- Some fences push probability **up** (teal, enclosing the outer arch) while others push probability **down** (amber, carving out the inner hole!).
- Combined by the output neuron, six simple creases wrap tightly around the entire curved therapeutic zone!

```{admonition} Scale up in PyTorch (Companion Colab Notebook 1)
:class: tip
Our peeking loop taught 25 weights in a fraction of a second. To train networks with thousands or millions of weights, researchers use **PyTorch**, where `loss.backward()` computes all downhill slopes simultaneously via backpropagation. Train this exact drug-synergy network (using ReLU layers, `Adam`, and deeper architectures) on a free GPU in [Companion Notebook 01](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/01-pytorch-neural-net.ipynb).
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.2.1 (Modify) — Delete the activation bend!
:class: hint
In `forward()` inside [](#ch32-train-scratch), remove the `tanh` bend by changing `hidden = np.tanh(X_in @ W1 + b1)` to `hidden = X_in @ W1 + b1` and re-train (with a smaller `lr = 0.05`). What happens to the test accuracy and the shape of the decision boundary?
```

```{admonition} Solution
:class: dropdown
Without `np.tanh(...)`, test accuracy crashes back down to roughly **~69%** — identical to plain Logistic Regression — and the decision boundary collapses into a single straight line! Even with 6 hidden neurons and 25 parameters, multiplying and adding without a nonlinear activation cannot bend.
```

```{admonition} Exercise 3.2.2 (Make) — Compare with scikit-learn's MLPClassifier
:class: hint
Train `MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=0)` from `sklearn.neural_network` (which uses real backpropagation!) on `X_tr, y_tr` and print its held-out test accuracy on `X_te, y_te`. How does it compare with our 25-parameter scratch network?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=0)
mlp.fit(X_tr, y_tr)
print(f"scikit-learn 2-layer MLP test accuracy: {mlp.score(X_te, y_te):.1%}")
```

Both reach roughly **90–94%** (close to the ceiling set by our ~4% biological assay noise!). Notice that adding two layers of 16 neurons (over 300 parameters) barely beats our transparent 6-neuron network because 6 ReLU hinges are already enough to fence this 2D window cleanly.
```

```{admonition} Exercise 3.2.3 (Critical Reasoning) — Why not always use a huge neural network?
:class: hint
Suppose you only have **80 patients** in a rare-disease biomarker study, and you train a deep neural network with **50,000 weights**. Using what you learned in Chapter 2.2 and Panel C above, what is the biggest danger — and what happens to training loss vs. test loss?
```

```{admonition} Solution
:class: dropdown
With 50,000 adjustable creases for just 80 patients, the network has vastly more flexibility than signal: it can wrap a tiny private fence around *every single training patient* — driving training loss to `0.000` while test loss climbs sharply (**severe overfitting**). On small tabular biomedical studies, simpler regularized models (logistic regression, shallow forests) almost always generalize better unless you have massive datasets or **pretrained** representations (Part 4!).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **neural network** chains together layers of artificial neurons, each doing a weighted sum plus a nonlinear **activation** (**ReLU** or **sigmoid**).
- Without the nonlinear bend, stacking layers collapses back into a single straight line.
- Each hidden ReLU neuron draws one simple crease; adding them together carves complex curved boundaries like our drug-synergy window.
- Neural networks learn by the **exact same gradient descent** from Chapter 3.1 — backpropagation is simply a fast shortcut for computing downhill slopes across millions of weights at once.
```

*Next up:* Plain dense layers work on 2 or 30 tabular features, but what happens when our input is a **1,000 × 1,000 medical image**? Let's discover how **convolutions** teach computers to see in [](3_3-medical-imaging-and-cnns.md).
