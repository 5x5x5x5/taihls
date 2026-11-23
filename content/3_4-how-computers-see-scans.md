---
title: 'Chapter 3.4 — Teaching Computers to See: Pixels and Image Filters'
short_title: '3.4 · Seeing medical scans'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.4 — Teaching Computers to See: Pixels and Image Filters

*Week 9 · ~45 minutes · Prerequisites: [](3_3-inside-a-neural-network.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/02-medical-images-cnn.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why a digital medical scan is simply a **grid of brightness numbers (pixels)**;
- trace a $3 \times 3$ **convolution** (sliding image filter) by hand and write one in 8 lines of Python;
- explain how **convolutional neural networks (CNNs)** learn filters automatically to detect edges, textures, and tumors anywhere in a scan;
- connect our toy filter to real clinical evidence — including Sweden's 100,000-patient **MASAI** mammography trial.
```

## Start with a case: helping radiologists read 100,000 scans

Radiologists look at thousands of chest X-rays and mammograms where 99 out of 100 look healthy — yet missing one tiny shadow can cost a life. In Sweden's landmark **MASAI randomized clinical trial** of over 100,000 women, pairing radiologists with medical-imaging AI detected **more cancers without raising false alarms** while cutting radiologist reading workload by roughly **44%** [@lang2023masai; @lang2026masai]. Similar image AI has also matched dermatologists at spotting suspicious skin lesions from photos [@esteva2017skin].

Yet a computer has no eyes! So how does software "look" at a scan?

## Step 1: Every picture is a grid of numbers

To a computer, a grayscale scan is just a grid of {term}`Pixel`s — one brightness number per tiny square (`0` = black air, big number = bright white bone). Let's print one real 8×8 grayscale image from `scikit-learn` (`load_digits`) as raw numbers so you can *see* the picture hidden in the table!

```{code-cell} python
:label: ch34-pixel-grid
from sklearn.datasets import load_digits

digits = load_digits()
print("Image #0 as a raw 8x8 table of brightness numbers (0 = blank, 16 = bright):")
print(digits.images[0].astype(int))
```

Squint at that table of numbers — you can literally see the oval shape of the digit **`0`** traced out by the big numbers `12–16` around a hollow center of `0`s!

## Step 2: Sliding a tiny detector across the scan (Convolution!)

Why not just flatten a 1,000 × 1,000 chest X-ray into a million separate inputs and hand them to Chapter 3.3's network? Two problems:
1. A million inputs need tens of millions of weights right away.
2. Flattening throws away **which pixels sit next to each other**!

Instead, a {term}`Convolutional neural network (CNN)` slides a tiny $3 \times 3$ magnifying glass of numbers — called a **filter** (or **kernel**) — across every spot in the image, multiplying overlapping numbers and adding them up. That sliding multiply-and-add is a {term}`Convolution`!

```{mermaid}
flowchart LR
    Img["Medical Scan<br/>(Pixel grid)"] --> F1["Layer 1 Filters<br/>Find simple edges<br/>& bright spots"]
    F1 --> F2["Deeper Filters<br/>Combine edges into<br/>shapes & nodules"]
    F2 --> Out["Decision<br/>Normal vs. Needs Review"]
```

**Test a $3 \times 3$ bright-spot filter by hand!**
Look at this 9-number filter (`+8` in the middle, `-1` on all 8 neighbors — notice its 9 numbers add up to `0`!):

```text
[-1, -1, -1]
[-1, +8, -1]
[-1, -1, -1]
```

- Slide it over a **solid white bone** (all `1`s): `8×1 + 8×(-1) = 0` → **ignores flat areas!**
- Slide it right onto a **tiny bright spot (`1`) surrounded by dark lung (`0`)**: `8×1 + 8×0 = +8` → **lights up brightly!**

## Write a real 2D convolution in 8 lines of Python!

Let's build a clean `20 × 20` synthetic scan containing a bright vertical bone wall on the left and a bright nodule spot on the right, then slide **two** $3 \times 3$ filters across it!

```{admonition} Predict before you run
:class: tip
We will slide:
1. A **vertical-edge filter** (left column `-1`, right column `+1`), and
2. Our **bright-spot filter** (`+8` center, `-1` border).

Which filter will light up the **edges of the vertical bone wall**, and which will highlight the **nodule's outline** while ignoring solid flat areas?
```

```{code-cell} python
:label: ch34-conv-scratch
import numpy as np
import matplotlib.pyplot as plt

# 1. Build a simple 20x20 synthetic scan: vertical bone bar + bright nodule + light grain
rng = np.random.default_rng(0)
scan = rng.normal(0.15, 0.04, size=(20, 20))
scan[:, 3:6]   += 0.75    # vertical bright bone wall on the left
scan[8:12, 13:17] += 0.80 # bright nodule spot on the right
scan = np.clip(scan, 0, 1)

# 2. 2D convolution in 8 clear lines!
def slide_filter(image, kernel):
    rows, cols = image.shape
    out = np.zeros((rows - 2, cols - 2))
    for r in range(rows - 2):
        for c in range(cols - 2):
            patch = image[r:r+3, c:c+3]
            out[r, c] = np.sum(patch * kernel)
    return out

vertical_filter = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

spot_filter = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

edges = np.abs(slide_filter(scan, vertical_filter))
spots = np.maximum(0, slide_filter(scan, spot_filter))  # ReLU keeps positive highlights!

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, img, title, cmap in [
    (axes[0], scan,  "A. Original 20×20 scan\n(bone bar + nodule)", "gray"),
    (axes[1], edges, "B. Vertical-edge filter\n(highlights vertical walls)", "magma"),
    (axes[2], spots, "C. Spot filter + ReLU\n(lights up corners/rims!)", "magma"),
]:
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9.8)
    ax.axis("off")
fig.tight_layout()
plt.show()
```

Look at Panels B and C! With just nine numbers slid across the image, a computer automatically extracts edges and bright boundaries — and **finds the same spot no matter where it appears on the scan!** In a real CNN, nobody designs those 9-number filters by hand: **gradient descent learns hundreds of useful filters automatically from labeled medical scans!**

```{admonition} Going deeper — train a real CNN on pediatric chest X-rays!
:class: dropdown
Want to train a real PyTorch CNN on **thousands of real pediatric chest X-rays** (PneumoniaMNIST) and inspect the exact scans it misses? Open [Companion Colab Notebook 02](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/02-medical-images-cnn.ipynb) on a free Colab GPU!

Modern medical imaging also uses **Vision Transformers (ViTs)** — which chop a scan into small square tiles (like words in a sentence!) so distant regions of a scan can compare notes directly using **attention** (Unit 4).
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.4.1 (Modify) — Build a horizontal-edge filter
:class: hint
Create a $3 \times 3$ **horizontal-edge filter** (top row `-1`, middle row `0`, bottom row `+1`), run `np.abs(slide_filter(scan, horiz_filter))`, and plot it. Which lines on our synthetic scan light up now?
```

```{admonition} Solution
:class: dropdown
```python
horiz_filter = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])
plt.figure(figsize=(3.6, 3.6))
plt.imshow(np.abs(slide_filter(scan, horiz_filter)), cmap="magma")
plt.title("Horizontal edges only!"); plt.axis("off"); plt.show()
```
The vertical bone bar disappears completely, leaving ONLY the top and bottom horizontal edges of the nodule!
```

```{admonition} Exercise 3.4.2 (Make) — Classify real 8×8 images!
:class: hint
Using `digits = load_digits()` (`X = digits.data / 16.0`, `y = digits.target`), split 70/30 and train a small `MLPClassifier(hidden_layer_sizes=(40,), max_iter=500, random_state=0)`. What held-out accuracy do you get on reading 10 handwritten digits from pixel grids?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
clf = MLPClassifier(hidden_layer_sizes=(40,), max_iter=500, random_state=0).fit(X_tr, y_tr)
print(f"Held-out accuracy on 8x8 digit images: {clf.score(X_te, y_te):.1%}")
```
Roughly **97–98%** — purely from treating each tiny picture as 64 brightness numbers!
```

```{admonition} Exercise 3.4.3 (Critical Reasoning) — Why did MASAI run a real hospital trial?
:class: hint
Even after image AI scored >95% on stored historical mammograms, Swedish doctors spent years running the **MASAI randomized trial** [@lang2023masai; @lang2026masai] on 100,000 real patients. Name **one thing** a live hospital trial measures that an archived dataset never can.
```

```{admonition} Solution
:class: dropdown
A live trial measures how **real doctors interact with the AI** (whether they catch AI mistakes or get tired/distracted), whether the AI catches genuinely dangerous cancers *earlier*, and whether it safely cuts workload on live scanners without raising false alarms!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Digital medical scans are grids of brightness **pixels**; flattening big images throws away neighbor geometry.
- A **2D convolution** slides a tiny filter across the scan to spot edges and blobs anywhere they appear.
- **CNNs** learn hundreds of layered filters automatically via gradient descent and already assist clinicians in trials like **MASAI** [@lang2026masai].
```

*Next up:* High test accuracy on medical images hides a famous trap! What happens when an image AI gets ~90% right **without even looking at the patient's lungs**? Catch a cheating hospital AI red-handed in [](3_5-shortcuts-and-clever-hans.md)!
