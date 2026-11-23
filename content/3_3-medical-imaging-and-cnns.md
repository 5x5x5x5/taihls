---
title: 'Chapter 3.3 — Teaching Computers to See: Pixels, Convolutions, and Medical Scans'
short_title: '3.3 · Medical imaging & CNNs'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.3 — Teaching Computers to See: Pixels, Convolutions, and Medical Scans

*Estimated time: ~50 minutes · Prerequisites: [](3_2-neural-networks-from-scratch.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/02-medical-images-cnn.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why a digital medical scan is a 2D (or 3D) **grid of brightness numbers** (pixels/voxels);
- implement a **2D convolution** filter from scratch in 10 lines of NumPy and see how small **kernels** detect edges and circular lesions;
- explain how **convolutional neural networks (CNNs)** and **Vision Transformers (ViTs)** build hierarchical visual understanding;
- summarize landmark real-world clinical imaging evidence, including the **MASAI** randomized mammography trial.
```

## Start with a case: reading 100,000 mammograms

Radiologists face an exhausting bottleneck: screening tens of thousands of mammograms, chest X-rays, and skin photographs where 99 out of 100 scans look normal, yet missing one subtle lesion can cost a life.

Early landmark studies showed that deep neural networks could classify skin lesions at dermatologist-level accuracy [@esteva2017skin]. Even more importantly, in Sweden's **MASAI randomized controlled trial** involving over **100,000 women** [@lang2023masai; @lang2026masai], researchers put AI-assisted mammography screening to the ultimate clinical test against standard double-reading by two radiologists: AI-supported triage detected **more cancers without increasing false positives**, reduced interval cancers, and cut radiologist screen-reading workload by roughly **44%**!

Yet a computer has no eyes. How does an algorithm transform a raw medical scan into clinical vision?

## Step 1: Every scan is a grid of numbers

A grayscale chest X-ray is a 2D matrix of **pixels** — each cell holds one brightness number (typically `0` = pitch black air, `255` = bright white bone). A color dermatology photo stacks **three** grids (Red, Green, Blue), and a CT or MRI stacks dozens of 2D slices into a 3D grid of **voxels**.

Why can't we just flatten a 512 × 512 X-ray into a flat 1D row of 262,144 numbers and feed it into Chapter 3.2's dense neural network? Two massive problems:
1. **Explosion of weights:** Just 100 hidden neurons connected to 262,144 pixels would require **26 million weights** in the first layer alone!
2. **Broken geometry:** A flat list has no idea that pixel `(row 10, col 10)` sits right next to `(row 11, col 10)` — and if a lung nodule shifts 5 pixels to the right, a flattened classifier treats it as a completely unfamiliar feature!

## Step 2: The magic of 2D convolution

Instead of learning separate weights for every pixel, a {term}`Convolutional neural network (CNN)` slides a tiny reusable window of weights — called a **kernel** or **filter** (often just $3 \times 3$ or $5 \times 5$) — across every position in the image, multiplying overlapping numbers and summing them up. That sliding dot-product is a {term}`Convolution`!

```{mermaid}
flowchart LR
    Img["Medical Scan<br/>(Pixel grid)"] --> C1["Conv Layer 1<br/>Detects edges &<br/>bright spots"]
    C1 --> P1["Pooling<br/>Shrinks grid,<br/>keeps strongest cues"]
    P1 --> C2["Deeper Conv Layers<br/>Combine edges into<br/>textures & nodules"]
    C2 --> Clf["Classifier Head<br/>P(Normal vs. Disease)"]
```

:::{margin}
Because the **same** $3 \times 3$ kernel slides across the whole scan, it needs only **9 weights** instead of millions — AND it detects a nodule whether that nodule sits in the upper-left lobe or the lower-right lobe (**translation equivariance**)!
:::

## Lab 1: Write a 2D convolution filter by hand!

Let's build a synthetic **28 × 28 simulated radiograph / microscopy patch** containing two clear anatomical structures — a vertical bone/vessel wall AND a small bright circular nodule — plus realistic scanner grain. Then we will write a **7-line 2D convolution** function in pure NumPy and slide **three hand-crafted $3 \times 3$ kernels** across the scan:
1. A **vertical-edge detector** (left column `−1`, right column `+1`),
2. A **horizontal-edge detector** (top row `−1`, bottom row `+1`),
3. A **lesion / spot detector** (bright `+8` center surrounded by `−1` ring — fires *only* on isolated bright blobs, ignoring flat white areas!).

```{admonition} Predict before you run
:class: tip
Think about the **lesion-spot kernel** (`+8` in the center, `−1` on all 8 neighbors — notice its 9 weights sum to `0`!). What number does that kernel output when it slides over a completely **uniform bright bone region** where every pixel is `1.0`? What happens when it lands directly on top of a **small bright nodule** surrounded by dark lung tissue (`0.0`)?
```

```{code-cell} python
:label: ch33-conv2d-scratch
import numpy as np
import matplotlib.pyplot as plt

# 1. Build a 28x28 synthetic scan: vertical rib/wall + bright circular nodule + noise
rng = np.random.default_rng(0)
scan = rng.normal(0.15, 0.05, size=(28, 28))
scan[:, 4:9] += 0.75                                 # bright vertical rib / boundary
yy, xx = np.ogrid[:28, :28]
nodule_mask = ((xx - 19) ** 2 + (yy - 13) ** 2) <= 3.0 ** 2
scan[nodule_mask] += 0.85                            # bright circular nodule!
scan = np.clip(scan, 0, 1)

# 2. 2D convolution in 7 lines of pure NumPy
def conv2d(image, kernel):
    H, W = image.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            patch = image[r : r + kh, c : c + kw]
            out[r, c] = np.sum(patch * kernel)
    return out

# 3. Three classic 3x3 kernels
k_vert = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=float)

k_horiz = k_vert.T

k_spot = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=float)

f_vert = np.abs(conv2d(scan, k_vert))
f_horiz = np.abs(conv2d(scan, k_horiz))
f_spot = np.maximum(0, conv2d(scan, k_spot))         # ReLU! Keep positive spot responses

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
panels = [
    (scan, "Original 28×28 scan\n(rib + circular nodule)", "gray"),
    (f_vert, "Vertical-edge kernel\n(highlights rib edges)", "magma"),
    (f_horiz, "Horizontal-edge kernel\n(highlights top/bottom)", "magma"),
    (f_spot, "Spot-detector + ReLU\n(lights up nodule rim!)", "magma"),
]
for ax, (img, title, cmap) in zip(axes, panels):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9.5)
    ax.axis("off")
fig.tight_layout()
plt.show()
```

**Look at those four panels in [](#ch33-conv2d-scratch)!**
- Over a uniform bright region, `8 × 1.0 + 8 × (−1.0) = 0` — the spot kernel completely ignores flat background and solid bone!
- The vertical-edge kernel lights up the vertical rib edges like neon.
- The spot kernel + ReLU isolates the curved boundary of the circular nodule!

In a real **CNN**, nobody hand-designs those kernels: **gradient descent learns hundreds of kernels automatically from labeled scans** — simple edges in Layer 1, textures in Layer 2, anatomical structures in deeper layers!

## Lab 2: From hand-made features to classifying real images

Even on small real images — like `scikit-learn`'s built-in 8×8 handwritten digits (`load_digits`) — turning pixels into features (or adding simple convolutional edge features!) lets a classifier read visual patterns accurately.

```{code-cell} python
:label: ch33-digits-clf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = load_digits()
X_d, y_d = digits.data / 16.0, digits.target          # scale pixel brightness to 0..1

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X_d, y_d, test_size=0.30, random_state=0, stratify=y_d
)

mlp_img = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0)
mlp_img.fit(X_tr_d, y_tr_d)
preds_d = mlp_img.predict(X_te_d)
print(f"Tiny neural net on 8×8 images: {(preds_d == y_te_d).mean():.1%} held-out accuracy")

# Show a mix of correct hits and actual mistakes
wrong_idx = np.where(preds_d != y_te_d)[0][:4]
right_idx = np.where(preds_d == y_te_d)[0][:4]
show_idx = np.concatenate([right_idx, wrong_idx])

fig, axes = plt.subplots(1, 8, figsize=(12.5, 2.1))
for ax, i in zip(axes, show_idx):
    ax.imshow(X_te_d[i].reshape(8, 8), cmap="gray_r")
    ok = preds_d[i] == y_te_d[i]
    ax.set_title(f"pred {preds_d[i]} (true {y_te_d[i]})",
                 fontsize=8.5, color="#009E73" if ok else "#D55E00")
    ax.axis("off")
fig.suptitle("Inspecting errors is vital: green = correct, vermilion = missed cases", fontsize=10, y=1.02)
fig.tight_layout()
plt.show()
```

## Beyond CNNs: Vision Transformers (ViTs) & Multimodal Medical Models

For a decade, CNNs ruled medical imaging. Today, a second architecture shares the crown: the **Vision Transformer (ViT)**! A ViT chops a high-resolution scan (such as a gigapixel pathology slide or 3D CT slice) into small square patches (e.g., $16 \times 16$ pixels), turns each patch into a **token** — just like a word in a sentence! — and uses **self-attention** (Chapter 4.2) so distant parts of a scan (like comparing the left lung directly to the right lung) can communicate in a single step! Modern open medical models like **MedGemma** combine a Vision Transformer image encoder directly with a clinical language model so one system can read a chest X-ray *and* discuss it with a physician.

```{admonition} Train a real CNN on pediatric chest X-rays (Companion Colab Notebook 2)
:class: tip
Ready to move from 28×28 synthetic scans to **thousands of real clinical chest X-rays**? Open [Companion Notebook 02](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/02-medical-images-cnn.ipynb) on a free Colab GPU to train a real PyTorch CNN on **PneumoniaMNIST** (from MedMNIST), compute its clinical sensitivity and specificity, and inspect the exact X-rays it misclassifies!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.3.1 (Modify) — Build a smoothing (noise-reduction) kernel
:class: hint
In [](#ch33-conv2d-scratch), create a $3 \times 3$ **blur/averaging kernel** where every entry equals `1/9`: `k_blur = np.ones((3, 3)) / 9.0`. Apply `conv2d(scan, k_blur)` and compare the pixel standard deviation of the noisy lung background before and after blurring. Why do radiologists sometimes apply mild smoothing before edge detection?
```

```{admonition} Solution
:class: dropdown
```python
k_blur = np.ones((3, 3)) / 9.0
smoothed = conv2d(scan, k_blur)
print(f"Background SD before blur: {scan[2:10, 14:24].std():.4f}")
print(f"Background SD after  blur: {smoothed[2:10, 14:24].std():.4f}")
```

Averaging 9 neighboring pixels cuts random scanner speckle noise by roughly $3\times$ ($\sqrt{9}$), making genuine anatomical edges stand out much more cleanly when an edge filter runs next!
```

```{admonition} Exercise 3.3.2 (Make) — Extract convolutional features for a classifier!
:class: hint
Using `conv2d` and `k_spot` on `scan` in [](#ch33-conv2d-scratch), write 3 lines of Python to find the `(row, col)` coordinates of the **brightest response** in `f_spot` (`np.unravel_index(np.argmax(f_spot), f_spot.shape)`). Does it land near the synthetic nodule we placed at `(row=13, col=19)`?
```

```{admonition} Solution
:class: dropdown
```python
peak_r, peak_c = np.unravel_index(np.argmax(f_spot), f_spot.shape)
print(f"Strongest spot-kernel response at (row={peak_r + 1}, col={peak_c + 1}) "
      f"— nodule center was (13, 19)")
```

Adding `+1` accounts for the 1-pixel border shrink of a $3 \times 3$ valid convolution. The spot detector locks straight onto the circular nodule's high-contrast rim!
```

```{admonition} Exercise 3.3.3 (Critical Reasoning) — Why did MASAI run a randomized clinical trial?
:class: hint
Before Sweden's **MASAI trial** [@lang2023masai; @lang2026masai], dozens of retrospective papers already claimed mammography AI achieved AUROC > 0.95 on stored hospital scans. Why did researchers still spend years running a **100,000-patient prospective randomized trial** before recommending widespread deployment?
```

```{admonition} Solution
:class: dropdown
Retrospective accuracy on archived images cannot measure what actually happens in a busy hospital workflow:
1. **Human–AI interaction:** Do radiologists suffer from automation bias (accepting AI mistakes) or ignore helpful AI flags?
2. **Real clinical outcomes:** Does AI catch *clinically dangerous* cancers earlier (lowering interval cancers between screens), or just trigger extra biopsies on harmless findings?
3. **Workload & safety:** Does reducing double-reading to AI-assisted single-reading save radiologist hours without harming patients?

Only a prospective randomized trial can answer those real-world questions!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- Digital medical scans are 2D/3D grids of brightness numbers; flattening destroys spatial neighborhood structure.
- A **2D convolution** slides a tiny shared **kernel** across the image to detect local edges, textures, and blobs regardless of position.
- **CNNs** learn hierarchies of convolutional kernels via gradient descent; **Vision Transformers** split scans into patch tokens linked by self-attention.
- Landmark trials like **MASAI** show medical-imaging AI can assist clinicians safely — when rigorously tested in real workflows.
```

*Next up:* High accuracy on medical images has a dark side! What if a CNN gets 97% accuracy **without even looking at the patient's lungs**? Prepare to catch a cheating image classifier red-handed in [](3_4-shortcuts-and-failure-modes.md).
