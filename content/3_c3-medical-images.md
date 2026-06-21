---
title: 'Chapter C3 — Reading medical images'
short_title: 'C3 · Medical images'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter C3 — Reading medical images

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/c3-medical-images-cnn.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why a digital image is just a **grid of numbers** a model can read;
- display small images and the array of pixel values behind them;
- train a simple classifier on images and report its **accuracy**;
- describe how real medical-image AI works and where it dangerously fails.
```

## Start with a story

A radiologist scans a chest X-ray for signs of pneumonia; a dermatologist studies a photo
of a mole for signs of melanoma. These are some of medicine's most visual tasks — and some
of the first where AI matched specialist performance [@esteva2017skin]. But a computer has
no eyes. So how can it "look" at a medical image at all?

The key realization is almost anticlimactic: to a computer, an image is **not a picture —
it is a table of numbers**. Once we see that, every tool from the earlier chapters
suddenly applies to images too.

→ Train a real convolutional network on chest X-rays in the [companion notebook](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/c3-medical-images-cnn.ipynb).

## An image is a grid of numbers

```{admonition} Definition — pixel
:class: important
A **pixel** is one tiny dot of an image. A grayscale image is a grid of pixels, and each
pixel is simply a number for its brightness — small for dark, large for light. Stack those
numbers in rows and columns and you have the whole image as a table.
```

We'll use `load_digits`, a built-in set of tiny 8×8 grayscale images of handwritten
digits. They aren't medical scans, but they make the same point with no download and no
fuss: each image is an 8×8 grid of brightness numbers.

```{code-cell} python
:label: load-images
from sklearn.datasets import load_digits

digits = load_digits()
print(f"{len(digits.images)} images, each {digits.images[0].shape[0]}×"
      f"{digits.images[0].shape[1]} pixels")

# The very first image, printed as the grid of numbers it really is:
print("\nImage #0 as a table of brightness values:")
print(digits.images[0].astype(int))
print(f"\nThis image is labeled: {digits.target[0]}")
```

That block of numbers *is* the image. High values trace the bright strokes of the digit;
zeros are the dark background. Let's confirm by drawing a few of them.

```{code-cell} python
:label: show-images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 6, figsize=(9, 2))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.imshow(image, cmap="gray_r")          # draw the number grid as pixels
    ax.set_title(str(label))
    ax.axis("off")
fig
```

Each picture in [](#show-images) is the same kind of table we printed above, just colored
by brightness. A model never sees the picture — it sees the 64 numbers.

## Training a classifier on images

If an image is just 64 numbers, then classifying images is the same problem we already
solved: take the numbers in, predict a category out. We'll flatten each 8×8 grid into a row
of 64 features and train **logistic regression**, a workhorse linear classifier.

```{admonition} Definition — logistic regression
:class: note
**Logistic regression** is a classifier that weights each input, adds them up, and squashes
the total with a sigmoid (the same bend from [](2_c2-what-is-a-neural-net.md)) to produce a
probability for each class. Think of it as a one-neuron network with no hidden layer.
```

```{code-cell} python
:label: train-clf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = digits.data            # each row is one image flattened to 64 numbers
y = digits.target          # the true digit, 0–9

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"Correct on {accuracy:.1%} of held-out images.")
```

Over ninety-five percent correct, from images treated as nothing but rows of numbers. Let's
look at where it slips by checking a handful of test images against their predictions.

```{code-cell} python
:label: predictions-fig
predictions = clf.predict(X_test)

fig, axes = plt.subplots(2, 6, figsize=(9, 3.5))
for ax, image, true, pred in zip(
        axes.ravel(), X_test, y_test, predictions):
    ax.imshow(image.reshape(8, 8), cmap="gray_r")
    ok = "✓" if true == pred else "✗"
    ax.set_title(f"{pred} {ok}", fontsize=9,
                 color="#16a085" if true == pred else "#c0392b")
    ax.axis("off")
fig
```

Most predictions in [](#predictions-fig) are correct (green ✓); the occasional miss (red ✗)
tends to be a genuinely ambiguous scrawl — the same digits a person might hesitate over.

## From toy digits to real medical scans

Real medical images are far larger and richer than an 8×8 grid — a chest X-ray might be
thousands of pixels on a side. Flattening such an image into a plain list of numbers throws
away a crucial fact: that **nearby pixels belong together**, forming edges, textures, and
shapes. The models that read medical images well are **convolutional neural networks**.

```{admonition} Definition — convolutional neural network (CNN)
:class: important
A **convolutional neural network** is a neural network built for images. Instead of one
weight per pixel, it slides small windows across the image to detect local patterns — an
edge here, a speckle there — and stacks these into ever more abstract features, much as the
hidden neurons of [](2_c2-what-is-a-neural-net.md) build on one another. This respect for
local structure is why CNNs read scans so much better than a flattened classifier.
```

CNNs of this kind have matched dermatologists at spotting skin cancer from photographs
[@esteva2017skin] and now assist radiologists across many tasks [@rajpurkar2022ai]. The
companion notebook trains a small CNN on real chest X-rays so you can see one work
end-to-end.

## When image AI fails

High accuracy on a test set is not a clean bill of health. Medical-image models fail in
ways that are easy to miss and dangerous in the clinic:

- **They learn the wrong cue.** A model meant to detect disease may instead latch onto a
  ruler a clinician placed beside skin lesions, or a hospital's scanner watermark — a
  shortcut that vanishes the moment the model meets a new hospital.
- **They break on data unlike their training.** A model trained on adult scans can be
  unreliable on children; one trained on one machine's images can stumble on another's.
- **They reflect who was in the training data.** If some skin tones or patient groups were
  scarce in training, accuracy can be far worse for them — a fairness problem we return to
  in [](1_e1-bias-in-medical-ai.md).
- **A confident wrong answer is still wrong.** Models report a probability, not certainty,
  and can be confidently mistaken — which is why a human stays in the loop
  ([](3_e3-humans-in-the-loop.md)).

```{admonition} Why test accuracy can mislead
:class: warning
A single accuracy number summarizes performance on data that *resembles the training set*.
The real question for a medical tool is how it behaves on patients, scanners, and clinics it
has **never seen** — and that almost always requires testing far beyond the original data.
```

## Exercises

```{admonition} Exercise C3.1 — Does a simpler model agree?
:class: hint
Swap the logistic-regression classifier for the k-nearest-neighbors idea from
[](1_b1-knn-tumor.md): use `from sklearn.neighbors import KNeighborsClassifier` with
`KNeighborsClassifier(n_neighbors=5)`. Train and score it on the same split. Is it better or
worse than logistic regression here?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"{knn.score(X_test, y_test):.1%}")   # ≈ 98–99%
```
k-NN actually edges out logistic regression on these tidy digits — nearby images in
pixel-space tend to be the same digit. On real, varied medical scans neither simple method
is enough, which is exactly why CNNs exist.
```

```{admonition} Exercise C3.2 — Which digits confuse it?
:class: hint
Using `from sklearn.metrics import confusion_matrix`, build the confusion matrix of
`y_test` vs `predictions`. Which pairs of digits does the model mix up most? Why might those
particular digits look alike as grids of numbers?
```

```{admonition} Solution
:class: dropdown
```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))
```
Off-diagonal entries are the mistakes. Common confusions involve digits whose strokes
overlap heavily as pixels — for example 8 with 1 or 9, or 3 with 8 — because their brightness
grids genuinely resemble one another. A model that sees only numbers will mix up whatever
*looks* numerically similar, which is exactly how the "ruler" and "watermark" shortcut
failures happen on real scans.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A digital image is a **grid of numbers** (pixels); once flattened, it is just a list of
  features a classifier can read.
- A simple classifier reaches high accuracy on tidy 8×8 digits — the same machinery from
  earlier chapters, applied to images.
- Real medical images use **convolutional neural networks**, which respect that nearby
  pixels form edges and shapes.
- High test accuracy can hide serious **failure modes** — wrong cues, unfamiliar data, and
  uneven performance across patient groups — so medical-image AI must be tested far beyond
  its training set and kept under human oversight.
```
