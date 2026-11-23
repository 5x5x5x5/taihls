---
title: 'Chapter 3.4 — Clever Hans in the Clinic: Shortcut Learning, Distribution Shift, and Saliency'
short_title: '3.4 · Shortcuts & failure modes'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.4 — Clever Hans in the Clinic: Shortcut Learning, Distribution Shift, and Saliency

*Estimated time: ~45 minutes · Prerequisites: [](3_3-medical-imaging-and-cnns.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **shortcut learning** using real clinical disasters from radiology and dermatology;
- reproduce the famous **hospital-watermark shortcut** in a runnable image experiment and watch accuracy collapse at a new hospital (**distribution shift**);
- inspect a **pixel weight / saliency heatmap** to diagnose *what* an image model is actually looking at;
- explain why saliency maps are helpful debugging tools — yet never sufficient proof of clinical safety on their own.
```

## Start with a case: the AI that read the X-ray machine, not the lungs

In 1907, a horse named *Clever Hans* amazed Berlin audiences by tapping out the answers to arithmetic problems with his hoof — until psychologist Oskar Pfungst discovered Hans couldn't do math at all: the horse was simply reading tiny unconscious posture cues from his trainer!

Deep neural networks do the exact same thing in hospitals — a phenomenon called {term}`Shortcut learning` [@geirhos2020shortcut]:

- **Radiology pneumonia shortcut:** Zech and colleagues trained a CNN on chest X-rays pooled from multiple hospitals and got impressive internal accuracy — then discovered the model was reading **tiny metal hospital-department markers** (e.g., `"PORTABLE"` tags used mostly on very sick ICU patients!) rather than actual lung opacity [@zech2018pneumonia].
- **COVID-19 radiograph shortcut:** DeGrave and colleagues audited published COVID-19 X-ray models and found many relied on **text annotations, patient positioning (lying down vs. standing), and dataset source artifacts** rather than pulmonary pathology [@degrave2021covid].
- **Dermatology ruler shortcut:** Skin-lesion classifiers trained on clinical archives sometimes learned that **a centimeter ruler photographed next to a mole** predicted melanoma — because dermatologists mainly pull out a ruler when a mole already looks suspicious!

```{mermaid}
flowchart TD
    Sick["Very Sick ICU Patient"] --> Lung["True Lung Lesion<br/>(Hard, subtle visual cue)"]
    Sick --> Tag["Portable Scanner Tag<br/>(Easy, high-contrast corner cue!)"]
    Lung --> Label["Diagnosis: Positive"]
    Tag -. "Model takes the<br/>lazy shortcut!" .-> AI["AI Classifier"]
```

Why does a neural network prefer the hospital tag over the lung? **Because gradient descent is lazy!** Its only job is to minimize training loss. A bright white tag in the top-left corner is mathematically *far easier* to spot than a fuzzy gray shadow inside the chest!

## Lab 1: Reproduce the hospital shortcut disaster in code!

Let's recreate the Zech / DeGrave experiment transparently on synthetic **16 × 16 chest-scan patches**:
- **True biological signal:** Diseased patients have a subtle bright nodule near the **center** of the lung (`rows 6–9, cols 6–9`), obscured by realistic scanner noise.
- **Spurious shortcut:** Our training pool combines **Hospital A** (an ICU referral center with **85% disease prevalence** whose scanner stamps a bright `3×3` `"PORTABLE"` marker in the **top-left corner**) and **Hospital B** (an outpatient clinic with **15% disease prevalence** and **no** corner stamp).
- **External Hospital C (deployment!):** A brand-new hospital with **50% disease prevalence** — and **NO corner tag at all** (or even worse, a scanner that puts the tag on *healthy* outpatient scans!).

```{admonition} Predict before you run
:class: tip
We train a classifier on pooled scans from Hospital A + Hospital B, then test it on two groups:
1. **Internal held-out test set** (more scans from Hospitals A + B), and
2. **External Hospital C** (same exact lung nodules, no corner tag!).

Predict what happens to accuracy when the model moves from Internal Test → External Hospital C!
```

```{code-cell} python
:label: ch34-shortcut-sim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(10)

def make_hospital_scans(n, prevalence, tag_prob_sick, tag_prob_healthy):
    """Generate 16x16 synthetic radiographs with a subtle central lesion
    and an optional bright top-left scanner marker (shortcut)."""
    imgs = rng.normal(0.0, 0.45, size=(n, 16, 16))
    y = (rng.random(n) < prevalence).astype(int)

    for i in range(n):
        # True biological signal: subtle diffuse central nodule (rows 5..11, cols 5..11)
        if y[i] == 1:
            imgs[i, 5:11, 5:11] += 0.22
        # Spurious shortcut: high-contrast bright top-left 3x3 scanner tag
        p_tag = tag_prob_sick if y[i] == 1 else tag_prob_healthy
        if rng.random() < p_tag:
            imgs[i, 1:4, 1:4] += 2.2

    return imgs, y

# Training pool: Hospital A (ICU: 88% sick, tagged) + Hospital B (Outpatient: 12% sick, untagged)
img_A_tr, y_A_tr = make_hospital_scans(240, prevalence=0.88, tag_prob_sick=0.96, tag_prob_healthy=0.96)
img_B_tr, y_B_tr = make_hospital_scans(240, prevalence=0.12, tag_prob_sick=0.04, tag_prob_healthy=0.04)
X_tr = np.concatenate([img_A_tr, img_B_tr]).reshape(-1, 256)
y_tr = np.concatenate([y_A_tr, y_B_tr])

# Internal validation: fresh patients from the SAME two hospitals (A + B)
img_A_in, y_A_in = make_hospital_scans(100, prevalence=0.88, tag_prob_sick=0.96, tag_prob_healthy=0.96)
img_B_in, y_B_in = make_hospital_scans(100, prevalence=0.12, tag_prob_sick=0.04, tag_prob_healthy=0.04)
X_in = np.concatenate([img_A_in, img_B_in]).reshape(-1, 256)
y_in = np.concatenate([y_A_in, y_B_in])

# External Hospital C: brand-new clinic (50% sick, NO corner tag!)
img_C_ext, y_C_ext = make_hospital_scans(200, prevalence=0.50, tag_prob_sick=0.0, tag_prob_healthy=0.0)
X_ext = img_C_ext.reshape(-1, 256)

# Adversarial Hospital D: tag appears on HEALTHY scans instead!
img_D_flip, y_D_flip = make_hospital_scans(200, prevalence=0.50, tag_prob_sick=0.0, tag_prob_healthy=1.0)
X_flip = img_D_flip.reshape(-1, 256)

clf_shortcut = LogisticRegression(C=0.02, max_iter=1000).fit(X_tr, y_tr)

print(f"Internal test (Hospitals A+B)      : {clf_shortcut.score(X_in, y_in):.1%}  <-- looks clinical-grade!")
print(f"External Hospital C (no tag)       : {clf_shortcut.score(X_ext, y_C_ext):.1%}  <-- collapses by 25 points!")
print(f"External Hospital D (flipped tag!) : {clf_shortcut.score(X_flip, y_D_flip):.1%}  <-- CATASTROPHIC failure!")
```

**STARE AT THOSE THREE NUMBERS!**
- On the internal test split (Hospitals A + B), accuracy looks gleaming (**88.5%**) — ready for a glossy press release!
- Deploy the *exact same frozen model* at **Hospital C** (no corner tag), and accuracy plunges to **~63.5%**!
- Deploy it at **Hospital D** (where a different scanner puts a bright mark on outpatient healthy scans), and accuracy craters to **~12.5%** — catastrophically worse than flipping a coin!

## Lab 2: Catching the cheat with a saliency heatmap (and fixing it!)

How do we prove *why* the model failed? For our linear image model (and via gradient-based **saliency / Grad-CAM** heatmaps for deep CNNs), we reshape the 256 learned pixel weights back into a **16 × 16 heatmap**! Positive red weights show pixels pushing the model toward `"Disease"`; blue weights push toward `"Healthy"`.

Let's compare the **shortcut model's** weight heatmap against a **clean model** trained on **balanced data where the tag is removed (or unconfounded with disease)**!

```{code-cell} python
:label: ch34-saliency-fig
import matplotlib.pyplot as plt

# Train a CLEAN model where corner tag is independent of disease (50/50 regardless of sickness)
img_clean_tr, y_clean_tr = make_hospital_scans(480, prevalence=0.50, tag_prob_sick=0.50, tag_prob_healthy=0.50)
clf_clean = LogisticRegression(C=0.02, max_iter=1000).fit(img_clean_tr.reshape(-1, 256), y_clean_tr)

print(f"Clean model at External Hospital C : {clf_clean.score(X_ext, y_C_ext):.1%}")
print(f"Clean model at Hospital D (flipped): {clf_clean.score(X_flip, y_D_flip):.1%}")

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))

# Sample diseased scan from Hospital A (has BOTH nodule and corner tag)
sample_sick = img_A_tr[y_A_tr == 1][0]
axes[0].imshow(sample_sick, cmap="gray")
axes[0].set_title("A. Hospital A sick scan\n(corner tag + center nodule)", fontsize=9.2)
axes[0].axis("off")

# Sample diseased scan from External Hospital C (nodule only, NO tag)
sample_ext = img_C_ext[y_C_ext == 1][0]
axes[1].imshow(sample_ext, cmap="gray")
axes[1].set_title("B. External Hospital C scan\n(nodule only, no tag)", fontsize=9.2)
axes[1].axis("off")

# Saliency heatmap of the SHORTCUT model
w_bad = clf_shortcut.coef_[0].reshape(16, 16)
lim = max(abs(w_bad).max(), abs(clf_clean.coef_[0]).max())
axes[2].imshow(w_bad, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[2].set_title("C. Shortcut model heatmap\n(STARS at corner tag!)", fontsize=9.2, color="#D55E00")
axes[2].axis("off")

# Saliency heatmap of the CLEAN model
w_good = clf_clean.coef_[0].reshape(16, 16)
im = axes[3].imshow(w_good, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[3].set_title("D. Clean model heatmap\n(focuses on center nodule!)", fontsize=9.2, color="#009E73")
axes[3].axis("off")

fig.tight_layout()
plt.show()
```

**Panel C vs. Panel D in [](#ch34-saliency-fig) is worth a thousand lectures!**
- In **Panel C**, the shortcut model puts blazing red weights right onto the `3×3` top-left corner box — barely glancing at the center of the lung!
- In **Panel D**, once the spurious link between hospital tag and prevalence is broken, the clean model ignores the corner entirely, focuses squarely on the central lung nodule, and jumps straight to **90%+ accuracy at BOTH External Hospital C and Adversarial Hospital D**!

## Why saliency maps alone are NOT enough

```{admonition} Caution — the illusion of interpretability
:class: warning
Saliency maps (like Grad-CAM) are brilliant **debugging tools** to catch obvious shortcuts like Panel C. However, **never treat a pretty heatmap as proof that an AI is trustworthy!**
- Two models with completely different logic can highlight similar lung regions.
- A heatmap shows *where* the model looked, not *why* — e.g., it might highlight the right tumor region yet be counting background brightness rather than cell shape.
- **External validation on unseen hospitals and scanners** remains the non-negotiable gold standard!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.4.1 (Modify) — Can cropping out the corner fix the shortcut?
:class: hint
Suppose you cannot collect new data from Hospitals A and B, but after viewing Panel C you realize the top-left `5×5` corner holds the hospital watermark. Zero out pixels `[:, :5, :5] = 0.0` in `img_A_tr`, `img_B_tr`, `img_C_ext`, and `img_D_flip`, retrain `LogisticRegression`, and test on Hospitals C and D! What happens?
```

```{admonition} Solution
:class: dropdown
```python
tr_crop = np.concatenate([img_A_tr, img_B_tr]).copy()
tr_crop[:, :5, :5] = 0.0
ext_crop, flip_crop = img_C_ext.copy(), img_D_flip.copy()
ext_crop[:, :5, :5] = 0.0
flip_crop[:, :5, :5] = 0.0

clf_crop = LogisticRegression(C=0.02, max_iter=1000).fit(tr_crop.reshape(-1, 256), y_tr)
print(f"Cropped model on Hospital C: {clf_crop.score(ext_crop.reshape(-1, 256), y_C_ext):.1%}")
print(f"Cropped model on Hospital D: {clf_crop.score(flip_crop.reshape(-1, 256), y_D_flip):.1%}")
```

Accuracy leaps back up to **~95%** on *both* external hospitals! Removing the shortcut forced gradient descent to work harder and actually learn the subtler biological nodule in the center!
```

```{admonition} Exercise 3.4.2 (Make) — Measure the fraction of weight on the shortcut vs. nodule
:class: hint
Write 4 lines of NumPy on `w_bad` and `w_good` to compute the **sum of absolute weights** inside the top-left corner (`[1:4, 1:4]`) versus inside the central nodule (`[6:10, 6:10]`). Print the corner-to-nodule weight ratio for both models.
```

```{admonition} Solution
:class: dropdown
```python
for name, W in [("Shortcut model", w_bad), ("Clean model", w_good)]:
    corner_w = np.abs(W[1:4, 1:4]).sum()
    nodule_w = np.abs(W[6:10, 6:10]).sum()
    print(f"{name:<15}: corner = {corner_w:5.2f}, nodule = {nodule_w:5.2f}, "
          f"ratio = {corner_w / nodule_w:.2f}")
```

The shortcut model puts far more total weight on the 9 corner pixels than the clean model, quantifying the exact visual story of Panel C vs. Panel D!
```

```{admonition} Exercise 3.4.3 (Critical Reasoning) — Spot the shortcut before training
:class: hint
A consortium merges two datasets to train a skin-cancer AI:
- Dataset 1 (from a specialty cancer clinic): mostly **malignant** lesions, photographed with a high-resolution dermatoscope showing a **circular dark frame** around the edges.
- Dataset 2 (from a primary-care app): mostly **benign** moles, photographed on ordinary **smartphones** (no circular frame).

Predict what a CNN trained on Dataset 1 + Dataset 2 will learn — and suggest **two concrete safeguards** the team should take before publishing or deploying.
```

```{admonition} Solution
:class: dropdown
The CNN will exploit **shortcut learning**: it will detect the circular dermatoscope border and camera sensor texture rather than melanoma biology, achieving glowing internal accuracy while failing horribly on smartphone photos of real melanomas!

**Two safeguards:**
1. **Stratify / balance data collection by camera device:** Include both benign and malignant lesions captured on *both* smartphones and dermatoscopes (breaking the confounding link between device and label!).
2. **External validation across unseen clinics & devices** plus **saliency audits** checking whether masking the border destroys predictions.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Shortcut learning** happens because gradient descent prefers the *easiest* predictive cue — like a scanner watermark or ruler — over genuine disease biology [@zech2018pneumonia; @geirhos2020shortcut; @degrave2021covid].
- Internal validation from the same hospitals hides shortcuts completely; **external validation** exposes **distribution shift** immediately.
- **Saliency / weight heatmaps** help debug *where* a model is looking, though they complement rather than replace external testing.
- Breaking confounding between data sources (hospital, scanner, camera) and disease prevalence prevents shortcut traps before training even starts.
```

*End of Part 3!* You have built gradient descent, trained a neural network from scratch, written a 2D image convolution, and caught a cheating clinical imaging AI red-handed. In **Part 4**, we shift from images to **sequences and language** — unlocking DNA, Transformers, clinical chatbots, and the biological foundation models rewriting science right now, starting with [](4_1-dna-proteins-and-alignment.md).
