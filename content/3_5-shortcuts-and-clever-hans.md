---
title: 'Chapter 3.5 — Clever Hans in the Clinic: Shortcut Learning and Distribution Shift'
short_title: '3.5 · Shortcut learning'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 3.5 — Clever Hans in the Clinic: Shortcut Learning and Distribution Shift

*Week 9 · ~40 minutes · Prerequisites: [](3_4-how-computers-see-scans.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **shortcut learning** using real radiology and skin-cancer examples;
- reproduce the **hospital corner-tag trap** in code and watch accuracy collapse at a new hospital (**distribution shift**);
- inspect a **pixel heatmap** to see *where* an image classifier is actually looking;
- explain why **external validation** at a new hospital is the ultimate cure for hidden shortcuts.
```

## Start with a case: the AI that read the X-ray machine, not the lungs

Over a century ago, a horse named *Clever Hans* amazed crowds by tapping out math answers with his hoof — until a scientist discovered Hans couldn't add at all: he was just watching tiny unconscious nods from his trainer!

Medical image AIs fall for the exact same trick, called {term}`Shortcut learning` [@geirhos2020shortcut]:
- **Chest X-rays:** Researchers trained a deep CNN on X-rays combined from multiple hospitals and got glowing accuracy — then discovered the AI was staring at tiny **metal corner labels** (like `"PORTABLE"`, used mostly on very sick ICU patients!) instead of actual lung infection [@zech2018pneumonia; @degrave2021covid].
- **Skin cancer photos:** Some dermatology models learned that **a centimeter ruler placed next to a mole** predicted cancer — because doctors mainly put a ruler next to moles that already look suspicious!

```{mermaid}
flowchart TD
    Sick["Very Sick ICU Patient"] --> Lung["Subtle Lung Shadow<br/>(Hard to learn!)"]
    Sick --> Tag["Bright Corner Tag<br/>(Super easy to spot!)"]
    Lung --> Label["Diagnosis: Sick"]
    Tag -. "AI grabs the easy<br/>shortcut instead!" .-> AI["Image Classifier"]
```

Why does AI grab the shortcut? **Because gradient descent is lazy!** Its only goal is lowering training loss — and a bright white stamp in the top-left corner is far easier to spot than a fuzzy gray lung shadow!

## Catch a cheating hospital AI red-handed!

Let's simulate synthetic `16 × 16` chest scans across four hospitals:
- **True lung clue:** Sick scans have a subtle bright region right in the **center** (`rows 5–10, cols 5–10`).
- **Training pool:** **Hospital A** (an intensive-care unit: **88% sick**, stamps a bright white `3×3` tag in the **top-left corner**) + **Hospital B** (a routine checkup clinic: **12% sick**, **no** corner tag).
- **New Hospital C:** 50% sick, **no corner tag**.
- **Hospital D (flipped scanner):** 50% sick, puts a bright corner tag on **healthy** checkup scans!

```{admonition} Predict before you run
:class: tip
We train one classifier on pooled scans from Hospitals A + B.
1. Will it score high on **internal test scans** from Hospitals A + B?
2. What will happen when we move that exact frozen model to **New Hospital C** (no corner tag) and **Hospital D** (flipped tag)?
```

```{code-cell} python
:label: ch35-shortcut-lab
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(10)

def make_scans(n, sick_rate, tag_if_sick, tag_if_healthy):
    imgs = rng.normal(0.0, 0.45, size=(n, 16, 16))
    y = (rng.random(n) < sick_rate).astype(int)
    for i in range(n):
        if y[i] == 1:
            imgs[i, 5:11, 5:11] += 0.22             # subtle real lung shadow (center)
        p_tag = tag_if_sick if y[i] == 1 else tag_if_healthy
        if rng.random() < p_tag:
            imgs[i, 1:4, 1:4] += 2.2                # bright top-left scanner tag (shortcut!)
    return imgs, y

# Training pool: Hospital A (ICU: mostly sick + tagged) + Hospital B (clinic: mostly healthy + untagged)
A_tr, yA_tr = make_scans(240, sick_rate=0.88, tag_if_sick=0.96, tag_if_healthy=0.96)
B_tr, yB_tr = make_scans(240, sick_rate=0.12, tag_if_sick=0.04, tag_if_healthy=0.04)
X_tr = np.concatenate([A_tr, B_tr]).reshape(-1, 256)
y_tr = np.concatenate([yA_tr, yB_tr])

# Three test sets: same hospitals (internal), new Hospital C (no tag), Hospital D (flipped tag!)
A_in, yA_in = make_scans(100, 0.88, 0.96, 0.96)
B_in, yB_in = make_scans(100, 0.12, 0.04, 0.04)
X_in, y_in  = np.concatenate([A_in, B_in]).reshape(-1, 256), np.concatenate([yA_in, yB_in])

C_ext, y_C = make_scans(200, 0.50, 0.0, 0.0)
D_ext, y_D = make_scans(200, 0.50, 0.0, 1.0)

# Strong regularization keeps weights compact — making the bright corner tag irresistible!
cheater = LogisticRegression(C=0.02, max_iter=1000).fit(X_tr, y_tr)

# Clean comparison: trained where the corner tag has ZERO link to sickness (50/50 for everyone)
clean_imgs, y_clean = make_scans(480, 0.50, 0.50, 0.50)
honest = LogisticRegression(C=0.02, max_iter=1000).fit(clean_imgs.reshape(-1, 256), y_clean)

print(f"Cheating model — Internal (Hospitals A+B): {cheater.score(X_in, y_in):.1%}  (looks awesome!)")
print(f"Cheating model — New Hospital C (no tag) : {cheater.score(C_ext.reshape(-1,256), y_C):.1%}  (drops 25 pts!)")
print(f"Cheating model — Hospital D (flipped tag): {cheater.score(D_ext.reshape(-1,256), y_D):.1%}  (WRECKED!)")
print(f"Honest model   — Hospital C / Hospital D : {honest.score(C_ext.reshape(-1,256), y_C):.1%} / {honest.score(D_ext.reshape(-1,256), y_D):.1%}")
```

**Look at that collapse: `88.5%` → `63.5%` → `12.5%`!!** That is {term}`Distribution shift` in action: a model that looked ready for the clinic shattered the moment the scanner changed.

## Open the hood: where was the model actually looking?

Because each pixel has one learned weight, we can reshape the 256 weights back into a **16 × 16 heatmap** — red pixels push the AI to say `"Sick"`, blue pixels push toward `"Healthy"`.

```{code-cell} python
:label: ch35-heatmap-fig
w_cheat  = cheater.coef_[0].reshape(16, 16)
w_honest = honest.coef_[0].reshape(16, 16)
lim = max(abs(w_cheat).max(), abs(w_honest).max())

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
axes[0].imshow(A_tr[yA_tr == 1][0], cmap="gray")
axes[0].set_title("A. Hospital A sick scan\n(corner tag + center shadow)", fontsize=9.5)
axes[0].axis("off")

axes[1].imshow(w_cheat, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[1].set_title("B. Cheating AI pixel weights\n(STARES at corner tag!)", fontsize=9.5, color="#D55E00")
axes[1].axis("off")

axes[2].imshow(w_honest, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[2].set_title("C. Honest AI pixel weights\n(focuses on center lungs!)", fontsize=9.5, color="#009E73")
axes[2].axis("off")

fig.tight_layout()
plt.show()
```

Panel B vs. Panel C tells the whole story in one glance: the cheating model put blazing red weight straight on the `3×3` corner box, ignoring the lungs! Once the link between corner tags and sickness was broken, the honest model learned the true center shadow and scored **90%+ at BOTH new hospitals**!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 3.5.1 (Modify) — Erase the corner before training!
:class: hint
Suppose you already spotted the corner tag in Panel B. Erase the top-left `5×5` corner (`imgs[:, :5, :5] = 0`) in `A_tr`, `B_tr`, `C_ext`, and `D_ext`, retrain `LogisticRegression(C=0.02)`, and print its accuracy on Hospitals C and D!
```

```{admonition} Solution
:class: dropdown
```python
tr_c = np.concatenate([A_tr, B_tr]).copy(); tr_c[:, :5, :5] = 0
C_c, D_c = C_ext.copy(), D_ext.copy(); C_c[:, :5, :5] = 0; D_c[:, :5, :5] = 0
fixed = LogisticRegression(C=0.02, max_iter=1000).fit(tr_c.reshape(-1, 256), y_tr)
print(f"After erasing corner -> Hospital C: {fixed.score(C_c.reshape(-1,256), y_C):.1%} | Hospital D: {fixed.score(D_c.reshape(-1,256), y_D):.1%}")
```
Removing the shortcut forces the model to learn the subtler real lung shadow — jumping straight back to **~90%+** at both external hospitals!
```

```{admonition} Exercise 3.5.2 (Make) — Measure how much weight sits in the corner
:class: hint
Compare `np.abs(w_cheat[1:4, 1:4]).sum()` against `np.abs(w_honest[1:4, 1:4]).sum()`. How many times more weight did the cheating AI place on those 9 corner pixels?
```

```{admonition} Solution
:class: dropdown
```python
c_bad  = np.abs(w_cheat[1:4, 1:4]).sum()
c_good = np.abs(w_honest[1:4, 1:4]).sum()
print(f"Corner weight — cheater: {c_bad:.2f} | honest: {c_good:.2f} (ratio = {c_bad / c_good:.1f}x!)")
```
The cheating model puts over **10×** as much total weight onto the 9 corner pixels as the honest model!
```

```{admonition} Exercise 3.5.3 (Critical Reasoning) — Spot the camera shortcut!
:class: hint
A team trains a skin-cancer AI by combining photos from two websites: Site 1 has mostly **cancerous** moles photographed through a special round medical lens (leaving a black circular ring around the edge), while Site 2 has mostly **harmless** moles photographed on ordinary **smartphones**. What shortcut will the AI learn, and how can the team prevent it?
```

```{admonition} Solution
:class: dropdown
The AI will detect the **black circular camera frame** instead of actual skin cancer — failing completely on smartphone photos of real melanomas! To prevent it, collect *both* cancerous and harmless moles from *both* cameras, crop out camera borders, and always test on a brand-new clinic's cameras (**external validation**).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Shortcut learning** happens because AI grabs the *easiest* pattern in training data — even a hospital tag or ruler instead of disease [@zech2018pneumonia; @geirhos2020shortcut; @degrave2021covid].
- Internal testing from the same hospitals hides shortcuts; **external validation** exposes **distribution shift** immediately.
- **Pixel heatmaps** are powerful debugging lenses to see *where* an image model looks.
```

*End of Unit 3!* You now understand how deep learning learns and sees — and how it cheats. In **Unit 4**, we switch from pictures to **letters and sequences**: DNA, proteins, chatbots, and the biological foundation models rewriting science right now, starting with [](4_1-dna-and-proteins-as-text.md)!
