---
title: 'Chapter 4.2 — Comparing Sequences: From Hamming to Edit Distance'
short_title: '4.2 · Comparing sequences'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.2 — Comparing Sequences: From Hamming to Edit Distance

*Week 10 · ~40 minutes · Prerequisites: [](4_1-dna-and-proteins-as-text.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- compute **Hamming distance** between equal-length sequences and explain why a single insertion or deletion fools it;
- fill out a small **edit-distance** table by hand using **dynamic programming**;
- write a clean edit-distance function in Python and read an optimal **sequence alignment**;
- explain why high sequence similarity is strong evidence of shared ancestry and biological function.
```

## Start with a case: matching an unknown virus

When a new virus appears in a hospital, the very first thing scientists do is read its genetic letters and compare them against a library of known viruses. If two sequences are nearly identical, they almost certainly share a recent ancestor — and behave in similar ways!

So we need a fair way to measure: **how different are two strings of letters?**

## Attempt 1: Hamming distance (line them up and count mismatches)

The simplest rule is {term}`Hamming distance`: lay two equal-length sequences on top of each other and count how many columns disagree.
- `"ACGT"` vs. `"ACGA"` → only the last letter differs → **Hamming distance = `1`**.

**Now watch it break!** Suppose the first letter `M` gets snipped off `"MVHLTPEE"` and a `K` gets added at the end, giving `"VHLTPEEK"` — really just **2 tiny edits**! Line them up rigidly:

```text
M V H L T P E E
V H L T P E E K
✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓   --> 7 out of 8 columns look wrong!!
```

Just like a frameshift in Chapter 4.1, shifting the letters by one spot tricked Hamming distance into thinking two near-twins are total strangers!

## Attempt 2: Edit distance (inserting gaps so letters realign!)

What we really want is {term}`Edit distance` — the **fewest** single-letter edits (**swap**, **delete**, or **insert**) needed to turn sequence `A` into sequence `B`!

How can a computer find the fewest edits without testing billions of combinations? By building a tiny grid of smaller sub-answers ({term}`Dynamic programming`)!

### Fill a tiny `3 × 3` grid by hand first (`"CAT"` → `"ACT"`)!

Each cell `(row, col)` holds the edit distance between the first `row` letters of `"CAT"` and the first `col` letters of `"ACT"`:
- Start the top row and left column with `0, 1, 2, 3` (cost of building from an empty string).
- For every inside box:
  - If the two letters **match**, copy the diagonal top-left neighbor (**free: +0**!).
  - Otherwise, take **`1 + smallest`** of its three neighbors (**up** = delete, **left** = insert, **diagonal** = swap)!

| | `∅` | `A` | `C` | `T` |
|---|---|---|---|---|
| **`∅`** | **0** | 1 | 2 | 3 |
| **`C`** | 1 | 1 *(swap C→A)* | **1** *(C=C, copy diag 1!)* | 2 |
| **`A`** | 2 | **1** *(A=A, copy diag 1!)* | 2 | 3 |
| **`T`** | 3 | 2 | 3 | **2** *(T=T, copy diag 2!)* |

The bottom-right corner says **`2`**: delete `C` from the front and insert `C` after `A` (`-ACT` vs. `CA-T`)!

## Running edit distance on hemoglobin in Python!

```{admonition} Predict before you run
:class: tip
Compare `"MVHLTPEE"` vs. `"VHLTPEEK"` again! Remember Hamming distance was fooled and said **`7`**. What will our dynamic-programming `edit_distance` function print in the bottom-right corner?
```

```{code-cell} python
:label: ch42-edit-dist
import numpy as np
import matplotlib.pyplot as plt

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def edit_table(a, b):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        D[i, 0] = i
    for j in range(m + 1):
        D[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                D[i, j] = D[i - 1, j - 1]                         # match: free!
            else:
                D[i, j] = 1 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D

seq_a = "MVHLTPEE"
seq_b = "VHLTPEEK"
D = edit_table(seq_a, seq_b)

print(f"Hamming distance (fooled by shift!) : {hamming(seq_a, seq_b)}")
print(f"Edit distance (bottom-right corner) : {D[-1, -1]}")

# Visualize the filled table — notice the low-cost diagonal lane!
fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.imshow(D, cmap="Blues")
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        ax.text(j, i, str(D[i, j]), ha="center", va="center", fontsize=10,
                color="white" if D[i, j] > 4 else "#111111", fontweight="bold")
ax.set_xticks(range(len(seq_b) + 1), ["∅"] + list(seq_b))
ax.set_yticks(range(len(seq_a) + 1), ["∅"] + list(seq_a))
ax.set_xlabel("Sequence B (VHLTPEEK)")
ax.set_ylabel("Sequence A (MVHLTPEE)")
ax.set_title("Edit-distance table: follow the pale diagonal of 1s!")
fig.tight_layout()
plt.show()
```

Follow that pale stripe of **`1`**s running diagonally down the grid! Moving down once (`delete M`), cruising diagonally for seven free matches (`VHLTPEE`), and stepping right once (`insert K`) gives the optimal {term}`Alignment`:

```text
A: M V H L T P E E -
B: - V H L T P E E K
```

:::{margin}
Real search engines like **BLAST** (which biologists use millions of times a week to search every known genome!) are fast cousins of this exact grid — and even your phone's spellchecker runs edit distance behind the scenes!
:::

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.2.1 (Modify) — Classic spellchecker test
:class: hint
Predict by hand the edit distance between `"KITTEN"` and `"SITTING"`, then confirm it with `edit_table("KITTEN", "SITTING")[-1, -1]`.
```

```{admonition} Solution
:class: dropdown
```python
print("Edit distance (KITTEN -> SITTING):", edit_table("KITTEN", "SITTING")[-1, -1])
```
**`3`** edits: swap `K → S`, swap `E → I`, and insert `G` at the end!
```

```{admonition} Exercise 4.2.2 (Make) — Which unknown sequence is closest to human hemoglobin?
:class: hint
Compare `ref = "MVHLTPEEKS"` against three candidates:
- `"MVHLTPVEKS"` (sickle-cell variant)
- `"VHLTPEEKS"` (missing start `M`)
- `"QWERTYUIOP"` (unrelated)

Write a 3-line `for` loop printing each candidate's edit distance from `ref`.
```

```{admonition} Solution
:class: dropdown
```python
ref = "MVHLTPEEKS"
for name, seq in [("Sickle variant", "MVHLTPVEKS"), ("Missing M", "VHLTPEEKS"), ("Unrelated", "QWERTYUIOP")]:
    print(f"{name:<15}: edit distance = {edit_table(ref, seq)[-1, -1]}")
```
Both real hemoglobin relatives sit just **1 edit** away, while the unrelated string sits **10 edits** away!
```

```{admonition} Exercise 4.2.3 (Critical Reasoning) — Why isn't edit distance enough to predict if a mutation is dangerous?
:class: hint
Both the dangerous sickle-cell mutation (`E → V`) and a totally harmless mutation elsewhere in hemoglobin sit at **edit distance = `1`** from normal human hemoglobin! Why can't plain edit distance tell them apart?
```

```{admonition} Solution
:class: dropdown
Edit distance treats *every* position and *every* letter swap identically (cost = `1`). It doesn't know which spots are vital to a protein's 3D folded shape or which amino acids have opposite chemical charges! Learning that deeper evolutionary context is why scientists built **protein language models** (Chapters 4.3–4.6).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Hamming distance** only counts mismatches at fixed positions and gets fooled by a single insertion or deletion.
- **Edit distance** finds the minimum swaps, insertions, and deletions via a small **dynamic-programming** table, producing an **alignment**.
- Close sequence alignment is strong evidence of shared ancestry — and sets the stage for neural sequence models.
```

*Next up:* Instead of just comparing two strings letter by letter, how do modern **language models** learn the *patterns* of text — or proteins — well enough to generate new sentences? Find out in [](4_3-predicting-the-next-word.md)!
