---
title: 'Chapter D2 — Comparing sequences'
short_title: 'D2 · Comparing sequences'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter D2 — Comparing sequences

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/5x5x5x5/taihls/HEAD)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- measure how different two equal-length sequences are with the **Hamming distance**;
- explain why insertions and deletions need a smarter measure;
- build a tiny **edit distance** (Levenshtein) table from scratch;
- connect sequence similarity to **mutations**, shared ancestry, and shared function.
```

## Start with a story

In [](1_d1-dna-as-text.md) we learned to read one sequence. But biology's real power
comes from *comparing* sequences. When researchers found that a human gene was nearly
identical to a yeast gene, they could guess its job in us from what it does in yeast.
When a new virus appears, the first thing scientists do is compare its sequence to
known ones — a close match tells you what you are dealing with.

So the central question becomes: **how different are two sequences?** We need to turn
"these look similar" into a number. Let's start with the simplest possible case.

## Counting the differences: Hamming distance

Suppose two sequences are the *same length* and we line them up letter by letter. The
simplest measure of difference just counts how many positions disagree.

```{admonition} Definition — Hamming distance
:class: important
The **Hamming distance** between two equal-length sequences is the number of
positions where the letters differ. Identical sequences have distance 0; the more
disagreements, the larger the distance.
```

A **mutation** is a change in a sequence — and a single-letter swap is the most
common kind. Hamming distance is, in effect, a count of how many such swaps separate
two sequences.

```{code-cell} python
:label: hamming
def hamming(a, b):
    if len(a) != len(b):
        raise ValueError("Hamming distance needs equal-length sequences")
    return sum(x != y for x, y in zip(a, b))

seq1 = "ACGTACGT"
seq2 = "ACGAACGT"   # one letter changed
print(f"Hamming distance: {hamming(seq1, seq2)}")
```

```{admonition} Definition — mutation
:class: important
A **mutation** is a change to a DNA sequence. The simplest is a **substitution**
(one base replaced by another); sequences can also gain a base (**insertion**) or
lose one (**deletion**).
```

Just one position differs, so the distance is 1. Hamming distance is fast and
intuitive — but it has a fatal weakness, which we can see immediately.

## When letters shift: the trouble with insertions

Consider `ACGTACGT` and `CGTACGT`. To our eyes the second is just the first with its
leading `A` chopped off — barely different at all. But line them up position by
position and *almost every column disagrees*, because the missing letter shoves
everything one step out of register.

```{code-cell} python
:label: hamming-fail
a = "ACGTACGT"
b = "CGTACGT"          # the same string with its leading A deleted
# Pad b to equal length just so Hamming will run, then compare:
print(f"Hamming after a deletion: {hamming(a, b + '-')}")
```

A single deletion produces the *maximum possible* Hamming distance (8 out of 8) —
the measure badly overstates
how different these sequences really are. We need a measure that understands
*insertions and deletions*, not just substitutions.

## A smarter measure: edit distance

```{admonition} Definition — edit distance (Levenshtein)
:class: important
The **edit distance** between two sequences is the *smallest number of single-letter
edits* — insertions, deletions, or substitutions — needed to turn one into the other.
This version, allowing all three operations, is called the **Levenshtein distance**.
```

How can a computer find the *smallest* number of edits without trying every
possibility? The trick is to build up answers for tiny prefixes and reuse them — a
strategy called **dynamic programming**. We fill a small table where entry $(i, j)$
holds the edit distance between the first $i$ letters of one word and the first $j$
letters of the other.

The rule for each cell is short. If the two current letters match, copy the diagonal
neighbour (no edit needed). Otherwise, take the cheapest of the three edits:

$$ D[i][j] = 1 + \min\big(D[i-1][j],\; D[i][j-1],\; D[i-1][j-1]\big) $$ (eq-edit)

where the three terms stand for a deletion, an insertion, and a substitution. Let's
build the table for two short words.

```{code-cell} python
:label: edit-distance
import numpy as np

def edit_distance(a, b):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)      # cost of deleting all of a's prefix
    D[0, :] = np.arange(m + 1)      # cost of inserting all of b's prefix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                D[i, j] = D[i - 1, j - 1]            # letters match: free
            else:
                D[i, j] = 1 + min(D[i - 1, j],       # deletion
                                  D[i, j - 1],       # insertion
                                  D[i - 1, j - 1])   # substitution
    return D

a, b = "CATG", "ACTG"
D = edit_distance(a, b)
print(D)
print(f"\nEdit distance between {a!r} and {b!r}: {D[-1, -1]}")
```

The number in the bottom-right corner, `2`, is the answer: we can turn `CATG` into
`ACTG` in two edits (insert an `A` at the front, then delete the original `A`). The
whole table is just those two travel directions — down for a deletion, right for an
insertion, diagonal for a match-or-substitution — with [](#eq-edit) choosing the
cheapest route to every corner.

```{admonition} Definition — alignment
:class: important
An **alignment** is a way of writing two sequences one above the other, inserting gap
symbols (`-`) so that matching letters line up. The edit distance is the cost of the
*best* alignment — the one needing the fewest gaps and mismatches.
```

## Why this matters: similarity is evidence

Edit distance lets us put a single number on how related two sequences are. That
number is biological gold. Let's compare one reference sequence against three others.

```{code-cell} python
:label: compare-many
reference = "ACGTGGCATG"
candidates = {
    "close relative": "ACGTGGCATC",   # one substitution
    "lost a base":    "ACGTGCATG",    # one deletion
    "unrelated":      "TTACCAGGTT",   # scrambled
}

for name, seq in candidates.items():
    d = edit_distance(reference, seq)[-1, -1]
    print(f"{name:>15}: edit distance {d}")
```

The "close relative" and the "lost a base" sequences sit a single edit away from the
reference, while the unrelated one is many edits off. This is the logic behind
comparing genes across species: two sequences that are *very* similar almost never
arrived there by chance. They far more likely share a common ancestor — and because
their letters are alike, they tend to fold into similar shapes and do similar jobs.
**Similarity is evidence of shared ancestry and shared function.**

That single idea — *related things have similar representations* — runs straight
through modern AI, and we'll see it return when we meet learned representations of
proteins in [](5_d5-foundation-models-bio.md).

## Exercises

```{admonition} Exercise D2.1 — Hamming by hand
:class: hint
What is the Hamming distance between `AAGGCC` and `ATGGCA`? Work it out by eye, then
check with the `hamming` function.
```

```{admonition} Solution
:class: dropdown
Compare position by position: positions 2 (`A` vs `T`) and 6 (`C` vs `A`) differ, so
the distance is **2**.

```python
print(hamming("AAGGCC", "ATGGCA"))   # → 2
```
```

```{admonition} Exercise D2.2 — Predict the edit distance
:class: hint
Before running any code, predict the edit distance between `"KITTEN"` and `"SITTING"`
— the classic textbook example. Then confirm it with `edit_distance`.

*Hint: change `K`→`S`, change `E`→`I`, and add a `G` at the end.*
```

```{admonition} Solution
:class: dropdown
Three edits are needed (two substitutions and one insertion), so the edit distance is
**3**.

```python
print(edit_distance("KITTEN", "SITTING")[-1, -1])   # → 3
```

Our edit-distance code works on any strings, not just DNA — the same dynamic-
programming table that compares genes also powers spell-checkers and "did you mean…?"
search suggestions.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- **Hamming distance** counts mismatched positions but only works for equal-length
  sequences and is fooled by a single insertion or deletion.
- **Edit (Levenshtein) distance** counts the fewest insertions, deletions, and
  substitutions — found efficiently by filling a small **dynamic-programming** table.
- The bottom-right cell of that table is the answer; tracing back through it gives an
  **alignment**.
- High similarity is strong evidence of **shared ancestry and function** — the
  reason sequence comparison is everywhere in biology.
```
