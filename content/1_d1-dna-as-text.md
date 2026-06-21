---
title: 'Chapter D1 — DNA and proteins as text'
short_title: 'D1 · DNA as text'
# kernelspec is required for code cells to execute.
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter D1 — DNA and proteins as text

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/5x5x5x5/taihls/HEAD)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- read a stretch of **DNA** as a string of four letters;
- compute the **GC content** of a sequence and say why biologists care;
- count **k-mers** — short repeated chunks — with `collections.Counter`;
- find a simple **motif** by scanning the sequence for a pattern.
```

## Start with a story

Inside almost every one of your cells sits a molecule about two metres long, coiled
up tighter than any headphone cable: your **DNA**. It carries the instructions to
build and run a human being. Astonishingly, those instructions are written in an
alphabet of just **four letters** — `A`, `C`, `G`, and `T`.

That last fact is the gateway to this whole part of the book. If life's instructions
are *text*, then the same tools we use to search, count, and compare text can be
pointed straight at biology. Let's name the pieces precisely.

```{admonition} Definition — sequence
:class: important
A **sequence** is an ordered string of letters. A DNA sequence is written left to
right, and the *order* matters: `AGCT` and `TCGA` mean completely different things,
just as "tea" and "eat" do.
```

```{admonition} Definition — nucleotide (base)
:class: important
Each letter in a DNA sequence stands for a **nucleotide**, also called a **base**.
The four DNA bases are adenine (`A`), cytosine (`C`), guanine (`G`), and thymine
(`T`). A base is the smallest unit of the DNA "text" — its single character.
```

## Our specimen: a short stretch of DNA

Real genomes are billions of bases long, but every idea here works the same on a
tiny example we can read with our own eyes. Here is a short made-up stretch.

```{code-cell} python
:label: dna-string
dna = "AGCTATAGGCGCATGCATGCAATAGGCATGCATGCGGCTATA"

print(f"The sequence is {len(dna)} bases long.")
print(f"First 10 bases: {dna[:10]}")
```

A string of 42 letters — that's all DNA is, at this level of description. Now let's
ask our first biological question of it.

## How much G and C? (GC content)

Not all bases are equally "sticky." A `G` always pairs with a `C`, and that pair is
held together more strongly than an `A`–`T` pair. So a region rich in `G` and `C` is
physically more stable. Biologists summarise this with one number.

```{admonition} Definition — GC content
:class: important
The **GC content** of a sequence is the fraction (or percentage) of its bases that
are `G` or `C`. High GC content means a more stable stretch of DNA; it also varies
between organisms, so it can hint at *where* a sequence came from.
```

Counting is the entire calculation — no biology lab required.

```{code-cell} python
:label: gc-content
gc = dna.count("G") + dna.count("C")
gc_fraction = gc / len(dna)

print(f"G + C bases: {gc}")
print(f"GC content: {gc_fraction:.1%}")
```

So exactly half of this sequence is `G` or `C`. If we screened thousands of
sequences and one had a wildly different GC content from its neighbours, that oddball
would be worth a closer look — a first, crude way to "see" something interesting in a
wall of letters.

## Chopping the text into k-mers

Single letters only tell us so much. Often the *short combinations* are what carry
meaning — like how "th" and "ing" are telltale chunks of English. In sequence
biology these chunks have a name.

```{admonition} Definition — k-mer
:class: important
A **k-mer** is a contiguous substring of length $k$. The 2-mers ("dimers") of
`AGCT` are `AG`, `GC`, and `CT` — every window of two letters, taken one step at a
time. Counting k-mers turns a sequence into a tally of its building blocks.
```

We slide a window of width $k$ along the sequence and record every chunk we see.
Python's `collections.Counter` was built for exactly this kind of tallying.

```{code-cell} python
:label: kmers
from collections import Counter

def count_kmers(seq, k):
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    return Counter(kmers)

dimers = count_kmers(dna, 2)
print("Five most common 2-mers:")
for kmer, n in dimers.most_common(5):
    print(f"  {kmer}: {n}")
```

`Counter.most_common` hands us the busiest chunks instantly. Notice `AT` and `GC`
near the top — that already tells us this sequence likes to alternate between those
partners. Let's make the picture visible.

```{code-cell} python
:label: kmer-fig
import matplotlib.pyplot as plt

top = dimers.most_common(8)
labels = [kmer for kmer, _ in top]
heights = [n for _, n in top]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, heights, color="#16a085")
ax.set_xlabel("2-mer")
ax.set_ylabel("count")
ax.set_title("Most common 2-mers in our sequence")
fig
```

The bar chart in [](#kmer-fig) is the sequence's "fingerprint" — a profile that
different genes, and different organisms, carry in different proportions. Comparing
such fingerprints is one practical way computers tell sequences apart.

## Finding a motif

Sometimes biologists hunt for one *specific* short pattern, because that exact
pattern does a job — it might be where a protein grabs the DNA to switch a gene on.

```{admonition} Definition — motif
:class: important
A **motif** is a short, meaningful pattern that recurs in sequences. Finding a motif
means locating every position where that pattern appears.
```

Let's search our sequence for the motif `CATGC` and report where it starts.

```{code-cell} python
:label: motif
motif = "CATGC"
positions = [i for i in range(len(dna) - len(motif) + 1)
             if dna[i:i + len(motif)] == motif]

print(f"Motif {motif!r} found at positions: {positions}")
print(f"It occurs {len(positions)} times.")
```

The motif turns up four times, clustered in the middle of the sequence. A real
motif search is just this idea scaled up — sometimes allowing a few mismatches, which
is exactly the kind of "almost-equal" comparison we tackle in
[](2_d2-comparing-sequences.md).

## Exercises

```{admonition} Exercise D1.1 — GC content by hand and by code
:class: hint
The sequence `GGGCCCATAT` is 10 bases long. First work out its GC content with pen
and paper, then confirm it in code.
```

```{admonition} Solution
:class: dropdown
Six of the ten bases (`GGGCCC`) are `G` or `C`, so the GC content is $6/10 = 60\%$.

```python
s = "GGGCCCATAT"
print((s.count("G") + s.count("C")) / len(s))   # → 0.6
```
```

```{admonition} Exercise D1.2 — Which 3-mer is most common?
:class: hint
Reuse `count_kmers` from above to count the **3-mers** of our `dna` sequence. Which
3-mer appears most often, and how many times?

*Hint: call `count_kmers(dna, 3).most_common(1)`.*
```

```{admonition} Solution
:class: dropdown
```python
print(count_kmers(dna, 3).most_common(1))   # → [('GCA', 5)]
```

The 3-mer `GCA` appears **5 times**. Just behind it, the 3-mer `ATG` appears 4 times
— worth noticing because `ATG` is biologically famous: it is the usual "start here"
signal that marks the beginning of a protein-coding region.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- DNA is a **sequence** — an ordered string over a four-letter alphabet of
  **nucleotides** (`A`, `C`, `G`, `T`).
- **GC content** is just the fraction of `G`/`C` bases; it measures stability and
  hints at a sequence's origin.
- Counting **k-mers** (with `collections.Counter`) turns a sequence into a
  fingerprint of its building blocks.
- A **motif** is a meaningful short pattern; finding one is a simple scan — the seed
  of the sequence comparison we do next.
```
