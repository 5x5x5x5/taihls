---
title: 'Chapter D5 — Foundation models for biology'
short_title: 'D5 · Foundation models'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter D5 — Foundation models for biology

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/d5-protein-foundation-models.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **foundation model** is and what makes it different;
- describe at an intuitive level what AlphaFold, ESM, and scGPT each do;
- explain what an **embedding** is and why "similar things get similar vectors";
- build a toy embedding from amino-acid composition and read a similarity heatmap.
```

## Start with a story

For decades, each biology problem got its own special-purpose program: one tool to
predict protein shape, another to classify cell types, another to spot a gene. Each
was built from scratch for its one job.

Then something changed. Researchers found that if you train *one very large model* on
a vast amount of biological data — the way an LLM trains on text — that single model
learns representations so rich that it can be pointed at *many* tasks. This is the
**foundation model** idea, and it is reshaping biology right now.

```{admonition} Definition — foundation model
:class: important
A **foundation model** is a single large model, pretrained on huge amounts of data,
that serves as a reusable starting point for *many* downstream tasks — rather than
being built for just one. Train once on everything; adapt cheaply to each new problem.
```

## One model, many tasks: three landmark examples

The sequence skills from this whole part — reading sequences ([](1_d1-dna-as-text.md)),
comparing them ([](2_d2-comparing-sequences.md)), and next-token prediction
([](3_d3-language-models.md)) — come together here. Three foundation models show the
range.

```{admonition} AlphaFold — from sequence to 3-D shape
:class: note
A protein is a chain of amino acids that folds into a precise 3-D shape, and the
shape determines what the protein *does*. Predicting that shape from the sequence was
a 50-year-old grand challenge. **AlphaFold** [@jumper2021alphafold] cracked it,
predicting structures so accurately that it has now mapped almost every known protein
— work that used to take years of lab experiments per protein.
```

```{admonition} ESM — a language model for proteins
:class: note
If DNA and proteins are *text* (our theme since [](1_d1-dna-as-text.md)), why not
train a language model on them? **ESM** [@lin2023esm] does exactly that: trained on
hundreds of millions of protein sequences with next-token-style prediction, it learns
the "grammar" of proteins. From sequence alone it can flag which mutations are
harmful and even help predict structure — never having been explicitly taught either.
```

```{admonition} scGPT — a foundation model for cells
:class: note
**scGPT** [@cui2024scgpt] applies the same recipe to single-cell data: trained on
tens of millions of individual cells, it learns a general representation of cell
biology that can be adapted to identify cell types, predict responses to drugs, and
more — one model standing in for many specialized pipelines.
```

The common thread: **pretrain once on a flood of data, then adapt to many tasks.** And
the engine inside each is the same idea we'll explore now — turning a sequence into a
list of numbers that captures its meaning.

## The key idea: embeddings

When a foundation model "reads" a protein, it converts it into a list of numbers — a
vector — chosen so that *similar proteins get similar vectors*. That is the secret
sauce, and it is the same logic as the sequence distances in
[](2_d2-comparing-sequences.md), only learned rather than hand-coded.

```{admonition} Definition — embedding
:class: important
An **embedding** is a representation of something (a word, a protein, a cell) as a
list of numbers — a vector — arranged so that things with **similar meaning land close
together**. Comparing embeddings is how a model judges similarity.
```

Real embeddings come from huge neural networks. But we can build a *toy* embedding by
hand to feel how it works: represent each protein by its **amino-acid composition** —
simply, what fraction of it is each amino acid. Proteins made of similar ingredients
will get similar vectors, just as real embeddings put similar proteins nearby.

```{code-cell} python
:label: toy-embed
import numpy as np

# A few very short protein sequences (letters are amino acids).
proteins = {
    "P1": "AAGGCC",
    "P2": "AAGGCG",   # almost identical to P1
    "P3": "WWWYYF",   # completely different ingredients
    "P4": "WWYYFF",   # similar ingredients to P3
}

alphabet = "ACDEFGHIKLMNPQRSTVWY"   # the 20 standard amino acids

def composition(seq):
    """Toy embedding: fraction of the sequence that is each amino acid."""
    counts = np.array([seq.count(a) for a in alphabet], dtype=float)
    return counts / counts.sum()

vectors = {name: composition(seq) for name, seq in proteins.items()}
print("Each protein is now a 20-number vector. P1 starts:")
print(np.round(vectors["P1"], 2))
```

Each protein is now a point in a 20-dimensional space — a vector. To compare two of
them, we measure how close their vectors are; here we use the same straight-line
distance idea from k-nearest-neighbors in [](1_b1-knn-tumor.md), turned into a
similarity (close = similar = high).

```{code-cell} python
:label: similarity
names = list(proteins)
V = np.array([vectors[n] for n in names])

# Pairwise straight-line distance, then convert to a 0–1 similarity.
sim = np.zeros((len(names), len(names)))
for i in range(len(names)):
    for j in range(len(names)):
        dist = np.sqrt(((V[i] - V[j]) ** 2).sum())
        sim[i, j] = 1 / (1 + dist)        # distance 0 → similarity 1

print("Similarity matrix (1.0 = identical):")
print(np.round(sim, 2))
```

```{code-cell} python
:label: heatmap
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(names)), names)
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                color="white" if sim[i, j] < 0.7 else "black")
ax.set_title("Toy protein embedding similarity")
fig.colorbar(im, ax=ax, label="similarity")
fig
```

The heatmap in [](#heatmap) tells the story at a glance: `P1` and `P2` light up as
very similar (they differ by one letter), and `P3` and `P4` form their own similar
pair, while the two pairs are dim against each other. The embedding *grouped proteins
by their makeup* without anyone telling it the groups. Real embeddings from ESM do the
same — but the grouping reflects deep biological function, not just letter counts.

```{admonition} Compute real protein embeddings (companion notebook)
:class: tip
Our composition vector is a hand-made stand-in. The companion Colab notebook loads the
real **ESM-2** protein language model (`facebook/esm2_t6_8M_UR50D`) on a GPU, turns
protein sequences into genuine learned embeddings, and shows their similarity — so you
can compare the toy idea above with the real thing:
<https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/d5-protein-foundation-models.ipynb>
```

## What they can — and can't — do

Foundation models are powerful, but not magic. They excel at finding patterns in the
kind of data they were trained on, and at being adapted to new tasks with relatively
little extra data. But they inherit the cautions from [](4_d4-hallucination.md): they
can be confidently wrong, they can carry biases from their training data, and a
prediction is a *hypothesis to test in the lab*, not a proven fact. AlphaFold predicts
a likely shape; it does not *measure* one. Used wisely — as accelerators for human
scientists, not replacements — they are among the most exciting tools in modern
biology.

## Exercises

```{admonition} Exercise D5.1 — Add a protein
:class: hint
Add a fifth protein `"P5": "AAGGGG"` to the `proteins` dictionary and rebuild the
similarity matrix. Which existing protein is `P5` most similar to, and does that match
your intuition about its ingredients?
```

```{admonition} Solution
:class: dropdown
```python
proteins["P5"] = "AAGGGG"
vectors = {name: composition(seq) for name, seq in proteins.items()}
# ...rebuild V and sim as above...
```

`P5` (two `A`, four `G`) is most similar to **`P2`** (0.81), then **`P1`** (0.68) —
both made of `A`, `G`, and `C`. It shares none of `P3`/`P4`'s `W`/`Y`/`F` ingredients
(similarity ~0.51), so it sits firmly with the first group — exactly what its
composition predicts.
```

```{admonition} Exercise D5.2 — Why "foundation"?
:class: hint
In two or three sentences, explain what makes AlphaFold, ESM, and scGPT *foundation*
models, rather than ordinary special-purpose programs. What is the single property
they share?
```

```{admonition} Solution
:class: dropdown
All three are **pretrained once on enormous amounts of biological data** and then
**adapted to many different downstream tasks**, instead of being hand-built for one job.
That reuse — one general model serving as the "foundation" for protein structure,
mutation effects, cell typing, and more — is the defining property. It mirrors how a
single large language model can write, summarize, and translate, all from the same
pretrained base.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **foundation model** is pretrained once on massive data, then adapted to many
  tasks — the opposite of one program per problem.
- **AlphaFold** [@jumper2021alphafold] predicts protein shape, **ESM** [@lin2023esm]
  is a language model for proteins, and **scGPT** [@cui2024scgpt] is one for cells.
- An **embedding** turns a sequence into a vector so that *similar things land close
  together*; comparing vectors is how a model measures similarity.
- These models are powerful accelerators, not oracles — their outputs are hypotheses
  to verify, and they can be confidently wrong, as in [](4_d4-hallucination.md).
```
