---
title: 'Chapter 4.1 — Life Written in Letters: DNA, Proteins, k-mers, and Sequence Alignment'
short_title: '4.1 · DNA, proteins & alignment'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.1 — Life Written in Letters: DNA, Proteins, k-mers, and Sequence Alignment

*Estimated time: ~45 minutes · Prerequisites: Part 2*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- analyze a **real human gene sequence** (hemoglobin beta, `HBB`) as text: compute **GC content**, count **k-mers**, and translate DNA **codons** into amino acids;
- contrast a single-letter **substitution** (the sickle-cell mutation) against a **frameshift deletion**;
- explain why **Hamming distance** fails when insertions or deletions occur, and build a **dynamic-programming edit distance** table from scratch;
- visualize a DP alignment matrix and trace how sequence similarity reveals shared evolutionary ancestry and function.
```

## Start with a case: one letter out of three billion

Your genome contains roughly **3 billion DNA letters** written in a four-letter alphabet: `A` (adenine), `C` (cytosine), `G` (guanine), and `T` (thymine). Inside that vast text sits the `HBB` gene, which encodes the beta chain of **hemoglobin** — the protein carrying oxygen inside your red blood cells.

Change just **one single letter** inside `HBB` — swapping an `A` to a `T` at the 20th coding base (`GAG` → `GTG`) — and the cell inserts the amino acid **Valine (`V`)** instead of **Glutamic acid (`E`)**. That one chemical swap makes hemoglobin molecules stick together into stiff fibers under low oxygen, twisting red blood cells into crescents: **sickle-cell disease**.

Life runs on sequences. Because DNA (`4` letters) and proteins (`20` amino-acid letters) are **ordered strings of symbols**, every text-searching, comparison, and language-modeling tool in AI can be pointed straight at molecular biology.

```{mermaid}
flowchart LR
    DNA["DNA Sequence<br/>4-letter alphabet<br/>(A, C, G, T)"] -- "Read in 3-letter<br/>codons" --> Prot["Protein Sequence<br/>20-letter amino acids<br/>(M, V, H, L, T, P, E...)"]
    Prot -- "Folds in 3D" --> Shape["3D Protein Structure<br/>& Biological Function"]
```

## Lab 1: Reading, translating, and fingerprinting real human hemoglobin (`HBB`)

Let's inspect the authentic opening 75 coding bases of human **hemoglobin beta (`HBB`)** alongside:
1. the real **sickle-cell variant** (1 base substituted: `A → T`), and
2. a **1-base frameshift deletion** (removing a single base early in the gene).

```{admonition} Definition — GC content, codon, and k-mer
:class: important
- {term}`GC content` is the fraction of `G` and `C` bases in a DNA sequence (because `G–C` pairs form three hydrogen bonds versus two for `A–T`, high-GC regions are physically more stable and vary distinctly across species).
- A {term}`Codon` is a 3-letter DNA triplet specifying one amino acid (or a STOP signal).
- A {term}`k-mer` is any contiguous window of $k$ letters; counting all k-mers turns a sequence into a numerical fingerprint.
```

```{admonition} Predict before you run
:class: tip
Both the sickle-cell variant and the frameshift variant change just **1 single DNA base** out of 75! When translated into protein amino acids, how many amino acids will change in the **substitution** versus the **1-base deletion**?
```

```{code-cell} python
:label: ch41-hbb-translate
from collections import Counter

# Authentic opening 75 coding bases of human HBB (RefSeq NM_000518)
hbb_normal = "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGT"
hbb_sickle = hbb_normal[:19] + "T" + hbb_normal[20:]        # 1 substitution: GAG -> GTG
hbb_frame  = hbb_normal[:7] + hbb_normal[8:]                # 1 deletion at index 7

CODON_TABLE = {
    "ATG": "M", "GTG": "V", "GTT": "V", "GTA": "V", "GTC": "V",
    "CAT": "H", "CAC": "H", "CTG": "L", "CTA": "L", "CTC": "L", "CTT": "L", "TTG": "L", "TTA": "L",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "GAG": "E", "GAA": "E", "AAG": "K", "AAA": "K",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "GCC": "A", "GCT": "A", "GCA": "A", "GCG": "A",
    "TGG": "W", "GGC": "G", "GGT": "G", "GGA": "G", "GGG": "G",
    "AAC": "N", "AAT": "N", "GAT": "D", "GAC": "D",
    "TGA": "*", "TAA": "*", "TAG": "*",
    "ATC": "I", "ATT": "I", "ATA": "I", "CGA": "R", "CGG": "R", "CGT": "R", "CGC": "R", "AGA": "R", "AGG": "R",
    "TTC": "F", "TTT": "F", "TAC": "Y", "TAT": "Y", "TGC": "C", "TGT": "C",
    "CAG": "Q", "CAA": "Q",
}

def translate(dna):
    return "".join(CODON_TABLE.get(dna[i:i+3], "?") for i in range(0, len(dna) - 2, 3))

def gc_content(dna):
    return (dna.count("G") + dna.count("C")) / len(dna)

p_norm   = translate(hbb_normal)
p_sickle = translate(hbb_sickle)
p_frame  = translate(hbb_frame)

print(f"HBB length : {len(hbb_normal)} bases | GC content = {gc_content(hbb_normal):.1%}")
print(f"Normal HBB : {p_norm}")
print(f"Sickle HBB : {p_sickle}  (only 1 amino acid changed: E -> V!)")
print(f"Frameshift : {p_frame}  (EVERY downstream amino acid scrambled + early STOP '*'!)")
```

Look at that jaw-dropping contrast:
- **Substitution (`hbb_sickle`):** Only position 7 changes (`E → V`) — yet even that single amino-acid change alters protein folding and causes sickle-cell anemia!
- **Frameshift (`hbb_frame`):** Deleting just *one* base shifts every subsequent 3-letter codon window out of phase, scrambling the entire protein and even creating a premature **STOP codon (`*`)**!

Now let's compare **k-mer fingerprints** of our human `HBB` gene against a low-GC AT-rich bacterial promoter region — this simple counting trick is how bioinformatics classifiers distinguish human genes, viral sequences, and regulatory elements!

```{code-cell} python
:label: ch41-kmers-fig
import numpy as np
import matplotlib.pyplot as plt

BLUE, VERMILION = "#0072B2", "#D55E00"

def kmer_freq(seq, k=2):
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    total = sum(counts.values())
    return {km: c / total for km, c in counts.items()}

at_rich_region = "TATAAATAGATTATAAATGCTATAAATTTAATATAGCATTATAAATGATATAATTTAATATGCTATAAAATTA"
f_hbb = kmer_freq(hbb_normal, k=2)
f_at  = kmer_freq(at_rich_region, k=2)

dimers = sorted(set(f_hbb) | set(f_at))
x = np.arange(len(dimers))

fig, ax = plt.subplots(figsize=(9.5, 3.6))
w = 0.38
ax.bar(x - w/2, [f_hbb.get(d, 0) * 100 for d in dimers], w, color=BLUE, label=f"Human HBB (GC={gc_content(hbb_normal):.0%})")
ax.bar(x + w/2, [f_at.get(d, 0) * 100 for d in dimers], w, color=VERMILION, label=f"AT-rich promoter (GC={gc_content(at_rich_region):.0%})")
ax.set_xticks(x, dimers)
ax.set_xlabel("2-mer (dinucleotide)")
ax.set_ylabel("Frequency (%)")
ax.set_title("2-mer fingerprints separate biological sequences immediately")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
```

## Lab 2: Comparing sequences — why Hamming fails and Dynamic Programming wins

Suppose a virologist isolates a new respiratory virus and wants to compare its genome to known strains. The simplest metric — {term}`Hamming distance` — counts how many positions disagree between two equal-length strings.

Watch what happens when just **one single base is deleted** at the start of a sequence: Hamming distance explodes, because every letter slides out of register (the exact same phase shift that wrecked our frameshift protein!).

To handle **substitutions, insertions, AND deletions** gracefully, bioinformatics uses {term}`Edit distance` (Levenshtein distance) and sequence {term}`Alignment`, computed efficiently via {term}`Dynamic programming`: build a small grid $D[i, j]$ holding the minimum edits needed to turn the first $i$ letters of sequence $A$ into the first $j$ letters of sequence $B$.

Each cell depends only on its three neighbors (up = deletion, left = insertion, diagonal = match or substitution):

$$ D[i, j] = \begin{cases} D[i-1, j-1] & \text{if } A[i] = B[j] \\ 1 + \min\bigl(D[i-1, j],\; D[i, j-1],\; D[i-1, j-1]\bigr) & \text{otherwise} \end{cases} $$ (eq-41-dp)

```{admonition} Predict before you run
:class: tip
Compare `"MVHLTPEE"` (normal human hemoglobin start) against `"VHLTPEEK"` (shifted left by deleting `M` and inserting `K` at the end — **only 2 real edits**!). What will **Hamming distance** report vs. **Dynamic-Programming edit distance**?
```

```{code-cell} python
:label: ch41-dp-align
def hamming(a, b):
    assert len(a) == len(b), "Hamming needs equal lengths"
    return sum(x != y for x, y in zip(a, b))

def edit_distance_table(a, b):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1,        # deletion
                          D[i, j - 1] + 1,        # insertion
                          D[i - 1, j - 1] + cost) # match / substitution
    return D

def traceback_alignment(a, b, D):
    """Walk backward from bottom-right corner to recover one optimal alignment."""
    i, j = len(a), len(b)
    aln_a, aln_b = [], []
    path = [(i, j)]
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            aln_a.append(a[i - 1]); aln_b.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            aln_a.append(a[i - 1]); aln_b.append("-"); i -= 1
        else:
            aln_a.append("-"); aln_b.append(b[j - 1]); j -= 1
        path.append((i, j))
    return "".join(reversed(aln_a)), "".join(reversed(aln_b)), path[::-1]

seq_a = "MVHLTPEE"
seq_b = "VHLTPEEK"
D = edit_distance_table(seq_a, seq_b)
aln_a, aln_b, opt_path = traceback_alignment(seq_a, seq_b, D)

print(f"Hamming distance (fooled by shift!) : {hamming(seq_a, seq_b)} / {len(seq_a)}")
print(f"DP Edit distance (true minimum)     : {D[-1, -1]}")
print(f"Recovered optimal alignment:\n  A: {aln_a}\n  B: {aln_b}")
```

Look how cleanly the traceback inserted a gap (`-`) to re-align all seven matching amino acids `VHLTPEE`! Let's plot the dynamic-programming matrix $D$ with the optimal traceback path highlighted in amber so you can *see* the algorithm walking from top-left to bottom-right.

```{code-cell} python
:label: ch41-dp-fig
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(D, cmap="Blues")

# Plot the optimal traceback route
pr = [p[0] for p in opt_path]
pc = [p[1] for p in opt_path]
ax.plot(pc, pr, "o-", color="#D55E00", lw=2.5, ms=8, mfc="none", mew=2.2, label="Optimal alignment path")

for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        ax.text(j, i, str(D[i, j]), ha="center", va="center", fontsize=10,
                color="white" if D[i, j] > D.max() * 0.6 else "#111111", fontweight="bold")

ax.set_xticks(range(len(seq_b) + 1), ["∅"] + list(seq_b))
ax.set_yticks(range(len(seq_a) + 1), ["∅"] + list(seq_a))
ax.set_xlabel("Sequence B (VHLTPEEK)")
ax.set_ylabel("Sequence A (MVHLTPEE)")
ax.set_title("Dynamic-programming edit-distance table & optimal path")
ax.legend(frameon=True, loc="lower left", fontsize=8.5)
fig.tight_layout()
plt.show()
```

:::{margin}
Real sequence aligners like **Needleman–Wunsch**, **Smith–Waterman**, and **BLAST** use this exact dynamic-programming grid — upgraded with biological substitution matrices (BLOSUM) so swapping two chemically similar amino acids (like `D ↔ E`) costs less than an extreme swap (like `E → V` in sickle-cell disease!).
:::

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.1.1 (Modify) — Comparing human hemoglobin beta to hemoglobin alpha
:class: hint
The opening 15 amino acids of human **hemoglobin alpha (`HBA1`)** are `"MVLSPADKTNVKAAW"`, while human **hemoglobin beta (`HBB`)** starts `"MVHLTPEEKSAVTAL"`. Compute both the **Hamming distance** and the **DP edit distance** (plus optimal alignment!) between them. Do gaps improve the match?
```

```{admonition} Solution
:class: dropdown
```python
hba = "MVLSPADKTNVKAAW"
hbb = "MVHLTPEEKSAVTAL"
D_ab = edit_distance_table(hba, hbb)
aa, bb, _ = traceback_alignment(hba, hbb, D_ab)
print(f"Hamming distance: {hamming(hba, hbb)} | Edit distance: {D_ab[-1, -1]}")
print(f"Alignment:\n  HBA: {aa}\n  HBB: {bb}")
```

Even though `HBA1` and `HBB` diverged from a common ancestral globin gene hundreds of millions of years ago, allowing small gaps lets edit distance align conserved positions (such as `M`, `V`, `L`, `K`, `A`) better than rigid position-by-position Hamming matching!
```

```{admonition} Exercise 4.1.2 (Make) — Hunt for a biological motif (the TATA box)
:class: hint
Many eukaryotic genes are preceded by a promoter motif called a **TATA box** (`"TATAAA"`), where proteins bind to start copying DNA into RNA. Write a 4-line list comprehension that finds every starting index of `"TATAAA"` inside `at_rich_region` from [](#ch41-kmers-fig).
```

```{admonition} Solution
:class: dropdown
```python
motif = "TATAAA"
hits = [i for i in range(len(at_rich_region) - len(motif) + 1)
        if at_rich_region[i:i+len(motif)] == motif]
print(f"Found {motif!r} {len(hits)} times at positions: {hits}")
```

Simple exact scans find rigid motifs instantly; when biological motifs tolerate slight variation (like `TATATA` vs. `TATAAA`), probabilistic motif profiles or deep genomic models like **AlphaGenome** (Chapter 4.4) take over!
```

```{admonition} Exercise 4.1.3 (Critical Reasoning) — Why isn't sequence alignment enough to predict mutation severity?
:class: hint
Both the sickle-cell mutation (`E → V` in `HBB`) and a harmless benign polymorphism elsewhere in `HBB` have the **exact same edit distance = 1** from normal human hemoglobin! Why can't plain edit distance tell which single-letter mutation causes life-threatening disease and which is completely harmless?
```

```{admonition} Solution
:class: dropdown
Edit distance treats every position and every letter swap equally (`cost = 1`). In biology, **context and 3D structure** are everything: swapping a negatively charged polar amino acid (`E`) for a water-fearing hydrophobic one (`V`) on the *exposed surface* of hemoglobin creates a sticky patch that causes fibers to clump, whereas swapping two similar residues in a flexible loop may do nothing at all! Learning that rich evolutionary context across hundreds of millions of sequences is exactly why **protein language models** (Chapters 4.2 & 4.4) revolutionized variant-effect prediction [@cheng2023alphamissense].
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- DNA (`A, C, G, T`) and proteins (`20` amino acids) are ordered text; **GC content**, **codons**, and **k-mer fingerprints** extract rich biological signals in a few lines of Python.
- A single-base **substitution** alters at most one amino acid (yet can still cause diseases like sickle-cell anemia), whereas a 1-base **deletion** causes a catastrophic **frameshift**.
- **Hamming distance** breaks under insertions/deletions; **dynamic programming** computes minimum **edit distance** and optimal **sequence alignments**.
- Sequence similarity is evidence of shared ancestry — setting up the shift from hand-coded alignments to **neural language models**.
```

*Next up:* What happens when we train a neural network to read sequences — clinical notes, medical questions, or proteins — by predicting the next token using **self-attention**? Open the Transformer hood in [](4_2-language-models-and-attention.md).
