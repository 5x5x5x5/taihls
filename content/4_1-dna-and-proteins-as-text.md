---
title: 'Chapter 4.1 — Life Written in Letters: DNA, Codons, and Mutations'
short_title: '4.1 · DNA & proteins as text'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.1 — Life Written in Letters: DNA, Codons, and Mutations

*Week 10 · ~40 minutes · Prerequisites: Unit 0*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- treat a real stretch of human **DNA** (the **hemoglobin** gene) as a string of four letters (`A`, `C`, `G`, `T`);
- compute **GC content** and count **k-mers** (short letter chunks) in Python;
- translate three-letter DNA **codons** into protein amino acids;
- compare a single-letter **substitution** (sickle-cell disease) against a single-letter **deletion** (**frameshift**).
```

## Start with a case: one letter out of three billion

Inside almost every cell in your body sits a molecule about two meters long coiled into a microscopic nucleus: your **DNA**. Astonishingly, the instruction manual to build a human being is written in an alphabet of just **four letters**: `A`, `C`, `G`, and `T`.

One gene in that manual — called **`HBB`** — gives instructions for building {term}`Hemoglobin`, the protein inside red blood cells that carries oxygen from your lungs to your muscles and brain.

Change just **one single letter** inside `HBB` (`A → T`), and the cell swaps one building block (**amino acid**) out of hundreds: `E` (glutamic acid) becomes `V` (valine). That single typo makes hemoglobin stick into stiff rods, bending red blood cells into crescent shapes: **sickle-cell disease**.

Because DNA (`4` letters) and proteins (`20` amino-acid letters) are **ordered text**, every text-analysis and AI language tool can be aimed straight at biology!

```{mermaid}
flowchart LR
    DNA["DNA Text<br/>4 letters: A, C, G, T"] -- "Read in 3-letter<br/>codons" --> Prot["Protein Text<br/>20 amino-acid letters"]
    Prot -- "Folds in 3D" --> Job["3D Shape & Job<br/>(e.g., carrying oxygen)"]
```

## Reading real human hemoglobin (`HBB`)

Let's check two simple DNA fingerprints first:
- **{term}`GC content`:** What percentage of the DNA is `G` or `C`? (`G` pairs with `C` more tightly than `A` pairs with `T`, so high-GC stretches are physically sturdier and vary distinctly across species!)
- **{term}`k-mer`s:** Every sliding chunk of $k$ consecutive letters. For `"AGCT"`, the 2-mers are `"AG"`, `"GC"`, and `"CT"`. Counting k-mers gives a sequence a quick "fingerprint."

And when cells turn DNA into a protein, they read **three letters at a time** — each triplet is a {term}`Codon`!
- Try translating `"ATGGTGCAT"` by hand: `ATG → M`, `GTG → V`, `CAT → H` → protein `"MVH"`!

```{admonition} Predict before you run
:class: tip
Below we start with the real opening 75 DNA letters of human `HBB` and create two mutated versions — each changing just **ONE letter**:
1. **Sickle-cell version:** swaps letter #20 from `A` to `T` (`GAG → GTG`).
2. **Frameshift version:** deletes letter #8 completely (making the DNA 74 letters long).

How many protein amino acids will change in the **swap** versus the **1-letter deletion**?
```

```{code-cell} python
:label: ch41-hbb
from collections import Counter

# Authentic opening 75 coding letters of human hemoglobin beta (HBB)
hbb_normal = "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGT"
hbb_sickle = hbb_normal[:19] + "T" + hbb_normal[20:]     # 1 swap (GAG -> GTG)
hbb_frame  = hbb_normal[:7]  + hbb_normal[8:]            # 1 deleted letter!

# Lookup table turning each 3-letter DNA codon into its 1-letter amino acid (* = STOP)
CODONS = {
    "ATG": "M", "GTG": "V", "GTT": "V", "GTA": "V", "GTC": "V",
    "CAT": "H", "CAC": "H", "CTG": "L", "CTA": "L", "CTC": "L", "CTT": "L", "TTG": "L", "TTA": "L",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "GAG": "E", "GAA": "E", "AAG": "K", "AAA": "K", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "AGT": "S", "AGC": "S", "GCC": "A", "GCT": "A", "GCA": "A", "GCG": "A", "TGG": "W",
    "GGC": "G", "GGT": "G", "GGA": "G", "GGG": "G", "AAC": "N", "AAT": "N", "GAT": "D", "GAC": "D",
    "TGA": "*", "TAA": "*", "TAG": "*", "CGA": "R", "CGG": "R", "CGT": "R", "CGC": "R", "AGA": "R", "AGG": "R",
    "ATC": "I", "ATT": "I", "ATA": "I", "TTC": "F", "TTT": "F", "TAC": "Y", "TAT": "Y", "TGC": "C", "TGT": "C",
    "CAG": "Q", "CAA": "Q",
}

def translate(dna):
    protein = ""
    for i in range(0, len(dna) - 2, 3):
        protein += CODONS.get(dna[i:i+3], "?")
    return protein

gc = (hbb_normal.count("G") + hbb_normal.count("C")) / len(hbb_normal)
print(f"Human HBB length: {len(hbb_normal)} bases | GC content = {gc:.1%}")
print(f"Normal protein  : {translate(hbb_normal)}")
print(f"Sickle protein  : {translate(hbb_sickle)}   <-- ONLY 1 amino acid changed (E -> V)!")
print(f"Frameshift      : {translate(hbb_frame)}    <-- EVERY later letter scrambled + STOP '*'!")
```

Look how dramatic a {term}`Frameshift` is! Deleting a single letter shifts every 3-letter reading frame out of step, turning the rest of the protein into nonsense and even hitting a premature **STOP (`*`)**!

## Visualizing a DNA k-mer fingerprint

Let's count the most common 2-mers in our human `HBB` gene and plot its bar-chart fingerprint.

```{code-cell} python
:label: ch41-kmer-plot
import matplotlib.pyplot as plt

dimers = Counter(hbb_normal[i:i+2] for i in range(len(hbb_normal) - 1))
top8 = dimers.most_common(8)
labels = [pair for pair, count in top8]
counts = [count for pair, count in top8]

plt.figure(figsize=(6.8, 3.6))
plt.bar(labels, counts, color="#009E73")
plt.xlabel("2-mer (pair of DNA letters)")
plt.ylabel("Count in HBB snippet")
plt.title("Most common 2-mers in human hemoglobin (HBB)")
plt.grid(axis="y", alpha=0.25)
plt.show()
```

Different genes and different organisms carry distinct k-mer fingerprints — turning walls of `ACGT` into tables of numbers that machine-learning classifiers can read!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.1.1 (Modify) — What if you delete THREE letters instead of one?
:class: hint
Create `hbb_del3 = hbb_normal[:6] + hbb_normal[9:]` (deleting one full 3-letter codon!) and print `translate(hbb_del3)`. Why does deleting 3 letters do far less scrambling than deleting 1 letter?
```

```{admonition} Solution
:class: dropdown
```python
hbb_del3 = hbb_normal[:6] + hbb_normal[9:]
print("Normal     :", translate(hbb_normal))
print("Deleted 3  :", translate(hbb_del3))
```
Deleting an exact multiple of 3 removes just **one** amino acid (`H`) and keeps every remaining 3-letter codon in frame — no frameshift!
```

```{admonition} Exercise 4.1.2 (Make) — Hunt for a promoter signal (`TATAAA`)
:class: hint
Before many genes begin, cells look for a short start-signal pattern called a **TATA box** (`"TATAAA"`). Write a short `for` loop to find every starting position of `"TATAAA"` inside `region = "GCGTATAAACGCTATAAAGGTATAAA"`.
```

```{admonition} Solution
:class: dropdown
```python
region = "GCGTATAAACGCTATAAAGGTATAAA"
hits = [i for i in range(len(region) - 5) if region[i:i+6] == "TATAAA"]
print("TATA box positions:", hits)   # -> [3, 12, 20]
```
Scanning for short meaningful patterns (**motifs**) is one of the oldest jobs in bioinformatics!
```

```{admonition} Exercise 4.1.3 (Critical Reasoning) — Silent mutations
:class: hint
Look at `CODONS` in [](#ch41-hbb): both `"GAG"` and `"GAA"` translate to the exact same amino acid **`E`**! If a DNA mutation changes `"GAG"` into `"GAA"`, what happens to the finished protein?
```

```{admonition} Solution
:class: dropdown
The finished amino-acid chain stays **100% identical**! Biologists call this a **silent (synonymous) mutation** — showing how nature built redundancy right into the genetic code.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- DNA (`A, C, G, T`) and proteins (`20` amino acids) are ordered sequences of letters.
- **GC content** and **k-mer counts** summarize a sequence into numbers a computer can compare.
- Cells read DNA in 3-letter **codons**: a 1-letter swap can still cause disease (like sickle-cell `E → V`), while a 1-letter deletion causes a **frameshift** that scrambles the rest of the protein.
```

*Next up:* How does a computer compare two DNA or protein sequences when one has a missing or extra letter? Let's build **sequence alignment** from scratch in [](4_2-comparing-sequences.md)!
