---
title: 'Chapter 4.6 — Foundation Models of Life: Proteins, Genomes, and Medicines'
short_title: '4.6 · Bio foundation models'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.6 — Foundation Models of Life: Proteins, Genomes, and Medicines

*Week 12 · ~45 minutes · Prerequisites: [](4_1-dna-and-proteins-as-text.md), [](4_4-embeddings-and-attention.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/04-protein-foundation-models-esm2.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **foundation model** is and why biology shifted from one-job tools to big reusable models;
- describe four landmark 2021–2026 biological AI breakthroughs (**AlphaFold 3**, **ESM3 / RFdiffusion**, **Evo 2 / AlphaGenome**, and **single-cell models**);
- map real proteins onto a transparent **2D biochemical property map** and compare simple drug feature checklists in Python;
- explain why healthy skepticism (checking simple baselines and running wet-lab tests!) matters even for Nobel-worthy AI.
```

## Start with a case: a 50-year puzzle solved — and a Nobel Prize!

Every living cell runs on tiny molecular machines called **proteins**. A protein starts as a 1D chain of amino-acid letters (Chapter 4.1) and folds within milliseconds into a precise **3D origami shape** — and that 3D shape decides its job!

For fifty years, finding a protein's 3D shape required slow, expensive laboratory X-ray or microscope experiments that could take years for a single protein. Then **AlphaFold** predicted 3D protein structures straight from amino-acid letters [@jumper2021alphafold], and **RFdiffusion** began designing brand-new synthetic proteins from scratch [@watson2023rfdiffusion] — breakthroughs that earned the **2024 Nobel Prize in Chemistry**!

That triumph sparked a huge shift: instead of writing a separate small program for every biology question, scientists began training giant {term}`Foundation model`s — pretrained once on enormous biological libraries, then adapted to dozens of tasks!

## Four breakthroughs reshaping biology (2021–2026)

::::{tab-set}

:::{tab-item} 1. 3D Shapes & Drug Binding (AlphaFold 3)
**AlphaFold 2** [@jumper2021alphafold] folded single proteins. **AlphaFold 3** [@abramson2024alphafold3] went a giant step further: predicting how **proteins, DNA, RNA, and small-molecule medicines lock together in 3D** — helping chemists see whether a candidate drug actually fits into its target pocket!
:::

:::{tab-item} 2. Protein Language & Design (ESM3 & RFdiffusion)
Just as chatbots read English sentences, **ESM-2** [@lin2023esm] and **ESM3** [@hayes2025esm3] train Transformers on hundreds of millions of natural protein sequences! They learn the "grammar of evolution" — spotting harmful mutations and even generating a brand-new glowing green protein (*esmGFP*) unlike anything seen in nature! **RFdiffusion** [@watson2023rfdiffusion] designs custom 3D binder proteins to neutralize toxins or viruses.
:::

:::{tab-item} 3. Reading the Whole Genome (Evo 2 & AlphaGenome)
Only about **2%** of your DNA actually spells out proteins; the other **98%** acts like dimmer switches controlling *when* and *where* genes turn on! Genomic models like **Evo 2** [@brixi2026evo2] and **AlphaGenome** [@avsec2026alphagenome] read up to a million DNA letters at once to predict how tiny single-letter variants affect gene switches and human disease (building on coding-variant tools like **AlphaMissense** [@cheng2023alphamissense]).
:::

:::{tab-item} 4. Virtual Cells — & a Reality Check!
Models like **scGPT** [@cui2024scgpt] and **Geneformer** [@theodoris2023geneformer] train on tens of millions of individual cells to map cell types and aim toward an **"AI Virtual Cell"** that could simulate how a cell responds to a new drug.

**Important scientific reality check:** A careful 2025 *Nature Methods* benchmark [@ahlmann2025perturbation] showed that when predicting how cells react to brand-new genetic changes, **simple linear baselines still match or beat massive deep-learning cell models!** Remember Unit 2's golden habit: *always compare flashy AI against a simple baseline!*
:::

::::

## Lab 1: Putting real proteins on a 2D property map!

When a real protein language model like **ESM-2** reads a protein, it turns the letter chain into an **embedding** (coordinates on a map) so proteins with similar jobs land close together.

Let's build a completely transparent **2-coordinate map** by hand for **7 real biological protein segments** from four distinct families (oxygen-carrying **Hemoglobins**, blood-sugar-regulating **Insulins**, glowing jellyfish **GFP**, and skin/bone structural **Collagen**):
- **x-coordinate:** *% water-fearing (hydrophobic) amino acids* (`A, I, L, M, F, W, V` — these tuck inside a protein's core!)
- **y-coordinate:** *% electrically charged amino acids* (`K, R, H, D, E` — these sit on the outer watery surface!)

```{admonition} Predict before you run
:class: tip
Before running the cell, remember we are giving our program **zero family labels** — only raw amino-acid letters! Will the two Insulins land near each other, and will the Hemoglobins cluster together away from structural Collagen?
```

```{code-cell} python
:label: ch46-protein-map
import matplotlib.pyplot as plt

# Authentic short segments from real biological proteins
proteins = {
    "Hemoglobin_B": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAH",
    "Hemoglobin_A": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVA",
    "Insulin_human":"MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVG",
    "Insulin_pig":  "MALWTRLLPLLALLALWAPAPAQAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREAENPQAG",
    "GFP_jellyfish":"MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTF",
    "Collagen":     "GPMGPMGPRGPPGPAGAPGPQGFQGNPGEPGEPGVSGPMGPRGPPGPPGKNGDDGEAGKPGRPG",
}

WATER_FEARING = set("AILMFWV")
CHARGED       = set("KRHDE")

def protein_coords(seq):
    hydro_pct   = sum(ch in WATER_FEARING for ch in seq) / len(seq) * 100
    charged_pct = sum(ch in CHARGED       for ch in seq) / len(seq) * 100
    return hydro_pct, charged_pct

colors = {
    "Hemoglobin_B": "#0072B2", "Hemoglobin_A": "#0072B2",
    "Insulin_human":"#D55E00", "Insulin_pig":  "#D55E00",
    "GFP_jellyfish":"#009E73", "Collagen":     "#E69F00",
}

plt.figure(figsize=(7.2, 4.2))
for name, seq in proteins.items():
    x, y = protein_coords(seq)
    plt.scatter(x, y, s=120, color=colors[name], edgecolors="black", lw=0.8)
    plt.text(x + 0.8, y + 0.4, name, fontsize=9.2, fontweight="bold")
    print(f"{name:<15}: water-fearing = {x:4.1f}% | charged = {y:4.1f}%")

plt.xlim(12, 62); plt.ylim(8, 30)
plt.xlabel("% water-fearing amino acids (protein core)")
plt.ylabel("% electrically charged amino acids (protein surface)")
plt.title("A 2D protein embedding: biological families group together automatically!")
plt.grid(alpha=0.25)
plt.show()
```

Look how clearly biology jumps off the graph!
- Both **Insulins** (human and pig — which doctors actually used to treat human diabetes for decades!) sit right on top of each other on the far right (very hydrophobic)!
- Both **Hemoglobins** sit together in the middle!
- **Collagen** (full of repeating springy glycine/proline letters for skin and tendons) sits all the way on the far left!

Real protein foundation models like **ESM-2** compute **320+ learned coordinates** that also understand letter order and 3D folding — try running the real ESM-2 model (including scoring the sickle-cell mutation!) on a free GPU in [Companion Colab Notebook 04](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/04-protein-foundation-models-esm2.ipynb)!

## Lab 2: Screening candidate medicines by shared chemical features

In AI drug discovery, chemists also represent small-molecule medicines as a simple **feature checklist** (which chemical building blocks are present) and measure overlap: **shared features ÷ total features present in either drug**!

```{code-cell} python
:label: ch46-drug-overlap
drugs = {
    "Ibuprofen":     {"carbon_ring", "acid_group", "carbon_chain", "oxygen_double_bond"},
    "Naproxen":      {"carbon_ring", "acid_group", "carbon_chain", "oxygen_double_bond", "double_ring"},
    "Aspirin":       {"carbon_ring", "acid_group", "oxygen_double_bond", "ester_link"},
    "Caffeine":      {"double_ring", "nitrogen_ring", "oxygen_double_bond", "methyl_group"},
}

def overlap_score(a, b):
    return len(a & b) / len(a | b)

print("Screening library against Ibuprofen (pain & swelling reliever):")
for name, feats in drugs.items():
    if name != "Ibuprofen":
        print(f"  {name:<12}: similarity = {overlap_score(drugs['Ibuprofen'], feats):.0%}")
```

**Naproxen (`80%`)** and **Aspirin (`60%`)** — the other two anti-inflammatory pain relievers — immediately rise to the top above **Caffeine (`14%`)**!

Yet always remember biology's humbling rule: **an AI prediction is a smart starting clue, never a finished medicine.** Even after AI predicts a drug fits a protein pocket, roughly **90% of candidate drugs still fail in laboratory or human clinical trials** because living human bodies are vastly more complex than any simulation!

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.6.1 (Modify) — Can our simple 2D map spot a scrambled protein?
:class: hint
Compute `protein_coords(proteins["Hemoglobin_B"][::-1])` (reversing the sequence into nonsense gibberish!). Does our 2D percentage map tell any difference between real hemoglobin and backwards gibberish? Why can real protein Transformers like **ESM-2** tell the difference?
```

```{admonition} Solution
:class: dropdown
```python
print("Real     :", protein_coords(proteins["Hemoglobin_B"]))
print("Reversed :", protein_coords(proteins["Hemoglobin_B"][::-1]))
```
They land on the **exact same point**! Counting percentages ignores letter *order*. Real protein Transformers use **position tracking and attention** (Chapter 4.4) so changing letter order completely changes the embedding!
```

```{admonition} Exercise 4.6.2 (Make) — Screen a brand-new candidate molecule
:class: hint
Add a candidate painkiller `"Candidate_X": {"carbon_ring", "acid_group", "carbon_chain", "oxygen_double_bond", "fluorine_atom"}` and compute its `overlap_score` against `"Ibuprofen"` and `"Caffeine"`. Which family does it resemble?
```

```{admonition} Solution
:class: dropdown
```python
cand_x = {"carbon_ring", "acid_group", "carbon_chain", "oxygen_double_bond", "fluorine_atom"}
print(f"vs Ibuprofen: {overlap_score(cand_x, drugs['Ibuprofen']):.0%}")
print(f"vs Caffeine : {overlap_score(cand_x, drugs['Caffeine']):.0%}")
```
It matches Ibuprofen at **80%** (vs. just 13% for Caffeine) — making it an immediate candidate for wet-lab testing!
```

```{admonition} Exercise 4.6.3 (Critical Reasoning) — Why predictions still need wet-lab tests
:class: hint
Suppose an AI model like AlphaFold 3 predicts that a brand-new designed protein will bind tightly to a flu virus and block it. Why do scientists still spend months testing it in real cells, animals, and human **clinical trials** before calling it a medicine?
```

```{admonition} Solution
:class: dropdown
A computer simulation only tests binding shape in isolation. Inside a real human body, a brand-new protein could accidentally bind healthy organs (causing side effects), trigger an unwanted immune reaction, or break down in the bloodstream before reaching the lungs! AI dramatically speeds up *choosing which hypotheses to test*, while real biology experiments remain the final judge.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Foundation models** pretrain once on huge biological libraries and power many tasks via rich **embeddings**.
- **2021–2026 landmarks:** **AlphaFold 2 & 3** (3D shapes & drug complexes), **ESM3 & RFdiffusion** (protein language & design), **Evo 2 & AlphaGenome** (reading whole genomes), and **scGPT / Geneformer** (mapping cells).
- Even at the frontier of science, checking **simple baselines** [@ahlmann2025perturbation] and testing predictions in real labs keeps biology honest.
```

*End of Unit 4!* You have traveled from four DNA letters to Nobel-winning protein AI. In **Unit 5**, we ask the hardest and most human questions in health AI: **Who gets left out? How do we protect patient privacy? How do doctors and laws keep AI safe?** Let's open Unit 5 with [](5_1-fairness-and-bias.md)!
