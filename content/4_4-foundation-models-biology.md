---
title: 'Chapter 4.4 — Foundation Models of Life: Proteins, Genomes, Drugs, and Virtual Cells'
short_title: '4.4 · Bio foundation models'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.4 — Foundation Models of Life: Proteins, Genomes, Drugs, and Virtual Cells

*Estimated time: ~55 minutes · Prerequisites: [](4_1-dna-proteins-and-alignment.md), [](4_2-language-models-and-attention.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/04-protein-foundation-models-esm2.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **foundation model** is and why biology shifted from narrow one-task tools to general pretrained models;
- describe four 2021–2026 biological AI frontiers: **structure prediction (AlphaFold 2 & 3)**, **protein language & de novo design (ESM-2, ESM3, RFdiffusion)**, **genomic foundation models (AlphaMissense, Evo 2, AlphaGenome)**, and **single-cell "virtual cells" (Geneformer, scGPT)**;
- build a transparent physicochemical **protein embedding** and a **Tanimoto molecular-fingerprint** drug similarity screen in Python;
- apply healthy scientific skepticism by explaining why simple linear baselines still rival giant single-cell perturbation models on strict zero-shot tests.
```

## Start with a case: a Nobel Prize — and what happened next

For fifty years, predicting how a 1D chain of amino acids folds into a 3D protein was one of biology's grandest unsolved challenges. Experimental lab imaging (X-ray crystallography or cryo-EM) could take a PhD student years for a *single* protein.

Then **AlphaFold 2** cracked the protein-folding problem [@jumper2021alphafold], and **RFdiffusion** began designing brand-new synthetic proteins that never existed in nature [@watson2023rfdiffusion] — breakthroughs honored with the **2024 Nobel Prize in Chemistry** (Demis Hassabis, John Jumper, and David Baker).

And biology didn't stop at single proteins! Between 2024 and 2026, the field leaped from *predicting single shapes* to **multimodal, generative {term}`Foundation model`s** spanning every layer of life — from non-coding DNA to drug complexes to whole cells!

```{mermaid}
flowchart LR
    G["Genomes (DNA)<br/>Evo 2 · AlphaGenome<br/>AlphaMissense"] --> P["Proteins & 3D Complexes<br/>AlphaFold 3 · ESM3<br/>RFdiffusion"]
    P --> M["Molecules & Drugs<br/>Molecular Fingerprints<br/>Binding & Virtual Screening"]
    M --> C["Cells & Tissues<br/>Geneformer · scGPT<br/>AI Virtual Cells"]
```

## Four frontiers reshaping the life sciences (2021–2026)

Explore each pillar below — notice how **every single one** builds directly on the ideas you already mastered: tokens, embeddings, attention, diffusion, and honest benchmarking!

::::{tab-set}

:::{tab-item} 1. Structure & Complexes (AlphaFold 3)
- **AlphaFold 2** [@jumper2021alphafold] predicted 3D coordinates of single proteins from evolutionary alignments, mapping virtually the entire known protein universe.
- **AlphaFold 3** [@abramson2024alphafold3] replaced parts of the older pipeline with a **diffusion architecture** (the same generative denoising trick behind modern image generators!) to predict joint 3D structures of **proteins + DNA + RNA + small-molecule drugs + chemical modifications** together — accelerating structure-based drug discovery dramatically.
:::

:::{tab-item} 2. Protein Language & Design (ESM3 & RFdiffusion)
- **ESM-2** [@lin2023esm] trained a Transformer directly on hundreds of millions of protein sequences by masking amino acids and asking the model to fill in the blanks — learning evolutionary "grammar" and rich protein **embeddings** without ever being explicitly taught chemistry!
- **ESM3** [@hayes2025esm3] unified **sequence, 3D structure, and biological function** into one generative multimodal model — even generating a novel glowing fluorescent protein (*esmGFP*) separated from any known natural protein by the equivalent of **500 million years of evolution**!
- **RFdiffusion** [@watson2023rfdiffusion] generates bespoke 3D protein backbones from scratch — for example, custom binders engineered to neutralize specific snake venoms or viral spikes.
:::

:::{tab-item} 3. Reading the Whole Genome (Evo 2 & AlphaGenome)
- Only ~2% of human DNA codes for proteins; the other **98% is non-coding regulatory DNA** controlling *when*, *where*, and *how strongly* genes turn on!
- **AlphaMissense** [@cheng2023alphamissense] scored every possible single-letter protein-coding mutation in the human proteome to help rare-disease geneticists separate harmless variants from pathogenic ones.
- **Evo** [@nguyen2024evo] and **Evo 2** [@brixi2026evo2] scaled long-context genomic foundation models across all domains of life — capable of reading up to a million DNA letters at once to score disease variants and design synthetic genetic systems.
- **AlphaGenome** [@avsec2026alphagenome] predicts how single-letter DNA changes across up to **1 million bases** alter gene splicing, chromatin accessibility, and gene expression across cell types — illuminating the non-coding "dark matter" of human disease!
:::

:::{tab-item} 4. Virtual Cells & The Reality Check!
- Single-cell RNA sequencing measures the activity of ~20,000 genes inside millions of individual cells. Foundation models like **Geneformer** [@theodoris2023geneformer] and **scGPT** [@cui2024scgpt] treat each cell's active genes like a "sentence," pretraining on tens of millions of cells to classify cell types and integrate multi-lab atlases.
- **The crucial 2025–2026 scientific reality check:** Can these models yet act as a true **"AI Virtual Cell"** predicting how an unseen cell line responds to a new CRISPR gene knockdown or drug? Rigorous leakage-free benchmarks — including a landmark *Nature Methods* study [@ahlmann2025perturbation] — discovered that on strict perturbation tests, **simple linear baselines still match or beat massive deep-learning foundation models**! Remember our mantra from Part 2: *always check the simple baseline!*
:::

::::

## Lab 1: Protein embeddings — seeing biological families cluster automatically

Inside a protein foundation model like **ESM-2**, reading a protein turns its variable-length letter string into a fixed-length {term}`Embedding` vector so that **proteins with similar structure and function sit close together in space** — even when their exact spelling differs!

Let's build a transparent, biologically meaningful **24-number physicochemical embedding** (combining the 20 amino-acid frequencies + 4 real biochemical properties: hydrophobicity, positive charge, negative charge, and aromaticity) across **8 real biological protein segments** from four distinct families:
- **Oxygen carriers:** Human Hemoglobin $\alpha$ (`HBA1`), Human Hemoglobin $\beta$ (`HBB`), Sickle-cell `HBB` (`E6V`)
- **Peptide hormones:** Human Insulin (`INS_human`), Pig/Porcine Insulin (`INS_pig` — historically used for decades to treat human diabetes!)
- **Structural fibers:** Human Collagen alpha-1 (`COL1A1` — famous `G-X-Y` glycine-proline repeats!)
- **Fluorescent proteins:** Jellyfish Green Fluorescent Protein (`GFP_wild`) & Enhanced `eGFP`

```{admonition} Predict before you run
:class: tip
Before running the cell, look at those four protein families: hemoglobins, insulins, collagen, and GFPs. Even though our toy embedding throws away letter order and only measures amino-acid ingredients + biochemical properties, will proteins from the **same biological family** cluster together in the cosine-similarity heatmap and 2D PCA map? Which protein will sit farthest out on its own?
```

```{code-cell} python
:label: ch44-protein-embed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Authentic short biological segments (N-terminal / core regions)
proteins = {
    "HBB_human":  "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAH",
    "HBB_sickle": "MVHLTPVEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAH",
    "HBA_human":  "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVA",
    "INS_human":  "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVG",
    "INS_pig":    "MALWTRLLPLLALLALWAPAPAQAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREAENPQAG",
    "GFP_wild":   "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTF",
    "eGFP_var":   "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT",
    "Collagen":   "GPMGPMGPRGPPGPAGAPGPQGFQGNPGEPGEPGVSGPMGPRGPPGPPGKNGDDGEAGKPGRPG",
}

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWV")
POS_CHARGE  = set("KRH")
NEG_CHARGE  = set("DE")
AROMATIC    = set("FWY")

def protein_embedding(seq):
    n = len(seq)
    comp = np.array([seq.count(a) / n for a in AA])
    props = np.array([
        sum(c in HYDROPHOBIC for c in seq) / n,
        sum(c in POS_CHARGE  for c in seq) / n,
        sum(c in NEG_CHARGE  for c in seq) / n,
        sum(c in AROMATIC    for c in seq) / n,
    ])
    return np.concatenate([comp, props])

names = list(proteins)
V = np.array([protein_embedding(proteins[k]) for k in names])

# Pairwise cosine similarity matrix
norms = np.linalg.norm(V, axis=1, keepdims=True)
V_unit = V / norms
cos_sim = V_unit @ V_unit.T

# 2D projection via PCA so human eyes can inspect the 24D embedding space!
coords = PCA(n_components=2, random_state=0).fit_transform(V)

BLUE, VERMILION, TEAL, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"
fam_colors = [BLUE, BLUE, BLUE, VERMILION, VERMILION, TEAL, TEAL, AMBER]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))

im = ax1.imshow(cos_sim, cmap="viridis", vmin=0.35, vmax=1.0)
ax1.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8.5)
ax1.set_yticks(range(len(names)), names, fontsize=8.5)
for i in range(len(names)):
    for j in range(len(names)):
        ax1.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                 color="white" if cos_sim[i, j] < 0.78 else "#111111")
ax1.set_title("A. Protein embedding cosine similarity")
fig.colorbar(im, ax=ax1, shrink=0.82)

ax2.scatter(coords[:, 0], coords[:, 1], s=110, c=fam_colors, edgecolors="#111111", lw=0.8)
for i, name in enumerate(names):
    offset_y = 0.015 if i % 2 == 0 else -0.028
    ax2.text(coords[i, 0] + 0.012, coords[i, 1] + offset_y, name, fontsize=8.8)
ax2.set_xlabel("Embedding PC 1")
ax2.set_ylabel("Embedding PC 2")
ax2.set_title("B. 2D map: biological families cluster without labels!")
ax2.grid(alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()
```

Look at Panel A & Panel B of [](#ch44-protein-embed)!
- Without a single human label, the four biological families snap into crisp islands: the **Hemoglobins** cluster together, the two **Insulins** sit side by side, the two **GFPs** overlap, and glycine/proline-rich **Collagen** stands alone in its own corner!
- **Wait — notice a subtle limitation too!** In our simple composition vector, `HBB_human` and `HBB_sickle` have cosine similarity `1.00` (almost identical!), because changing just 1 amino acid barely changes overall composition. A deep contextual Transformer like **ESM-2** or **AlphaMissense** sees *where* in the sequence and folded pocket that single substitution lands — flagging `E6V` as disruptive! You will test real **ESM-2** embeddings and zero-shot mutation scoring on GPU in [Companion Notebook 04](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/04-protein-foundation-models-esm2.ipynb)!

## Lab 2: AI in drug discovery — molecular fingerprints & virtual screening

How does AI help chemists search through **billions of candidate small-molecule drugs** to find ones that bind a disease target safely?

Cheminformatics converts each molecule's 2D chemical graph into a binary **molecular fingerprint** — a checklist of `1`s and `0`s recording which chemical substructures (*benzene ring, carboxylic acid, ester, beta-lactam ring, amide, amine, ...*) are present. Chemists then compare candidate molecules to a known effective drug using {term}`Tanimoto similarity`:

$$ \text{Tanimoto}(A, B) = \frac{\text{substructures shared by BOTH } A \text{ and } B}{\text{substructures present in EITHER } A \text{ or } B} $$ (eq-44-tanimoto)

Let's run a miniature virtual screen across six famous real drugs — **Aspirin, Ibuprofen, Naproxen** (all NSAID pain/inflammation drugs), **Acetaminophen**, **Penicillin G** (antibiotic), and **Caffeine**!

```{code-cell} python
:label: ch44-drug-fingerprints
# 10 interpretable chemical substructure bits:
# [aromatic_ring, carboxylic_acid, ester, amide, beta_lactam,
#  sulfur, carbonyl, methyl_group, hydroxyl_phenol, fused_bicyclic]
drugs = {
    "Ibuprofen (NSAID)":   np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0]),
    "Naproxen (NSAID)":    np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 1]),
    "Aspirin (NSAID)":     np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0]),
    "Acetaminophen":       np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0]),
    "Penicillin G":        np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1]),
    "Caffeine":            np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 1]),
}

def tanimoto(a, b):
    both = np.sum((a == 1) & (b == 1))
    either = np.sum((a == 1) | (b == 1))
    return both / either

query_drug = "Ibuprofen (NSAID)"
print(f"Virtual screen ranked by Tanimoto similarity to {query_drug}:")
ranked = sorted(
    [(name, tanimoto(drugs[query_drug], fp)) for name, fp in drugs.items() if name != query_drug],
    key=lambda x: x[1], reverse=True
)
for name, sim in ranked:
    print(f"  {name:<20}: Tanimoto = {sim:.2f}  {'█' * int(sim * 25)}")
```

Immediately, the two closest structural hits to **Ibuprofen** are **Naproxen** (`0.80`) and **Aspirin** (`0.80`) — the exact other two anti-inflammatory NSAIDs! Modern AI drug-discovery pipelines pair fingerprints, graph neural networks, and **AlphaFold 3** binding-pocket predictions to shrink billions of candidates down to a few dozen to synthesize in a wet lab.

Yet remember biology's humbling rule: **a computer prediction is a hypothesis, not a medicine.** Even after AI designs a promising molecule that binds a target tightly in a computer simulation, roughly **90% of drug candidates still fail in human clinical trials** due to unexpected toxicity, poor absorption, or complex whole-body physiology! AI speeds up the *search*, while rigorous wet-lab experiments and human trials remain irreplaceable.

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.4.1 (Modify) — How sensitive is our composition embedding to scrambling?
:class: hint
Create a completely scrambled version of `HBB_human` by reversing the string: `hbb_rev = proteins["HBB_human"][::-1]` (biologically non-functional gibberish!). Compute the cosine similarity between `protein_embedding(proteins["HBB_human"])` and `protein_embedding(hbb_rev)`. What number do you get, and what fundamental weakness of bag-of-letters embeddings does this expose?
```

```{admonition} Solution
:class: dropdown
```python
v1 = protein_embedding(proteins["HBB_human"])
v2 = protein_embedding(proteins["HBB_human"][::-1])
print(f"Cosine similarity (real HBB vs. reversed gibberish): {(v1 @ v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)):.4f}")
```

**Exactly `1.0000`!** Because our hand-built composition vector counts ingredients without looking at *order*, it is completely blind to sequence arrangement (just like `"listen"` and `"silent"` have identical letter counts). Real protein Transformers like **ESM-2** and **ESM3** use **positional encodings + self-attention** so letter order and long-range 3D contacts shape the embedding!
```

```{admonition} Exercise 4.4.2 (Make) — Screen for beta-lactam antibiotic analogs
:class: hint
Using `tanimoto()` and `drugs` from [](#ch44-drug-fingerprints), add a new candidate molecule `"Amoxicillin"` with fingerprint `np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])` (same core as Penicillin G plus one phenolic hydroxyl group!). Compute its Tanimoto similarity to every drug in `drugs` and print the closest match.
```

```{admonition} Solution
:class: dropdown
```python
amox = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
for name, fp in drugs.items():
    print(f"Tanimoto(Amoxicillin, {name:<18}) = {tanimoto(amox, fp):.2f}")
```

Amoxicillin matches **Penicillin G** at **0.89** — immediately explaining chemically *why* patients with severe penicillin allergies often cross-react to amoxicillin (connecting straight back to our Mini-RAG guideline in Chapter 4.3)!
```

```{admonition} Exercise 4.4.3 (Critical Reasoning) — Interpreting the "Virtual Cell" baseline lesson
:class: hint
In 2025, Ahlmann-Eltze et al. [@ahlmann2025perturbation] showed that giant single-cell foundation models trained on tens of millions of cells did not consistently beat a simple linear baseline at predicting unseen CRISPR gene-knockdown outcomes. Does this mean biological foundation models are useless — or what specific lesson should young scientists take away?
```

```{admonition} Solution
:class: dropdown
It definitely does *not* mean they are useless! Single-cell foundation models already shine at **cell-type annotation, multi-lab atlas harmonization, and exploratory embeddings**. The true lesson is twofold:
1. **Scale alone does not guarantee causal prediction:** Predicting an *intervention* (knocking out a gene) in a new biological context is a hard causal generalization problem, not just pattern matching.
2. **Simple baselines keep science honest:** Any time a flashy new billion-parameter bio-AI benchmark claims a breakthrough, demand a leak-free zero-shot comparison against a strong linear or classical baseline!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Foundation models** pretrain once on massive biological data (sequences, structures, single cells) and adapt to many downstream tasks via rich learned **embeddings**.
- **2024–2026 landmarks:** **AlphaFold 3** predicts full biomolecular complexes [@abramson2024alphafold3]; **ESM3** [@hayes2025esm3] and **RFdiffusion** [@watson2023rfdiffusion] design novel proteins; **Evo 2** [@brixi2026evo2] and **AlphaGenome** [@avsec2026alphagenome] decode non-coding DNA; **Geneformer** [@theodoris2023geneformer] and **scGPT** [@cui2024scgpt] map single cells.
- **Tanimoto similarity** over molecular fingerprints illustrates how AI accelerates virtual drug screening — though wet-lab assays and clinical trials remain essential.
- Rigorous benchmarking against simple baselines [@ahlmann2025perturbation] separates real progress from hype.
```

*End of Part 4!* You now understand how modern AI models data, images, language, DNA, proteins, and molecules. In **Part 5**, we confront the hardest and most important questions of all: **Who gets left behind? How do we protect patient privacy? How do humans and regulators keep health AI safe?** Let's open Part 5 with [](5_1-algorithmic-fairness.md).
