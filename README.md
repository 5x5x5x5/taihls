# TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences

An interactive, open-source online textbook built with **Jupyter Book 2 (MyST)** for advanced high-school students and first-year undergraduates. Designed around **PRIMM** (*Predict–Run–Investigate–Modify–Make*) pedagogy, dual coding (conceptual diagrams + live runnable code + publication-grade visualizations), and current 2024–2026 science across medicine and biology.

## Course Structure (5 Parts · 20 Chapters · 6 Companion Colab Notebooks)

| Part | Focus | Chapters |
|------|-------|----------|
| **Part 1** | Data, Uncertainty, and Causality | Outbreak models (SIR), clinical data exploration, confounding & Simpson's paradox, base rates & bootstrap uncertainty |
| **Part 2** | Classical Machine Learning in the Clinic | k-NN & scaling, overfitting & data leakage, thresholds/ROC/PR/calibration, decision boundaries & ensembles |
| **Part 3** | Deep Learning and Medical Imaging | Loss landscapes & calculus-free gradient descent, neural networks from scratch, convolutions & medical imaging, shortcut learning & saliency |
| **Part 4** | Sequences, Language, and Foundation Models | DNA/proteins & DP alignment, Transformers/attention/temperature, hallucination & mini-RAG, biological foundation models (AlphaFold 3, ESM3, Evo 2, AlphaGenome, Virtual Cells) |
| **Part 5** | Responsible, Safe, and Equitable Health AI | Proxy & representation bias, k-anonymity / differential privacy / federated learning, human-in-the-loop & regulation (FDA PCCPs, EU AI Act, TRIPOD+AI), biomedical capstone |

## Project Layout

| Path | Purpose |
|------|---------|
| `myst.yml` | Project configuration, branding, bibliography, JupyterLite/Binder settings |
| `toc.yml` | Book table of contents (imported via `extends:`) |
| `index.md`, `preface.md`, `advice.md`, `pacing.md` | Front matter and instructor/self-study pacing guide |
| `content/` | 20 executable MyST chapters (`1_1` through `5_4`) |
| `notebooks/` | 6 companion Google Colab notebooks for GPU/deep-dive labs and capstone |
| `back/` | Unified interactive glossary, reproducibility guide, references, acknowledgements, license |
| `assets/` | Custom SVG logos, favicon, and subtle CSS polish |
| `references.bib` | Verified BibTeX bibliography of landmark studies (2002–2026) |

## Local Development

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                      # install exact pinned dependencies
uv run jupyter book start --execute          # live preview at http://localhost:3000
uv run jupyter book build --html --execute --strict   # full static HTML build
```

## Two-Tier Execution for Readers

1. **Tier 1 (In-Page / Browser via JupyterLite):** Every chapter runs lightweight Python (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`) with built-in or self-contained biomedical datasets — zero installation, zero downloads.
2. **Tier 2 (Google Colab Companion Notebooks):** GPU-accelerated labs (`notebooks/*.ipynb`) train real PyTorch models on chest X-rays (MedMNIST), inspect live LLM next-token probabilities, compute ESM-2 protein embeddings, and provide the capstone audit workflow.

## License

- **Prose, figures, and instructional content:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code cells and notebooks:** [MIT License](https://opensource.org/licenses/MIT)
