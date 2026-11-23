# TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences

A **full-semester (16-week)** interactive online textbook built with **Jupyter Book 2 (MyST)** for **advanced high-school students (grades 11–12)** and introductory undergraduates.

Designed around **PRIMM** (*Predict → Run → Investigate → Modify → Make*) pedagogy, **one clear big idea per chapter**, hand-worked arithmetic examples before code, plain-English definitions (zero calculus, zero linear algebra, zero unexplained medical jargon), and optional collapsible **Going Deeper** boxes for students who want extra math or technical depth.

## Course Structure (6 Units · 28 Bite-Sized Chapters · 6 Companion Colab Notebooks)

| Unit | Weeks | Lessons | Focus |
|---|---|---|---|
| **Unit 0** | Week 1 | 0.1 | Python & data warm-up (variables, lists, loops, functions, first chart) |
| **Unit 1** | Weeks 1–3 | 1.1 – 1.5 | Exponential vs. logistic growth, SIR outbreaks, clinical data exploration, correlation vs. causation, base rates & sample noise |
| **Unit 2** | Weeks 4–6 | 2.1 – 2.5 | k-NN & rescaling, train/test & overfitting, data leakage, confusion matrices & thresholds, lines vs. trees vs. forests |
| **Unit 3** | Weeks 7–9 | 3.1 – 3.5 | Fitting a line & MSE loss, calculus-free gradient descent, neural networks from scratch, 2D image filters & scans, shortcut learning |
| **Unit 4** | Weeks 10–12 | 4.1 – 4.6 | DNA/codons/sickle-cell `HBB`, sequence alignment, next-word prediction & temperature, embeddings & attention, hallucination & open-book RAG, biological foundation models (AlphaFold 3, ESM3, Evo 2, AlphaGenome, Virtual Cells) |
| **Unit 5** | Weeks 13–16 | 5.1 – 5.6 | Fairness & proxy bias, re-identification & k-anonymity, differential privacy & federated learning, humans-in-the-loop & abstention, clinical trials & governance, 2-week biomedical capstone |

## Local Development

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter book start --execute                 # live preview at http://localhost:3000
uv run jupyter book build --html --execute --strict # strict static HTML build into _build/html/
```

## License

- **Prose, diagrams, and figures:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code cells and companion notebooks:** [MIT License](https://opensource.org/licenses/MIT)
