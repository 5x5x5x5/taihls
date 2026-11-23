---
title: '16-Week Semester Pacing Guide'
short_title: 'Semester pacing guide'
---

# 16-Week Semester Pacing Guide

**TAIHLS** is paced as a **full-semester (16-week) advanced high-school or introductory college course**: roughly **two bite-sized chapters per week** across Weeks 1–14, followed by a **two-week Capstone Project** in Weeks 15–16.

Each chapter is designed for **one 45–50 minute class period** (or ~30–40 minutes of focused self-study), with natural breaks for partner discussions, hands-on code tinkering, and six optional Google Colab lab days.

## How the six units build on each other

```{mermaid}
flowchart TD
    U0["Unit 0: Python Warm-Up<br/>(Week 1)"] --> U1["Unit 1: Models, Data & Causality<br/>(Weeks 1–3 · 5 lessons)"]
    U1 --> U2["Unit 2: Classical Machine Learning<br/>(Weeks 4–6 · 5 lessons)"]
    U2 --> U3["Unit 3: Neural Nets & Imaging<br/>(Weeks 7–9 · 5 lessons)"]
    U2 --> U4["Unit 4: DNA, Language & Bio AI<br/>(Weeks 10–12 · 6 lessons)"]
    U3 --> U4
    U3 --> U5["Unit 5: Fairness, Privacy & Capstone<br/>(Weeks 13–16 · 6 lessons)"]
    U4 --> U5
```

## Week-by-week 16-week schedule

| Week | Unit | Lessons | Big Idea & Companion Lab |
|---|---|---|---|
| **Week 1** | Unit 0 & 1 | **0.1** Python Warm-Up · **1.1** Exponential & Logistic Growth | Python basics + how simple growth models work |
| **Week 2** | Unit 1 | **1.2** Simulating an Outbreak (SIR) · **1.3** Exploring Clinical Data | Flattening the curve + mean, median, skew & honest charts |
| **Week 3** | Unit 1 | **1.4** Correlation & Confounding · **1.5** Base Rates & Uncertainty | Simpson's paradox, RCTs, natural frequencies & sample size |
| **Week 4** | Unit 2 | **2.1** Nearest Neighbors · **2.2** Train/Test & Overfitting | First tumor classifier, rescaling & honest grading |
| **Week 5** | Unit 2 | **2.3** Data Leakage · **2.4** Mistakes & Thresholds | Catching fake 97% scores + confusion matrices & thresholds |
| **Week 6** | Unit 2 | **2.5** Lines, Trees & Forests + *Unit 1–2 Review / Quiz* | Comparing 3 clinical models + feature importance |
| **Week 7** | Unit 3 | **3.1** Fitting a Line & Loss · **3.2** Gradient Descent | MSE loss valley + rolling downhill without calculus |
| **Week 8** | Unit 3 | **3.3** Inside a Neural Network + **Colab Lab 01 (PyTorch)** | Neurons, ReLU/sigmoid, training a curved boundary |
| **Week 9** | Unit 3 | **3.4** How Computers See Scans + **Colab Lab 02 (Chest X-rays)** · **3.5** Shortcuts | Hand-coded 2D image filter + catching a cheating hospital AI |
| **Week 10** | Unit 4 | **4.1** DNA & Proteins as Text · **4.2** Comparing Sequences | Real human hemoglobin, sickle-cell mutation & edit distance |
| **Week 11** | Unit 4 | **4.3** Next-Word Prediction + **Colab Lab 03 (LLM)** · **4.4** Embeddings & Attention | Tokens, temperature, meaning maps & plain-English attention |
| **Week 12** | Unit 4 | **4.5** Hallucination & Open-Book RAG · **4.6** Foundation Models + **Colab Lab 04 (ESM-2)** | Grounded lookup + AlphaFold 3, ESM3, Evo 2, AlphaGenome |
| **Week 13** | Unit 5 | **5.1** Fairness & Bias · **5.2** Anonymity & k-Anonymity | Wrong-label & missing-group bias + re-identification |
| **Week 14** | Unit 5 | **5.3** Differential Privacy & Federated Learning · **5.4** Humans in the Loop + **Colab Lab 05** | Coin-flip noise, 3-hospital learning & saying "I don't know" |
| **Week 15** | Unit 5 | **5.5** From Lab to Hospital · **5.6** Capstone Launch + **Colab Lab 06** | Clinical trials, safety checklist & Capstone Milestones 1–3 |
| **Week 16** | Unit 5 | **5.6** Capstone Completion & Class Showcase | Milestones 4–5, peer Model Card audits & final presentations |

## Tips for classroom teachers

- **Typical 50-minute period:** 5 min hook story → 10 min hand-worked example on the board → 20 min paired PRIMM coding (*Predict → Run → Modify*) → 10 min Exercise 3 (*Critical Reasoning*) class discussion → 5 min exit ticket.
- **Differentiation built in:** Every chapter's core path requires only high-school algebra and basic Python. Students who finish early can explore the collapsible **Going Deeper** boxes or try the bonus **Make** exercises.
- **Zero IT headaches:** All 28 chapters run directly in the browser (JupyterLite) or locally via `uv` with zero downloads or API keys. The 6 Colab notebooks only need a free Google account (and Notebooks 05–06 don't even require a GPU).
