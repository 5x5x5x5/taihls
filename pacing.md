---
title: 'Course Map & Pacing Guide'
short_title: 'Course map & pacing'
---

# Course Map & Pacing Guide

Whether you are teaching a university seminar, running an advanced high-school elective, or studying independently over a break, this guide maps how the **20 chapters** and **6 companion Colab notebooks** fit together.

## How the five parts connect

TAIHLS uses a **spiral curriculum**: core ideas (baselines, data quality, representation, uncertainty, and fairness) appear early in simple form and return at deeper sophistication across imaging, sequences, foundation models, and clinical governance.

```{mermaid}
flowchart TD
    P1["Part 1: Data, Uncertainty & Causality<br/>(Ch 1.1 – 1.4)"] --> P2["Part 2: Classical ML in the Clinic<br/>(Ch 2.1 – 2.4)"]
    P2 --> P3["Part 3: Deep Learning & Medical Imaging<br/>(Ch 3.1 – 3.4)"]
    P2 --> P4["Part 4: Sequences, Language & Foundation Models<br/>(Ch 4.1 – 4.4)"]
    P3 --> P4
    P3 --> P5["Part 5: Responsible, Safe & Equitable Health AI<br/>(Ch 5.1 – 5.4)"]
    P4 --> P5
```

## Complete chapter & companion notebook map

| Chapter | Core Theme | Est. Time | Companion Colab Notebook |
|---|---|---:|---|
| **1.1** Outbreak Models | Exponential, logistic & SIR modeling assumptions | 35 min | — (in-browser) |
| **1.2** Exploring Clinical Data | Distributions, skew, outliers & visual integrity | 35 min | — (in-browser) |
| **1.3** Correlation & Causation | Confounding, causal DAGs, Simpson's paradox, RCTs | 40 min | — (in-browser) |
| **1.4** Probability & Base Rates | Natural frequencies, sequential tests, bootstrap CIs | 40 min | — (in-browser) |
| **2.1** Nearest Neighbors | k-NN from scratch, feature scaling, distance spaces | 40 min | — (in-browser) |
| **2.2** Generalization & Leakage | Overfitting, cross-validation, patient-level leakage | 45 min | — (in-browser) |
| **2.3** Clinical Metrics & Thresholds | Confusion matrices, ROC, PR curves & calibration | 45 min | — (in-browser) |
| **2.4** Trees, Ensembles & Boundaries | Linear vs. trees vs. forests & feature importance | 45 min | — (in-browser) |
| **3.1** Loss & Gradient Descent | Dose–response MSE, calculus-free descent, learning rate | 45 min | — (in-browser) |
| **3.2** Neural Networks from Scratch | Neurons, ReLU/sigmoid, training on curved bio data | 50 min | `01-pytorch-neural-net.ipynb` |
| **3.3** Medical Imaging & CNNs | Pixels, 2D convolutions by hand, CNNs, ViTs, MASAI | 50 min | `02-medical-images-cnn.ipynb` |
| **3.4** Shortcuts & Failure Modes | Hospital tag shortcut lab, saliency, distribution shift | 45 min | — (in-browser) |
| **4.1** DNA, Proteins & Alignment | Real HBB sequence, codons, k-mers, DP edit distance | 45 min | — (in-browser) |
| **4.2** Language Models & Attention | Bigrams → Transformers, self-attention, temperature | 50 min | `03-clinical-llm-probabilities.ipynb` |
| **4.3** Hallucination & Mini-RAG | Why LLMs invent facts, building a grounded mini-RAG | 45 min | — (in-browser) |
| **4.4** Biological Foundation Models | AlphaFold 3, ESM3, Evo 2, AlphaGenome, Virtual Cells | 55 min | `04-protein-foundation-models-esm2.ipynb` |
| **5.1** Algorithmic Fairness | Proxy-label bias, subgroup metrics, health equity | 45 min | — (in-browser) |
| **5.2** Privacy-Enhancing Tech | Re-identification, k-anonymity, DP, federated learning | 50 min | — (in-browser) |
| **5.3** Humans, Trials & Regulation | Abstention, automation bias, FDA PCCP, EU AI Act | 45 min | `05-fairness-and-abstention-audit.ipynb` |
| **5.4** Biomedical Capstone | 3 end-to-end health/bio tracks + Model Card rubric | 3–6 hrs | `06-capstone-starter.ipynb` |

## Suggested pacing schedules

::::{tab-set}

:::{tab-item} 15-Week College Semester
- **Weeks 1–3:** Part 1 (Ch 1.1–1.4) — scientific reasoning & Python fluency.
- **Weeks 4–6:** Part 2 (Ch 2.1–2.4) — classical ML + first mini-audit assignment.
- **Weeks 7–9:** Part 3 (Ch 3.1–3.4) + Colab Labs 1 & 2 (PyTorch MLP & Chest X-ray CNN).
- **Weeks 10–12:** Part 4 (Ch 4.1–4.4) + Colab Labs 3 & 4 (Clinical LLM & ESM-2).
- **Weeks 13–14:** Part 5 (Ch 5.1–5.3) + Colab Lab 5 (Fairness & Abstention Audit).
- **Week 15:** Ch 5.4 Biomedical Capstone presentations & peer Model Card review.
:::

:::{tab-item} 30-Week High-School Honors Course
Spend **1 to 1.5 weeks per chapter**:
- **Day 1:** Read the story, Mermaid diagram, and complete the **Predict** prompts as a class discussion.
- **Day 2:** Run & Investigate the in-page Python labs in pairs; dissect the plots.
- **Day 3:** Complete the **Modify** & **Make** exercises (or GPU Colab notebook on paired chapters).
- **Day 4:** Discuss the **Critical Reasoning** exercise and current medical AI news.
- **Final 4 Weeks:** Guided biomedical capstone (Ch 5.4) with milestone check-ins.
:::

:::{tab-item} 4-Week Self-Study Sprint
- **Week 1:** Part 1 + Ch 2.1–2.2 (Foundations & honest evaluation).
- **Week 2:** Ch 2.3–2.4 + Part 3 (Metrics, neural nets, imaging & shortcuts).
- **Week 3:** Part 4 (Sequences, Transformers, RAG & bio foundation models).
- **Week 4:** Part 5 + Capstone notebook (Fairness, privacy, governance & portfolio project).
:::

::::
