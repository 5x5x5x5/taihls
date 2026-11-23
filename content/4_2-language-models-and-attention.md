---
title: 'Chapter 4.2 — How Large Language Models Work: Tokens, Embeddings, and Attention'
short_title: '4.2 · Language models & attention'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.2 — How Large Language Models Work: Tokens, Embeddings, and Attention

*Estimated time: ~50 minutes · Prerequisites: [](3_2-neural-networks-from-scratch.md), [](4_1-dna-proteins-and-alignment.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/03-clinical-llm-probabilities.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain **next-token prediction** as predicting a probability distribution over vocabulary tokens;
- build a tiny **bigram** clinical language model and explain why simple count tables hit a **data-sparsity wall**;
- visualize how sampling **temperature** reshapes a model's probability distribution from cautious to chaotic;
- run a 15-line pure-NumPy **self-attention** demo and explain how Transformers link distant words in a medical sentence.
```

## Start with a case: the doctor's new conversational colleague

In 2025, researchers published a landmark *Nature* study of **AMIE** (*Articulate Medical Intelligence Explorer*), a clinical dialogue AI evaluated in blinded simulated consultations against board-certified primary-care physicians [@tu2025amie]. Alongside open developer models like **MedGemma**, large language models (LLMs) now draft clinical visit summaries, explain biopsy jargon in plain language, and assist with differential diagnosis.

Yet beneath the fluent bedside manner, an LLM performs **one astonishingly simple core task**: given a sequence of {term}`Token`s, output a probability distribution for **what token comes next**, sample one, append it, and repeat!

```{mermaid}
flowchart LR
    Txt["Clinical Prompt<br/>'Patient has severe...'"] --> Tok["1. Tokenizer<br/>Split into subword<br/>tokens"]
    Tok --> Emb["2. Embeddings<br/>Turn tokens into<br/>meaning vectors"]
    Emb --> Attn["3. Transformer Blocks<br/>Self-attention links<br/>related words"]
    Attn --> Soft["4. Next-Token Probs<br/>Score every word in<br/>the vocabulary"]
    Soft -- "Sample with<br/>temperature T" --> Txt
```

Let's build **each piece** of that diagram in transparent Python!

## Lab 1: The simplest language model (bigram counts) & temperature

A **bigram** {term}`Language model` only looks back **one word**, counting how often every word followed it in a training corpus. Let's build one on tiny clinical notes — and also see how the **temperature** knob ($T$) controls how bold or cautious the model gets when sampling!

```{admonition} Definition — token and temperature
:class: important
- A {term}`Token` is one atomic chunk of text (a word or subword fragment like `amox` + `icillin`).
- {term}`Temperature` ($T > 0$) rescales a model's raw scores (*logits*) before converting them to probabilities: low $T$ ($0.25$) makes the top word nearly 100% certain; $T = 1.0$ uses the model's natural probabilities; high $T$ ($2.5$) flattens the curve so unlikely words get picked often!
```

```{admonition} Predict before you run
:class: tip
Suppose a clinical LLM assigns raw scores (logits) to four candidate next tokens after `"The patient was prescribed"`: `amoxicillin` (`4.2`), `ibuprofen` (`3.5`), `rest` (`2.4`), and `banana` (`0.2`). When we lower temperature to `T = 0.3` vs. raise it to `T = 2.5`, what happens to `P(amoxicillin)` and `P(banana)`?
```

```{code-cell} python
:label: ch42-bigram-temp
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

BLUE, VERMILION, TEAL, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"

# 1. Tiny bigram language model on clinic sentences
corpus = """
the patient has a fever and cough
the patient needs rest and fluids
the doctor checks the patient carefully
the nurse monitors the patient vitals
the patient feels better after rest
""".split()

following = defaultdict(Counter)
for cur, nxt in zip(corpus, corpus[1:]):
    following[cur][nxt] += 1

rng = np.random.default_rng(4)
def generate_bigram(start="the", length=7):
    out, w = [start], start
    for _ in range(length - 1):
        if w not in following:
            break
        words = list(following[w])
        counts = np.array([following[w][x] for x in words], dtype=float)
        w = rng.choice(words, p=counts / counts.sum())
        out.append(w)
    return " ".join(out)

print("Sampled bigram clinic sentences:")
for _ in range(3):
    print("  •", generate_bigram("the"))

# 2. Temperature scaling via softmax on candidate next-token logits
candidates = ["amoxicillin", "ibuprofen", "rest", "banana"]
logits = np.array([4.2, 3.5, 2.4, 0.2])

def softmax_temp(z, T):
    scaled = z / T
    ex = np.exp(scaled - scaled.max())
    return ex / ex.sum()

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True)
for ax, T, col, desc in [
    (axes[0], 0.3, BLUE, "T = 0.3 (cautious / focused)"),
    (axes[1], 1.0, TEAL, "T = 1.0 (balanced)"),
    (axes[2], 2.5, VERMILION, "T = 2.5 (flat / wild!)"),
]:
    probs = softmax_temp(logits, T)
    bars = ax.bar(candidates, probs * 100, color=col, alpha=0.85)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, p * 100 + 2, f"{p:.0%}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_title(desc, fontsize=10)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Next-token probability (%)")
fig.suptitle('Prompt: "The patient was prescribed ___" — how temperature shapes sampling', fontsize=10.5, y=1.02)
fig.tight_layout()
plt.show()
```

Look at why simple count tables hit a brick wall: our bigram model only remembers **1 word**! If we try to remember the previous **10 words** with a count table over a 50,000-word vocabulary, there are $50{,}000^{10}$ possible contexts — almost none of which will *ever* appear twice in human history (**data sparsity**)!

## Lab 2: Inside the Transformer — Embeddings & Self-Attention!

How did the **Transformer** architecture solve the long-memory problem and launch the modern AI era? Two brilliant ideas:

1. **{term}`Embedding`s:** Instead of treating `"penicillin"` and `"amoxicillin"` as unrelated dictionary IDs, the network maps every token to a list of numbers (a vector) so **related concepts point in similar directions**.
2. **Self-{term}`Attention`:** Instead of blindly compressing a whole paragraph into one fixed summary, every word gets to **look at every other word in the sentence** and ask: *"Which earlier words are most relevant to understanding ME right now?"*

In algebra, self-attention is just **3 lines of NumPy**:
- Measure how well word $i$'s **Query** ("what am I looking for?") aligns with word $j$'s **Key** ("what information do I carry?") by taking their dot product (multiplying matching entries and summing!).
- Pass those compatibility scores through `softmax` so every row adds up to `100%` — those percentages are the **attention weights**!
- Use those weights to take a focused weighted average of the relevant words!

Let's watch self-attention work on a high-stakes clinical sentence:
> `"patient with severe penicillin allergy must avoid amoxicillin"`

We assign each word a transparent 4-number meaning vector representing four intuitive concepts: `[drug/antibiotic, safety/warning, patient/person, clinical modifier]`. Watch which earlier words `"amoxicillin"` automatically attends to!

```{admonition} Predict before you run
:class: tip
When the final token `"amoxicillin"` computes its attention weights over the sentence `"patient with severe penicillin allergy must avoid amoxicillin"`, which two earlier words should light up brightest in the attention heatmap?
```

```{code-cell} python
:label: ch42-attention-demo
tokens = ["patient", "with", "severe", "penicillin", "allergy", "must", "avoid", "amoxicillin"]

# Transparent 4D toy embeddings: [antibiotic/drug, safety/allergy, person, intensity]
E = np.array([
    [0.0, 0.1, 1.0, 0.1],  # patient
    [0.0, 0.0, 0.1, 0.1],  # with
    [0.0, 0.5, 0.0, 1.0],  # severe
    [1.0, 0.6, 0.0, 0.1],  # penicillin (antibiotic + linked to allergy context)
    [0.5, 1.0, 0.2, 0.6],  # allergy (strong safety warning!)
    [0.0, 0.3, 0.0, 0.4],  # must
    [0.4, 0.9, 0.0, 0.4],  # avoid (safety action!)
    [1.0, 0.5, 0.0, 0.1],  # amoxicillin (penicillin-class antibiotic!)
])

# Scaled dot-product self-attention (the core equation of Transformers!)
scale = np.sqrt(E.shape[1])
scores = (E @ E.T) / scale                    # similarity of every token pair

# Causal mask: each token can only look at itself and EARLIER tokens (no peeking ahead!)
mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
scores_masked = np.where(mask, -1e9, scores)

# Softmax row by row so attention weights sum to 1.0
ex = np.exp(scores_masked - scores_masked.max(axis=1, keepdims=True))
attn = ex / ex.sum(axis=1, keepdims=True)

print("Where 'amoxicillin' focuses its attention (earlier words):")
for tok, w in zip(tokens, attn[-1]):
    print(f"  {tok:<12}: {w:5.1%}  {'█' * int(w * 40)}")

fig, ax = plt.subplots(figsize=(6.8, 5.6))
im = ax.imshow(attn, cmap="YlGnBu", vmin=0, vmax=0.35)
ax.set_xticks(range(len(tokens)), tokens, rotation=35, ha="right")
ax.set_yticks(range(len(tokens)), tokens)
for i in range(len(tokens)):
    for j in range(i + 1):
        ax.text(j, i, f"{attn[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if attn[i, j] > 0.20 else "#111111")
ax.set_xlabel("Attended-to earlier tokens (Keys)")
ax.set_ylabel("Current token (Queries)")
ax.set_title("Causal self-attention heatmap on a clinical safety sentence")
fig.colorbar(im, ax=ax, label="Attention weight", shrink=0.82)
fig.tight_layout()
plt.show()
```

**Look at the bottom row (`amoxicillin`) of [](#ch42-attention-demo)!**
Even though `"penicillin"` and `"allergy"` sat **three to four words earlier** — completely invisible to a bigram model! — self-attention reached straight back across the sentence, ignored filler words like `"with"`, and placed high attention weight onto **`penicillin`**, **`allergy`**, and **`avoid`**!

In a real clinical LLM (like **MedGemma** or **AMIE** [@tu2025amie]), the model doesn't have just 4 hand-coded embedding numbers and 1 attention layer — it learns thousands of embedding dimensions and dozens of stacked multi-head attention layers via gradient descent across trillions of tokens! Yet the engine inside every single layer is the exact 5-line dot-product + softmax calculation you just ran.

```{admonition} Inspect a real LLM's live next-token probabilities (Companion Colab Notebook 3)
:class: tip
Want to watch a real pretrained Transformer generate clinical completions and display its live top-10 next-token probability distributions? Open [Companion Notebook 03](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/03-clinical-llm-probabilities.ipynb) on Google Colab and test temperature sweeps and medical facts firsthand!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.2.1 (Modify) — What happens when temperature approaches zero?
:class: hint
In [](#ch42-bigram-temp), call `softmax_temp(logits, T)` for `T = 0.05` and `T = 10.0`. Why do clinical documentation and medical coding systems usually run LLMs at very low temperature (`T = 0.0` to `0.2`) rather than `T = 1.5`?
```

```{admonition} Solution
:class: dropdown
```python
print("T = 0.05:", np.round(softmax_temp(logits, 0.05), 3))
print("T = 10.0:", np.round(softmax_temp(logits, 10.0), 3))
```

At `T = 0.05`, the single top token gets `~100%` probability (**greedy decoding** — deterministic and repeatable), whereas `T = 10.0` turns the probabilities nearly uniform (`~25%` each, including `"banana"`!). Clinical workflows prefer low temperature for consistency and repeatability, though low temperature does **not** eliminate hallucination — it merely picks the single most plausible-sounding continuation every time!
```

```{admonition} Exercise 4.2.2 (Make) — Measure embedding similarity between two drugs
:class: hint
Write a 3-line cosine similarity function `cos(a, b) = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))` and compute the similarity between `"penicillin"` (`E[3]`) and `"amoxicillin"` (`E[7]`) versus `"penicillin"` and `"patient"` (`E[0]`).
```

```{admonition} Solution
:class: dropdown
```python
def cos(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"cos(penicillin, amoxicillin) = {cos(E[3], E[7]):.3f}")
print(f"cos(penicillin, patient)     = {cos(E[3], E[0]):.3f}")
```

`penicillin` and `amoxicillin` score nearly **1.00** (close neighbors in embedding space!), while `penicillin` and `patient` score near **0.05** (almost perpendicular — unrelated concepts!).
```

```{admonition} Exercise 4.2.3 (Critical Reasoning) — Why causal masking matters during training
:class: hint
In [](#ch42-attention-demo), we masked out the upper-right triangle (`scores_masked = np.where(mask, -1e9, scores)`) so each word could only attend to *earlier* words, never future ones. Why would training a next-token predictor *without* that causal mask be a textbook case of **data leakage** (Chapter 2.2)?
```

```{admonition} Solution
:class: dropdown
If token $t$ could attend to token $t+1$ while being trained to *predict* token $t+1$, it would simply peek at the answer key and copy it with 100% training accuracy — learning nothing about language! (Interestingly, **protein language models** like ESM-2 sometimes remove the causal mask and instead hide random middle letters — called *masked language modeling* — because a protein sequence folds as a whole rather than being read strictly left to right!)
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **language model** predicts a probability distribution over the next **token**; sampling **temperature** trades focused predictability against risky randomness.
- Simple n-gram count tables fail on long context due to **data sparsity**.
- **Transformers** overcome this with **embeddings** (vectors encoding meaning) and **self-attention** (letting every token dynamically pull information from relevant earlier tokens across long distances).
- Clinical LLMs like **AMIE** [@tu2025amie] and **MedGemma** scale this exact mechanism — yet next-token prediction comes with a built-in Achilles' heel.
```

*Next up:* Because a language model predicts *likely-sounding* tokens rather than consulting a truth database, it can fabricate medical facts with total confidence. Let's dissect **hallucination** and build a working **Retrieval-Augmented Generation (RAG)** defense in [](4_3-hallucination-grounding-rag.md)!
