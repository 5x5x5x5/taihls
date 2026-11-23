---
title: 'Chapter 4.3 — How Language Models Write: Tokens, Probabilities, and Temperature'
short_title: '4.3 · Next-word prediction'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.3 — How Language Models Write: Tokens, Probabilities, and Temperature

*Week 11 · ~40 minutes · Prerequisites: [](4_1-dna-and-proteins-as-text.md)*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/03-clinical-llm-probabilities.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **language model** does in one plain sentence (**next-token prediction**);
- build a tiny **bigram** word-counting model from scratch and sample new clinic sentences from it;
- explain why simple word-counting tables break down on longer sentences (**data sparsity**);
- describe how the **temperature** dial controls whether an AI plays it safe or takes creative risks.
```

## Start with a case: autocomplete on steroids

Every time your phone finishes `"See you ___"` with **`soon`** or **`tomorrow`**, you are watching a {term}`Language model` at work. Surprisingly, the giant AI chatbots that summarize doctor visits or answer biology questions are built on that **exact same core game**:

> **Read the text so far, assign a probability to every possible next {term}`Token` (word or word piece), pick one, and repeat!**

Let's strip away the mystery by building a miniature word predictor right on this page.

## Count by hand first: the 1-word-memory (bigram) model

Suppose our entire training library consists of just four tiny clinic notes:
- *"the patient has fever"*
- *"the patient needs rest"*
- *"the doctor helps the patient"*
- *"the patient feels better"*

Look only at the word **`"the"`** (`5` appearances). Which word came right after `"the"` each time?
- **`patient`** appeared **4 times** → probability = $4 / 5 = \mathbf{80\%}$
- **`doctor`** appeared **1 time** → probability = $1 / 5 = \mathbf{20\%}$

Notice those chances add up to **100% (`1.0`)** — a **probability distribution**! A model that looks back just **one** word like this is called a **bigram** model.

## Build and sample from a clinic bigram model!

```{admonition} Predict before you run
:class: tip
In our tiny clinic corpus below, the word `"patient"` is followed by several different words (`has`, `needs`, `feels`, `carefully`, `vitals`). When we call `generate_sentence("the")` three times, will it print the **exact same sentence** every time, or slightly different plausible-sounding clinic sentences?
```

```{code-cell} python
:label: ch43-bigram
from collections import defaultdict, Counter
import numpy as np

corpus = """
the patient has a fever and cough
the patient needs rest and fluids
the doctor checks the patient carefully
the nurse monitors the patient vitals
the patient feels better after rest
""".split()

# Count which words follow every word
next_words = defaultdict(Counter)
for current, nxt in zip(corpus, corpus[1:]):
    next_words[current][nxt] += 1

print('Words seen right after "the":', dict(next_words["the"]))

rng = np.random.default_rng(4)

def generate_sentence(start_word="the", length=7):
    words = [start_word]
    cur = start_word
    for _ in range(length - 1):
        if cur not in next_words:
            break
        choices = list(next_words[cur].keys())
        counts  = np.array(list(next_words[cur].values()), dtype=float)
        probs   = counts / counts.sum()          # turn counts into probabilities summing to 1
        cur = rng.choice(choices, p=probs)       # weighted coin flip!
        words.append(cur)
    return " ".join(words)

print("\nGenerated clinic sentences:")
for _ in range(4):
    print("  •", generate_sentence("the"))
```

Out of pure counting and weighted coin flips, our 15-line program writes smooth-sounding English (*"the doctor checks the patient feels better"*) — even though it has **zero idea** what a doctor or a patient is!

### Why simple count tables hit a wall ("data sparsity")

Notice why `"the doctor checks the patient feels better"` went slightly sideways at the end: looking back only **one word** meant the model forgot `"the doctor checks"`!

Why not just count **10-word** histories instead? With a 50,000-word vocabulary, there are $50{,}000^{10}$ possible 10-word phrases — more than atoms in the universe! Almost every real sentence you type would have a count of **zero** (**data sparsity**). That is why modern AI replaced count tables with **neural networks** (Chapter 4.4!).

## Turning the creativity dial: Temperature

When a real LLM predicts scores for the next word after `"The doctor prescribed ___"`, we can adjust how adventurously it samples using a dial called {term}`Temperature`:
- **Low temperature (`T = 0.3`):** Concentrates almost all probability on the #1 safest guess (**predictable**, great for medical notes).
- **Normal temperature (`T = 1.0`):** Uses the model's natural probabilities.
- **High temperature (`T = 2.5`):** Flattens the bars so even silly/rare words get picked (**wild**!).

```{code-cell} python
:label: ch43-temperature
import matplotlib.pyplot as plt

candidates = ["antibiotics", "ibuprofen", "rest", "banana"]
raw_scores = np.array([4.2, 3.5, 2.4, 0.2])    # model's raw preference scores

def to_probs(scores, T):
    scaled = scores / T
    weights = np.exp(scaled - scaled.max())
    return weights / weights.sum()

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
for ax, T, col, label in [
    (axes[0], 0.3, "#0072B2", "T = 0.3 (safe / focused)"),
    (axes[1], 1.0, "#009E73", "T = 1.0 (balanced)"),
    (axes[2], 2.5, "#D55E00", "T = 2.5 (wild / risky!)"),
]:
    probs = to_probs(raw_scores, T) * 100
    bars = ax.bar(candidates, probs, color=col, alpha=0.85)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width()/2, p + 2, f"{p:.0f}%", ha="center", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_title(label, fontsize=9.8)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Next-word chance (%)")
fig.suptitle('After "The doctor prescribed ___": how temperature reshapes probabilities', y=1.02)
fig.tight_layout()
plt.show()
```

```{admonition} Going deeper — run a real language model on Colab!
:class: dropdown
Our helper `to_probs` above is the famous **softmax** function used at the final layer of every LLM (`exp(score / T)` divided by the sum). Open [Companion Colab Notebook 03](https://colab.research.google.com/github/5x5x5x5/taihls/blob/curriculum-v2/notebooks/03-clinical-llm-probabilities.ipynb) to inspect the **live top-10 next-token probabilities** of a real pretrained Transformer (`distilgpt2`) on medical prompts!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.3.1 (Modify) — Inspect the distribution after `"patient"`
:class: hint
Print `dict(next_words["patient"])` from [](#ch43-bigram). Which words follow `"patient"`, and is the model more certain after `"the"` or after `"patient"`?
```

```{admonition} Solution
:class: dropdown
```python
print(dict(next_words["patient"]))
```
`"patient"` splits evenly across **5** different next words (`has`, `needs`, `carefully`, `vitals`, `feels` — 20% each!), so the model is much less certain after `"patient"` than after `"the"` (where `"patient"` dominates!).
```

```{admonition} Exercise 4.3.2 (Make) — What happens at `T = 0.05` vs `T = 10`?
:class: hint
Call `to_probs(raw_scores, T)` from [](#ch43-temperature) for `T = 0.05` and `T = 10.0`. Describe both extremes in one sentence!
```

```{admonition} Solution
:class: dropdown
```python
print("T = 0.05:", np.round(to_probs(raw_scores, 0.05) * 100, 1))
print("T = 10.0:", np.round(to_probs(raw_scores, 10.0) * 100, 1))
```
At `T = 0.05`, `"antibiotics"` takes **100%** (always picks the top word); at `T = 10.0`, all four choices flatten near **25%** each — making `"banana"` almost as likely as real medicine!
```

```{admonition} Exercise 4.3.3 (Critical Reasoning) — Why use low temperature in hospitals?
:class: hint
When hospitals use an AI assistant to summarize a patient's clinic visit, they usually turn **temperature down near `0.0`–`0.2`** instead of `1.5`. Why? And does low temperature guarantee the summary has zero mistakes?
```

```{admonition} Solution
:class: dropdown
Low temperature makes the summary steady, consistent, and less likely to wander onto bizarre tangents. However, **it does NOT guarantee truth!** If the model's #1 highest-probability guess is factually wrong, low temperature simply repeats that wrong guess confidently every single time (we'll tackle hallucinations in Chapter 4.5!).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- A **language model** predicts a probability distribution for the next **token** and writes by sampling one token at a time.
- Simple word-counting tables (**bigrams**) break on longer context because of **data sparsity**.
- **Temperature** dials sampling from focused and predictable (low $T$) to flat and adventurous (high $T$).
```

*Next up:* How did modern AI (like ChatGPT, MedGemma, and AMIE) break past the 1-word memory limit? By giving words **meaning maps (embeddings)** and **attention** — let's build both in plain algebra in [](4_4-embeddings-and-attention.md)!
