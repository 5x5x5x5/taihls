---
title: 'Chapter 4.4 — How Modern AI Understands Context: Embeddings and Attention'
short_title: '4.4 · Embeddings & attention'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.4 — How Modern AI Understands Context: Embeddings and Attention

*Week 11 · ~40 minutes · Prerequisites: [](4_3-predicting-the-next-word.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what an **embedding** is using a simple 2D **meaning map** of health words;
- explain how **attention** lets a model link related words far apart in a medical sentence;
- trace a plain-arithmetic **attention match score** by hand and plot which earlier words `"amoxicillin"` focuses on;
- connect these two ideas to modern clinical dialogue models like **AMIE** and **MedGemma**.
```

## Start with a case: reading across a whole medical sentence

Read this critical safety sentence from a patient's chart:

> *"The patient with a severe **penicillin** **allergy** must **avoid** **amoxicillin**."*

To realize why **amoxicillin** is dangerous here, you have to connect two clues:
1. You know **amoxicillin** and **penicillin** belong to the *same antibiotic family*.
2. Even though `"penicillin"` and `"allergy"` appeared **four words earlier** — completely invisible to Chapter 4.3's bigram model! — your brain reaches back across the sentence and links them straight to `"amoxicillin"`.

Modern AI language models (called **Transformers**) achieve both tricks using two remarkably clean ideas: **embeddings** and **attention**.

## Idea 1: Embeddings (putting words on a meaning map)

In a bigram count table, `"penicillin"` and `"amoxicillin"` are just two unrelated dictionary entries. An {term}`Embedding` fixes that by giving every word **coordinates on a meaning map** — so words with similar meaning land right next to each other!

Imagine a super-simple 2-number map where:
- **x-coordinate** = *"How much is this a **medicine**?"* (`0` to `10`)
- **y-coordinate** = *"How much is this about **fighting infection / bacteria**?"* (`0` to `10`)

Let's plot six everyday health words on that 2D meaning map!

```{code-cell} python
:label: ch44-word-map
import matplotlib.pyplot as plt

# Simple 2D meaning coordinates: (medicine-ness, infection-ness)
word_map = {
    "penicillin":  (9.2, 9.5),
    "amoxicillin": (9.0, 9.0),
    "aspirin":     (9.0, 1.5),
    "ibuprofen":   (8.6, 1.2),
    "fever":       (1.2, 7.8),
    "cough":       (1.5, 6.5),
}

plt.figure(figsize=(6.8, 4.2))
for word, (x, y) in word_map.items():
    plt.scatter(x, y, s=110, color="#0072B2")
    plt.text(x + 0.25, y + 0.15, word, fontsize=10, fontweight="bold")

plt.xlim(0, 11.2); plt.ylim(0, 11)
plt.xlabel("Medicine-ness (0 = symptom, 10 = drug)")
plt.ylabel("Infection / antibiotic-ness")
plt.title("A 2D embedding map: related medical words land close together!")
plt.grid(alpha=0.25)
plt.show()
```

Look how cleanly the three neighborhoods appear: **antibiotics** (`penicillin`, `amoxicillin`) sit side by side in the top-right, **pain relievers** (`aspirin`, `ibuprofen`) cluster in the bottom-right, and **symptoms** (`fever`, `cough`) sit together on the left! Real LLMs use **hundreds or thousands** of learned coordinates instead of 2 — capturing chemistry, grammar, tone, and medical facts automatically!

## Idea 2: Attention (spotting which earlier words matter right now!)

Now suppose the model reaches the word **`"amoxicillin"`** at the end of:
`["patient", "with", "severe", "penicillin", "allergy", "must", "avoid", "amoxicillin"]`

{term}`Attention` asks a simple question for every earlier word:
> **How strongly does this earlier word's meaning match what `"amoxicillin"` cares about?**

We measure a **match score** with ordinary multiplication and addition: give each word a tiny 3-number checklist `[is_antibiotic, is_safety_warning, is_person]`, multiply matching checklist items together, and add!

**Try two pairs by hand!**
- `"amoxicillin"` = `[1.0, 0.5, 0.0]`
- Match with `"with"` (`[0.0, 0.0, 0.0]`): $1.0\times 0 + 0.5\times 0 + 0\times 0 = \mathbf{0.0}$ (ignore!)
- Match with `"penicillin"` (`[1.0, 0.6, 0.0]`): $1.0\times 1.0 + 0.5\times 0.6 + 0\times 0 = \mathbf{1.3}$ (**strong match!**)

Let's compute the match score between `"amoxicillin"` and *every* earlier word, convert those scores into percentages that sum to 100% (**attention weights**), and draw a bar chart!

```{admonition} Predict before you run
:class: tip
In our sentence `"patient with severe penicillin allergy must avoid amoxicillin"`, which **three** earlier words will get the tallest attention bars when the model looks back from `"amoxicillin"`?
```

```{code-cell} python
:label: ch44-attention-bars
import numpy as np

# Each word gets a 3-number checklist: [antibiotic, safety_warning, person]
sentence = [
    ("patient",     [0.0, 0.1, 1.0]),
    ("with",        [0.0, 0.0, 0.0]),
    ("severe",      [0.0, 0.6, 0.0]),
    ("penicillin",  [1.0, 0.6, 0.0]),
    ("allergy",     [0.5, 1.0, 0.1]),
    ("must",        [0.0, 0.2, 0.0]),
    ("avoid",       [0.4, 0.9, 0.0]),
]
amox_vec = [1.0, 0.5, 0.0]

# Step 1: Multiply matching checklist items and add (ordinary arithmetic!)
scores = []
for word, vec in sentence:
    match = vec[0]*amox_vec[0] + vec[1]*amox_vec[1] + vec[2]*amox_vec[2]
    scores.append(match)

# Step 2: Turn raw match scores into attention percentages adding up to 100%
weights = np.exp(np.array(scores) * 2.0)     # sharpen so bigger matches stand out
attn_pct = weights / weights.sum() * 100

words = [w for w, _ in sentence]
for w, pct in zip(words, attn_pct):
    print(f"  {w:<12}: {pct:5.1f}% attention")

plt.figure(figsize=(7.2, 3.8))
colors = ["#D55E00" if p > 15 else "#0072B2" for p in attn_pct]
plt.bar(words, attn_pct, color=colors)
plt.ylabel("Attention weight (%)")
plt.title('Looking back from "amoxicillin": attention jumps straight to the safety clues!')
plt.grid(axis="y", alpha=0.25)
plt.show()
```

**Look at those three vermilion bars!** Even though `"penicillin"` and `"allergy"` sat far earlier in the sentence, attention skipped right over filler words (`"with"`, `"must"`) and shone a spotlight straight onto **`penicillin`**, **`allergy`**, and **`avoid`**!

## From our 3-number toy to real clinical AI

Modern healthcare language models — like **AMIE** (a diagnostic conversation AI tested in *Nature* against primary-care physicians [@tu2025amie]) and **MedGemma** (an open medical AI suite) — stack dozens of these exact attention layers trained via gradient descent across enormous text libraries. Yet every single layer is powered by the exact same two ideas: **embeddings** (meaning maps) and **attention** (weighted spotlights across context).

```{admonition} Going deeper — Queries, Keys, and Values in real Transformers
:class: dropdown
In a real Transformer, each word doesn't just use one fixed vector — it builds three specialized projections: a **Query** (*"what clues am I looking for?"*), a **Key** (*"what clues do I advertise?"*), and a **Value** (*"what message do I pass along if selected?"*). The attention equation you hear researchers mention — $\text{softmax}(Q K^T / \sqrt{d})\,V$ — is literally our exact 2-step multiply-add-and-normalize loop run in parallel across every word in a document!
```

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.4.1 (Modify) — Add a new word to the 2D meaning map
:class: hint
Where would you place **`"acetaminophen"`** (a pain & fever reliever like aspirin/ibuprofen) and **`"rash"`** (a symptom) on `word_map` in [](#ch44-word-map)? Add both to `word_map` and re-plot!
```

```{admonition} Solution
:class: dropdown
```python
word_map["acetaminophen"] = (8.8, 1.8)   # near aspirin & ibuprofen!
word_map["rash"]          = (1.0, 5.0)   # near fever & cough (symptoms)!
```
Notice how intuitive embedding space is: you immediately placed `"acetaminophen"` beside the other pain relievers and `"rash"` beside the other symptoms!
```

```{admonition} Exercise 4.4.2 (Make) — Measure distance between two words on the map
:class: hint
Using the Pythagorean distance formula from Chapter 2.1, write 3 lines of Python to compute the distance on `word_map` between `"penicillin"` and `"amoxicillin"` versus `"penicillin"` and `"cough"`.
```

```{admonition} Solution
:class: dropdown
```python
def dist(w1, w2):
    x1, y1 = word_map[w1]
    x2, y2 = word_map[w2]
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

print(f"penicillin <-> amoxicillin: {dist('penicillin', 'amoxicillin'):.2f}")
print(f"penicillin <-> cough      : {dist('penicillin', 'cough'):.2f}")
```
`penicillin` and `amoxicillin` sit just `0.54` apart, while `penicillin` and `cough` sit `8.26` apart!
```

```{admonition} Exercise 4.4.3 (Critical Reasoning) — Why does attention help with medical pronouns?
:class: hint
Read this hospital note: *"The patient's **mother** had diabetes, and **she** also takes blood-pressure pills; however, the **patient** has normal blood sugar."* Explain in two sentences how **attention** helps an AI tell *who* `"she"` refers to so it doesn't put the mother's diagnosis onto the patient!
```

```{admonition} Solution
:class: dropdown
Attention lets the word `"she"` look back across the whole sentence and link strongly to `"mother"` (right before it) while keeping `"the patient"` linked to `"normal blood sugar"`. A simple 1-word bigram model could never track who is who across a 20-word sentence!
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Embeddings** turn words into coordinates on a meaning map so related concepts sit close together.
- **Attention** measures how well earlier words match the current word and focuses the model's spotlight on the most relevant clues across long sentences.
- Modern medical LLMs like **AMIE** [@tu2025amie] scale these exact ideas up to thousands of dimensions and dozens of layers.
```

*Next up:* Even with brilliant embeddings and attention, next-word predictors have one giant weakness: they can invent fake medical facts with total confidence! Let's see why **hallucinations** happen and build an **open-book AI (RAG)** fix in [](4_5-hallucination-and-grounding.md)!
