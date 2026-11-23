---
title: 'Chapter 4.3 — Fluent Is Not True: Hallucination, Retrieval-Augmented Generation, and Clinical Prompting'
short_title: '4.3 · Hallucination & RAG'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.3 — Fluent Is Not True: Hallucination, Retrieval-Augmented Generation, and Clinical Prompting

*Estimated time: ~45 minutes · Prerequisites: [](4_2-language-models-and-attention.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain *why* AI **hallucinations** arise naturally from next-token prediction rather than being a simple software bug;
- demonstrate in code that a language model's **confidence (high token probability) does not guarantee truth**;
- build a working **mini-RAG (Retrieval-Augmented Generation)** clinical search pipeline in 25 lines of Python — including an explicit **"not in guidelines" abstention gate**;
- apply evidence-based **clinical prompting hygiene** and explain how **tool use** makes medical AI safer.
```

## Start with a case: the citation that never existed

A pre-med student asks a general-purpose chatbot for peer-reviewed studies showing whether an herbal supplement interferes with blood thinners. Two seconds later, the screen fills with a crisp paragraph citing *"Martinez et al., New England Journal of Medicine, 2023, Vol. 388, pp. 112–119 — a randomized trial of 1,400 patients."*

Every detail looks textbook-perfect — the journal name, the volume number for 2023, the clinical phrasing. There is just one catch: **the paper does not exist.** Neither Martinez nor the 1,400-patient trial ever happened.

```{admonition} Definition — hallucination
:class: important
A {term}`Hallucination` occurs when a generative AI outputs fluent, authoritative-sounding claims, numbers, or citations that are factually false or completely fabricated. The model is not "lying" — it has no internal concept of truth; it is simply sampling high-probability token sequences.
```

## Lab 1: Watching a tiny model hallucinate — with equal confidence!

Let's strip the mystery away completely. Train our tiny next-token predictor from Chapter 4.2 on five 100%-true pharmacology facts, then ask the model to score a **true** claim versus a **dangerous fabricated mix-up**!

```{admonition} Predict before you run
:class: tip
Our tiny training corpus says `"warfarin prevents blood clots"` and `"ibuprofen reduces joint pain"` — it **never** says `"warfarin reduces joint pain"`! Yet both `"warfarin"` and `"ibuprofen"` appear in overlapping clinical patterns. Will the model assign probability `0.00` to the false drug claim, or rate it just as plausible as a true fact?
```

```{code-cell} python
:label: ch43-hallucination-toy
from collections import defaultdict, Counter
import numpy as np

facts = """
aspirin relieves fever
aspirin thins blood
ibuprofen relieves joint pain
warfarin thins blood
acetaminophen relieves fever
""".split()

trans = defaultdict(Counter)
for a, b in zip(facts, facts[1:]):
    trans[a][b] += 1

def p_next(cur, nxt):
    total = sum(trans[cur].values())
    return trans[cur][nxt] / total if total > 0 else 0.0

def seq_prob(words):
    p = 1.0
    for a, b in zip(words, words[1:]):
        p *= p_next(a, b)
    return p

true_stmt  = ["aspirin", "relieves", "fever"]
false_stmt = ["aspirin", "relieves", "joint", "pain"]
false_mix  = ["acetaminophen", "relieves", "joint", "pain"]

print(f"P('aspirin relieves fever')             [TRUE ] = {seq_prob(true_stmt):.3f}")
print(f"P('aspirin relieves joint pain')        [FALSE] = {seq_prob(false_stmt):.3f}")
print(f"P('acetaminophen relieves joint pain')  [MIXUP] = {seq_prob(false_mix):.3f}")
```

Look closely! Even though `"acetaminophen relieves joint pain"` was **never in the training text**, the model assigns it the **exact same probability (`0.333`)** as the completely true fact `"aspirin relieves fever"` (because `"acetaminophen"` was always followed by `"relieves"` in this tiny corpus)!

**Fluency is not truth, and high token probability is not correctness.** Even 100-billion-parameter LLMs suffer from a scaled-up version of this exact mechanism whenever asked to recall rare dosages, obscure studies, or patient-specific charts purely from compressed weights.

## The engineering cure: Retrieval-Augmented Generation (RAG)

If asking a model to recite medical guidelines from blurry memory invites hallucinations, what is the fix? **Give the model an open-book exam!**

That architecture is called {term}`Retrieval-Augmented Generation (RAG)` — and today it is the backbone of almost every serious clinical decision-support assistant.

```{mermaid}
flowchart LR
    Q["Clinician Question<br/>'Can I give amoxicillin<br/>with penicillin allergy?'"] --> Ret["1. Retriever<br/>Search verified<br/>clinical guidelines"]
    DB[("Verified Guideline<br/>Knowledge Base")] --> Ret
    Ret --> Gate{"Top match score<br/>≥ threshold?"}
    Gate -- "No (not found)" --> Abstain["Refuse to guess:<br/>'Not in verified<br/>guidelines'"]
    Gate -- "Yes (top cards)" --> Gen["2. Grounded Generator<br/>Answer ONLY from<br/>retrieved cards + cite ID"]
```

## Lab 2: Build a working clinical Mini-RAG in 25 lines of Python!

Let's build a genuine, runnable **Mini-RAG** system using `scikit-learn`'s `TfidfVectorizer` (which converts each clinical card and question into a word-importance vector) and **cosine similarity** — complete with an explicit **abstention gate** when the knowledge base lacks evidence!

```{admonition} Predict before you run
:class: tip
We will feed three clinical questions into our Mini-RAG:
1. A question **covered** by our guideline cards (*penicillin allergy & amoxicillin*),
2. A question about **warfarin + NSAID bleeding risk** (also covered), and
3. A question about an **experimental cancer drug NOT in our guidelines** (*"What is the recommended dose of zolbetuximab for advanced gastric cancer?"*).

Watch how our similarity threshold (`min_score = 0.25`) catches Question #3 before a generator can invent a dangerous dose!
```

```{code-cell} python
:label: ch43-mini-rag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

guidelines = [
    ("GUIDE-01", "Penicillin allergy: Avoid amoxicillin and ampicillin in patients with severe IgE-mediated penicillin allergy due to cross-reactivity. Consider azithromycin or doxycycline per local antibiogram."),
    ("GUIDE-02", "Warfarin interactions: Combining warfarin with NSAIDs such as ibuprofen or naproxen significantly increases gastrointestinal bleeding risk. Prefer acetaminophen for mild pain and monitor INR."),
    ("GUIDE-03", "Pediatric fever: Avoid aspirin in children and teenagers with viral illness due to the risk of Reye syndrome. Weight-based acetaminophen or ibuprofen is standard."),
    ("GUIDE-04", "Metformin safety: Hold metformin temporarily during acute kidney injury or severe dehydration due to lactic acidosis risk. Check eGFR before restarting."),
    ("GUIDE-05", "Anaphylaxis emergency: Intramuscular epinephrine into the mid-outer thigh is first-line treatment for anaphylaxis. Do not delay epinephrine for antihistamines."),
    ("GUIDE-06", "Hypertension lifestyle: Sodium reduction, regular aerobic exercise, and DASH dietary patterns lower systolic blood pressure alongside first-line pharmacotherapy."),
]

doc_ids = [g[0] for g in guidelines]
doc_texts = [g[1] for g in guidelines]

vectorizer = TfidfVectorizer(stop_words="english")
doc_matrix = vectorizer.fit_transform(doc_texts)

def mini_rag(query, min_score=0.25):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_matrix)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    print(f"QUESTION: {query}")
    if best_score < min_score:
        print(f"  -> [ABSTAINED | max similarity = {best_score:.2f} < {min_score}]")
        print("  -> ANSWER: No verified guideline found. Do NOT guess — consult primary literature or pharmacy.\n")
    else:
        print(f"  -> [RETRIEVED {doc_ids[best_idx]} | similarity = {best_score:.2f}]")
        print(f"  -> GROUNDED EVIDENCE: {doc_texts[best_idx]}\n")

mini_rag("Can a patient with severe penicillin allergy take amoxicillin?")
mini_rag("Is ibuprofen safe for someone taking warfarin blood thinners?")
mini_rag("What is the recommended dose of zolbetuximab for advanced gastric cancer?")
```

Let's visualize the **retrieval similarity matrix** across all three queries and all six guidelines so you can *see* how clearly the abstention cutoff separates grounded medical knowledge from unverified territory!

```{code-cell} python
:label: ch43-rag-fig
import matplotlib.pyplot as plt

queries = [
    "Q1: Penicillin allergy\n& amoxicillin?",
    "Q2: Warfarin &\nibuprofen safety?",
    "Q3: Dose of zolbetuximab\nfor gastric cancer?",
]
q_texts = [
    "Can a patient with severe penicillin allergy take amoxicillin?",
    "Is ibuprofen safe for someone taking warfarin blood thinners?",
    "What is the recommended dose of zolbetuximab for advanced gastric cancer?",
]
sim_mat = cosine_similarity(vectorizer.transform(q_texts), doc_matrix)

fig, ax = plt.subplots(figsize=(8.2, 3.6))
im = ax.imshow(sim_mat, cmap="YlGnBu", vmin=0, vmax=0.65, aspect="auto")
ax.set_xticks(range(len(doc_ids)), doc_ids)
ax.set_yticks(range(len(queries)), queries)
for i in range(sim_mat.shape[0]):
    for j in range(sim_mat.shape[1]):
        val = sim_mat[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if val > 0.35 else "#111111", fontweight="bold")
ax.set_title("Mini-RAG retrieval similarities: Q3 scores 0.00 across every card → safe abstention!")
fig.colorbar(im, ax=ax, label="Cosine similarity")
fig.tight_layout()
plt.show()
```

:::{margin}
**From RAG to Clinical Tool-Use (Agents):** Modern clinical AI assistants go one step further than text retrieval — they call **deterministic calculators** (e.g., weight-based pediatric dosing, kidney eGFR equations) and live EHR queries so arithmetic and lab lookup never rely on fuzzy next-token guessing!
:::

## Clinical prompting hygiene: four rules that save lives

Even with RAG, an LLM can still misread a retrieved paragraph or omit a critical contraindication [@rajpurkar2022ai]. Whenever you — or a clinician — interact with a language model on a health or biology question, enforce four non-negotiable habits:

::::{tab-set}

:::{tab-item} 1. Supply Grounded Context
Paste the exact guideline excerpt, lab report, or paper abstract and instruct: *"Answer **only** using the text provided below. Quote the relevant sentence."*
:::

:::{tab-item} 2. Explicitly Permit "I Don't Know"
Default chatbots are trained to be helpful, which makes them guess when uncertain. Add: *"If the provided text does not answer the question, state explicitly that the information is missing."*
:::

:::{tab-item} 3. Verify Every Citation & Dose
Never trust a generated PubMed ID, DOI, or mg/kg drug dose without opening the primary source or calculating it independently.
:::

:::{tab-item} 4. Never Paste Real Patient Names (Privacy!)
Consumer chatbots are **not** HIPAA-compliant! Never paste real names, birthdates, or identifiable patient records into public LLMs (see Chapter 5.2!).
:::

::::

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.3.1 (Modify) — Expose a limitation of word-matching (TF-IDF) RAG
:class: hint
Call `mini_rag("What emergency shot stops a severe allergic reaction?")` using the code from [](#ch43-mini-rag). Does our keyword-based `TfidfVectorizer` retriever find `GUIDE-05` (*"Anaphylaxis emergency: Intramuscular epinephrine..."*), or does it miss because the prompt used **synonyms** (`"allergic reaction"` instead of `"anaphylaxis"`, `"shot"` instead of `"intramuscular"`)? How do neural **embedding** retrievers fix this?
```

```{admonition} Solution
:class: dropdown
```python
mini_rag("What emergency shot stops a severe allergic reaction?")
```

TF-IDF checks exact shared words — so `"allergic reaction"` barely overlaps with `"anaphylaxis"` (only `"emergency"` matches weakly!), giving a low score or picking the wrong card (`GUIDE-01` because of `"allergy"`!). Real clinical RAG systems use **dense neural embeddings** (like the meaning vectors in Chapter 4.2!) where `"anaphylaxis"` and `"severe allergic reaction"` sit right next to each other in vector space, even with zero shared spelling!
```

```{admonition} Exercise 4.3.2 (Make) — Add a new guideline card and test it
:class: hint
Append a seventh card `("GUIDE-07", "Hypoglycemia treatment: Conscious patients with blood glucose below 70 mg/dL should receive 15 grams of fast-acting oral carbohydrates and recheck glucose after 15 minutes.")` to `guidelines`, rebuild `doc_matrix`, and query `"How do you treat low blood glucose below 70?"`. Verify that `GUIDE-07` is retrieved cleanly!
```

```{admonition} Solution
:class: dropdown
```python
guidelines_v2 = guidelines + [
    ("GUIDE-07", "Hypoglycemia treatment: Conscious patients with blood glucose below 70 mg/dL should receive 15 grams of fast-acting oral carbohydrates and recheck glucose after 15 minutes.")
]
vec2 = TfidfVectorizer(stop_words="english")
mat2 = vec2.fit_transform([g[1] for g in guidelines_v2])
sims2 = cosine_similarity(vec2.transform(["How do you treat low blood glucose below 70?"]), mat2)[0]
best = int(np.argmax(sims2))
print(f"Top hit: {guidelines_v2[best][0]} (sim = {sims2[best]:.2f})")
print(guidelines_v2[best][1])
```

`GUIDE-07` is immediately retrieved with high similarity — notice we updated the system's medical knowledge in milliseconds **without retraining a single neural-network weight**! That instant updatability is another huge advantage of RAG when medical guidelines change.
```

```{admonition} Exercise 4.3.3 (Critical Reasoning) — Can RAG still hallucinate?
:class: hint
Suppose a hospital builds a clinical RAG chatbot linked to real, verified medical guidelines. Name **two ways** the chatbot could *still* give a dangerous answer to a doctor!
```

```{admonition} Solution
:class: dropdown
1. **Retrieval failure:** The retriever pulls an outdated guideline, an *adult* dosing card for a *pediatric* patient, or misses a second card listing a drug allergy!
2. **Faithfulness failure (ungrounded generation):** Even when the right guideline card is retrieved into the prompt, the LLM may accidentally invert a negation (*"avoid X"* → *"use X"*) or blend in conflicting patterns from pretraining memory.

RAG drastically slashes hallucination rates, but **never replaces clinician verification** (Chapter 5.3).
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Hallucinations** happen because language models optimize for fluent, probable next tokens — **not** factual verification.
- High token probability reflects linguistic familiarity, never clinical truth.
- **Retrieval-Augmented Generation (RAG)** grounds answers in verified, updatable documents and uses a similarity cutoff to abstain instead of guessing.
- Safe clinical use combines RAG, deterministic tools/calculators, strict prompting hygiene, and human review [@rajpurkar2022ai].
```

*Next up:* What happens when we point the exact Transformer and diffusion architectures behind modern AI not at human text, but at **the language of life itself** — proteins, whole genomes, small-molecule drugs, and single cells? Welcome to the 2026 frontier in [](4_4-foundation-models-biology.md)!
