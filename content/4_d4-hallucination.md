---
title: 'Chapter D4 — When AI makes things up'
short_title: 'D4 · Hallucination'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter D4 — When AI makes things up

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what an AI **hallucination** is and why it happens;
- show, with a tiny model, that **fluent does not mean correct**;
- describe why hallucinations are especially dangerous in medicine;
- apply basic **prompting** hygiene to reduce (but not eliminate) the risk.
```

## Start with a story

A medical student asks a chatbot for a citation supporting a treatment. Back comes a
perfectly formatted reference: real-sounding authors, a real journal, a plausible
year. There is just one problem — the paper does not exist. The model *invented* it,
and did so with total confidence.

This failure has a name, and understanding *why* it happens is the whole point of this
chapter. The surprising answer is that it is not a bug bolted onto language models —
it falls straight out of how they work, which we built by hand in
[](3_d3-language-models.md).

```{admonition} Definition — hallucination
:class: important
A **hallucination** is when an AI produces text that is fluent and confident-sounding
but **factually wrong or entirely made up**. The model is not lying — it has no
concept of truth. It is doing exactly its job: producing likely-sounding next tokens.
```

## Watch a tiny model make things up

Recall our bigram model from the previous chapter: it just learns which word tends to
follow which, then samples. It has *no model of the world* — only of word patterns. So
it will happily generate sentences that read smoothly yet state things that were never
true. Let's build one on a small medical-style corpus and watch.

```{code-cell} python
:label: d4-corpus
from collections import defaultdict, Counter
import numpy as np

corpus = """
aspirin reduces fever
aspirin thins the blood
ibuprofen reduces inflammation
the drug lowers blood pressure
the drug reduces pain
""".split()

following = defaultdict(Counter)
for current, nxt in zip(corpus, corpus[1:]):
    following[current][nxt] += 1

print(f"{len(corpus)} tokens learned.")
```

Every sentence in the corpus is true. But the model only remembers *local* word
transitions, so when it stitches them together it can cross wires — pairing a real
subject with a real predicate that, together, form a false claim.

```{code-cell} python
:label: generate
rng = np.random.default_rng(2)

def next_distribution(word):
    counts = following[word]
    words = list(counts)
    probs = np.array([counts[w] for w in words], dtype=float)
    return words, probs / probs.sum()

def generate(start, length=5):
    out, word = [start], start
    for _ in range(length - 1):
        if word not in following:
            break
        words, probs = next_distribution(word)
        word = rng.choice(words, p=probs)
        out.append(word)
    return " ".join(out)

for _ in range(5):
    print(generate("aspirin"))
```

Look closely at what came out. Some lines are fine, but watch for a sentence like
**"aspirin reduces inflammation"** — fluent, grammatical, and *not something our
corpus ever said* (in our corpus it is *ibuprofen* that reduces inflammation, not
aspirin). The model recombined true
fragments into a confident new claim that may be false. That is hallucination in
miniature.

```{admonition} Fluency is not truth
:class: warning
Our model never checked a fact in its life. It strings together *probable* words, and
probable-sounding is not the same as *true*. The very smoothness that makes the output
convincing is produced by the same mechanism that lets it go wrong.
```

## Confidence is not correctness

It is tempting to think: surely the model is *less sure* when it is wrong? It is not.
Our model assigns a probability to every continuation it produces — including the
false ones — and that number reflects only *how common the word pattern was*, never
whether the statement is true.

```{code-cell} python
:label: confidence
def sentence_probability(words_list):
    """Probability the model assigns to a specific sequence of words."""
    p = 1.0
    for current, nxt in zip(words_list, words_list[1:]):
        words, probs = next_distribution(current)
        p *= probs[words.index(nxt)] if nxt in words else 0.0
    return p

true_claim  = ["aspirin", "reduces", "fever"]
false_claim = ["aspirin", "reduces", "inflammation"]

print(f"P('aspirin reduces fever')        = {sentence_probability(true_claim):.2f}")
print(f"P('aspirin reduces inflammation') = {sentence_probability(false_claim):.2f}")
```

The model rates the false claim and the true claim with the **same** probability —
both flow naturally from the word `reduces`. Confidence (a high probability) tells you
the text is *typical*, not that it is *correct*. This is the trap, and it is built in.

## Why this is dangerous in medicine

In a chat about your weekend plans, a hallucinated detail is harmless. In healthcare
it can hurt people: a fabricated drug dose, an invented contraindication, a made-up
study used to justify a decision. Because hallucinations arrive in the same fluent,
authoritative voice as correct answers, a busy clinician — or a worried patient — has
no easy way to tell them apart. As [@rajpurkar2022ai] stress, AI tools in medicine
must be deployed with human oversight and rigorous validation precisely because a
confident wrong answer can be more dangerous than no answer at all.

```{admonition} Definition — prompting
:class: important
**Prompting** is how you phrase your request to a language model. Because the model
only continues your text, *what you ask and how you ask it* strongly shapes what you
get back — including how likely it is to invent things.
```

## Prompting hygiene: reduce, don't trust

You cannot make a language model incapable of hallucinating, but a few habits make it
less likely and easier to catch.

```{admonition} Practical prompting hygiene
:class: tip
- **Ask for sources you can check** — then actually verify them; a citation is not
  proof it exists.
- **Give the model the facts** to work from ("Using only the text below, …") instead
  of asking it to recall them from memory.
- **Invite "I don't know"** — tell the model it may say it is unsure rather than
  guess.
- **Never let an AI have the final word** on a medical decision. Treat its output as a
  draft for a qualified human to check — the theme of Part E,
  [](3_e3-humans-in-the-loop.md).
```

You can see this same behavior in a *real* large model using the
[D3 companion notebook](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/d3-run-an-llm.ipynb):
ask `distilgpt2` for a fact and watch it produce something fluent — and often wrong —
with no signal that it is unsure.

## Exercises

```{admonition} Exercise D4.1 — Spot the hallucination
:class: hint
Run the generation cell a few more times (try changing the seed in
`np.random.default_rng(2)`). List one sentence the model produced that is **fluent but
was never in the corpus**, and explain in one line how the model arrived at it.
```

```{admonition} Solution
:class: dropdown
Answers vary with the seed, but a typical example is **"aspirin reduces inflammation"**
or **"aspirin thins the blood pressure"**. The model produced it by following valid
one-step transitions (`aspirin → reduces`, `reduces → inflammation`) that each
appeared in *different* sentences, then chaining them into a claim that, as a whole,
the corpus never made. No step felt unusual to the model, so nothing flagged it as
wrong.
```

```{admonition} Exercise D4.2 — Why hygiene helps
:class: hint
Of the four prompting-hygiene habits above, which *directly* attacks the cause of
hallucination we demonstrated — that the model recalls patterns, not facts? Explain
why.
```

```{admonition} Solution
:class: dropdown
**"Give the model the facts to work from."** Our model hallucinated because it had only
*word-pattern memory* and no source of truth to consult. Supplying the relevant facts
in the prompt changes the task from "recall something plausible" to "use what's in
front of you," which removes the gap where invention happens. (The other habits help
you *catch* hallucinations; this one helps *prevent* them.)
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **hallucination** is fluent, confident text that is factually wrong — a natural
  consequence of next-token prediction, not a fixable glitch.
- **Fluency ≠ truth** and **confidence ≠ correctness**: our toy model rates a false
  claim exactly as highly as a true one.
- In medicine, confidently wrong output is dangerous, which is why human oversight is
  non-negotiable [@rajpurkar2022ai].
- Good **prompting** hygiene — checkable sources, supplying facts, allowing "I don't
  know," and keeping a human in the loop — reduces but never eliminates the risk.
```
