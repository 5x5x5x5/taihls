---
title: 'Chapter D3 — Machines that read and write'
short_title: 'D3 · Language models'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter D3 — Machines that read and write

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/d3-run-an-llm.ipynb)

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain what a **language model** does in one sentence;
- describe **next-token prediction** as a guessing game over a **probability distribution**;
- build a working **bigram (Markov) model** from a tiny corpus and sample text from it;
- see how the *same idea* scales up to the large language models behind modern chatbots.
```

## Start with a story

You have used a language model already today. When your phone offers "you" right
after you type "thank", or "doctor" after "see the", it is doing one thing: guessing
the next word from the words so far. Scale that same trick up enormously — train it on
much of the internet — and you get the chatbots that can draft an email, summarize a
study, or answer a patient's question.

In this chapter we strip the trick down to its bones and build a tiny version we can
read line by line. The same machinery, just bigger, is what powers ChatGPT and its
cousins. Let's name the pieces.

```{admonition} Definition — token
:class: important
A **token** is one unit of text that a model reads or writes — for us, a single
**word**. (Real models often use word *fragments* as tokens, but whole words are
easier to picture and work exactly the same way.)
```

```{admonition} Definition — language model
:class: important
A **language model** is a system that, given some text, predicts what text is likely
to come next. That single ability — *predict the next token* — is enough to read,
continue, and even appear to "converse."
```

## The whole idea: next-token prediction

```{admonition} Definition — next-token prediction
:class: important
**Next-token prediction** is the task of guessing the next token from the tokens
before it. A model trained only to play this guessing game, billions of times, learns
a startling amount about grammar, facts, and style.
```

How does a machine "guess"? Not with a single answer, but with a spread of options,
each carrying a chance.

```{admonition} Definition — probability distribution
:class: important
A **probability distribution** is a list of possible outcomes together with the
chance of each, where all the chances add up to 1. A language model's output is a
probability distribution over *every possible next token*.
```

## Building a bigram model from a tiny corpus

The simplest possible language model looks back just **one** token. It asks: "given
the current word, which words have followed it before, and how often?" That count,
turned into fractions, is its probability distribution. A model that uses pairs of
consecutive tokens this way is called a **bigram** (or first-order **Markov**) model.

Here is a small corpus — a handful of sentences a clinic chatbot might have seen.

```{code-cell} python
:label: corpus
corpus = """
the patient has a fever
the patient needs rest
the doctor checks the patient
the nurse helps the patient
the patient feels better
""".split()

print(f"{len(corpus)} tokens, {len(set(corpus))} of them unique.")
print("First few tokens:", corpus[:6])
```

Now we count, for every word, which words come next. `collections.defaultdict` plus
`Counter` makes the bookkeeping painless.

```{code-cell} python
:label: counts
from collections import defaultdict, Counter

following = defaultdict(Counter)
for current, nxt in zip(corpus, corpus[1:]):
    following[current][nxt] += 1

print('After "the", we have seen:')
for word, n in following["the"].most_common():
    print(f"  {word}: {n}")
```

So in our corpus the word `the` was followed by `patient` most often, then by
`doctor` and `nurse`. To turn those raw counts into a **probability distribution**, we
divide each count by the total.

```{code-cell} python
:label: probs
import numpy as np

def next_distribution(word):
    counts = following[word]
    words = list(counts)
    totals = np.array([counts[w] for w in words], dtype=float)
    probs = totals / totals.sum()
    return words, probs

words, probs = next_distribution("the")
for w, p in zip(words, probs):
    print(f"  P(next = {w!r:>10} | 'the') = {p:.2f}")
print(f"\nThese probabilities sum to {probs.sum():.1f}.")
```

The probabilities add to 1, exactly as a distribution must. The model now *believes*
that after `the`, the next word is `patient` about 71% of the time, with `doctor` and
`nurse` sharing the rest.

## Sampling: letting the model write

A language model writes by repeatedly **sampling** — drawing a next token at random,
but weighted by the probabilities. Likely words come up often; rare ones occasionally.
We seed the randomness so the result is reproducible.

```{code-cell} python
:label: sample
rng = np.random.default_rng(0)

def generate(start, length=6):
    out = [start]
    word = start
    for _ in range(length - 1):
        if word not in following:        # dead end: nothing ever followed it
            break
        words, probs = next_distribution(word)
        word = rng.choice(words, p=probs)
        out.append(word)
    return " ".join(out)

for _ in range(4):
    print(generate("the"))
```

Out of nothing but counting and weighted coin-flips, our model produces sentences
that *sound like the corpus* — "the patient feels better," "the doctor checks the
patient." It has no idea what a patient is. It has only learned which words tend to
follow which, yet that alone yields fluent-looking text.

## From a toy to a giant

```{admonition} Run a real language model (companion notebook)
:class: tip
Our bigram model looks back one word and counts. A real **large language model** looks
back over *thousands* of tokens and learns its probabilities with a neural network
trained on enormous text. The companion Colab notebook lets you load a genuine small
LLM (`distilgpt2`) and watch it predict next-token probabilities and generate text on
a GPU:
<https://colab.research.google.com/github/5x5x5x5/taihls/blob/course-curriculum/notebooks/d3-run-an-llm.ipynb>
```

The leap from our toy to ChatGPT is one of *scale and memory*, not of basic idea. Both
answer the same question — "what token comes next?" — by producing a probability
distribution and sampling from it. Keeping that in mind is the key to understanding
both their power and, as we'll see in [](4_d4-hallucination.md), their failures.

## Exercises

```{admonition} Exercise D3.1 — A different distribution
:class: hint
Using `next_distribution`, print the probability distribution for the word
`"patient"`. Which word is most likely to follow it, and how does the *spread* of this
distribution compare to the one for `"the"`?
```

```{admonition} Solution
:class: dropdown
```python
words, probs = next_distribution("patient")
for w, p in zip(words, probs):
    print(f"{w}: {p:.2f}")
```

`patient` is followed by four different words — `has` (0.20), `needs` (0.20), `the`
(0.40), and `feels` (0.20). The most likely is `the`, at 0.40. Compared with `the`'s
distribution (where `patient` alone takes 0.71), this one is **more spread out**: no
option is anywhere near as dominant, so the model is *less certain* about what comes
after `patient`.
```

```{admonition} Exercise D3.2 — Longer memory
:class: hint
Our model looks back only one word (a bigram). In one or two sentences, explain why
looking back *two* words (a **trigram**) might write more sensible text — and what new
problem appears as you make the memory longer and longer.
```

```{admonition} Solution
:class: dropdown
A trigram conditions on the last *two* words, so it captures more context (e.g. "the
patient feels" strongly predicts "better") and tends to produce more coherent text.
The new problem is **data sparsity**: there are far more possible two-word histories
than one-word ones, so most of them are never seen in a small corpus, leaving the
model with nothing to go on. Real LLMs solve this by learning *patterns* across
histories with a neural network instead of memorizing exact counts.
```

## Key takeaways

```{admonition} Key takeaways
:class: note
- A **language model** does one thing: **next-token prediction** — guess the next
  token from the ones before.
- Its output is a **probability distribution** over all possible next tokens, and it
  writes by **sampling** from that distribution.
- A **bigram (Markov) model** is just counting which token follows which, normalized
  to probabilities — buildable in a dozen lines.
- Real LLMs differ in **scale and memory**, not in the core idea — which is why the
  same lens explains both their fluency and their tendency to make things up.
```
