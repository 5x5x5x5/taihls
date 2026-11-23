---
title: 'Chapter 4.5 — Fluent Is Not True: Hallucination and Open-Book AI (RAG)'
short_title: '4.5 · Hallucination & RAG'
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4.5 — Fluent Is Not True: Hallucination and Open-Book AI (RAG)

*Week 12 · ~40 minutes · Prerequisites: [](4_3-predicting-the-next-word.md), [](4_4-embeddings-and-attention.md)*

```{admonition} Learning objectives
:class: note
By the end of this chapter you will be able to:
- explain why AI **hallucinations** happen naturally from next-word prediction (not just a random glitch);
- demonstrate in code that a language model's **confidence does not equal truth**;
- build a 15-line **open-book lookup (Mini-RAG)** over verified clinic safety cards — including an explicit **"I don't know"** safety gate;
- apply four practical rules for using AI chatbots safely on health and science questions.
```

## Start with a case: the study that never existed

A high-school student researching a biology paper asks a chatbot for a real medical study proving a vitamin prevents colds. Two seconds later, the screen displays a crisp, impressive citation: *"Dr. Elena Martinez et al., Journal of Clinical Medicine, 2023 — a trial of 1,400 teens."*

It sounds 100% professional. There is just one problem: **neither Dr. Martinez nor the study exists.** The AI invented it smoothly without a hint of hesitation!

```{admonition} Definition — hallucination
:class: important
A {term}`Hallucination` is when an AI outputs smooth, confident-sounding facts, doses, or citations that are completely false or made up. The model is not "lying" — it has no built-in truth table; it is simply doing its job of predicting plausible-sounding next words.
```

## Watch a tiny model mix up true facts with EQUAL confidence!

Train our tiny word predictor on five **100% true** statements, then compare the probability it assigns to a **true** statement vs. a **mixed-up claim** that was *never* in the training text!

```{code-cell} python
:label: ch45-hallucination
from collections import defaultdict, Counter

true_notes = """
aspirin relieves fever
aspirin thins blood
ibuprofen relieves joint pain
warfarin thins blood
acetaminophen relieves fever
""".split()

counts = defaultdict(Counter)
for cur, nxt in zip(true_notes, true_notes[1:]):
    counts[cur][nxt] += 1

def phrase_chance(words):
    chance = 1.0
    for cur, nxt in zip(words, words[1:]):
        total = sum(counts[cur].values())
        chance *= counts[cur][nxt] / total if total else 0.0
    return chance

print(f"Chance of 'aspirin relieves fever'            [TRUE ] = {phrase_chance(['aspirin', 'relieves', 'fever']):.3f}")
print(f"Chance of 'acetaminophen relieves joint pain' [MIXUP] = {phrase_chance(['acetaminophen', 'relieves', 'joint', 'pain']):.3f}")
```

**Exact same probability (`0.333`)!** Both follow smooth, common word transitions — so the model has zero clue that the second statement was stitched together from two different drugs. **Fluency is not truth, and confidence is not accuracy.**

## The engineering fix: give the AI an open-book exam (RAG)!

Asking an AI to remember exact medical rules purely from blurry compressed weights is like taking a closed-book pharmacology exam. The fix is {term}`Retrieval-Augmented Generation (RAG)` — **an open-book exam!**

```{mermaid}
flowchart LR
    Q["Student / Clinic Question"] --> Find["1. Search Verified<br/>Health Handbook"]
    Find --> Check{"Found a strong<br/>matching card?"}
    Check -- "No" --> Refuse["Say 'Not in my handbook —<br/>please check a doctor!'"]
    Check -- "Yes" --> Answer["2. Answer ONLY using<br/>that card + cite its ID"]
```

Let's build a super-clean **15-line Open-Book Lookup (Mini-RAG)** using ordinary Python word sets (`set(words_a) & set(words_b)` — counting shared meaningful words!) over six plain-English health safety cards!

```{admonition} Predict before you run
:class: tip
We will ask our Mini-RAG three questions:
1. A question about **penicillin allergies and amoxicillin** (covered in Card #1),
2. A question about **treating low blood sugar** (covered in Card #4), and
3. A question about a **rare new drug NOT in our handbook** (*"zolbetuximab"*).

What should our safety cutoff (`min_matches = 2`) do on Question #3?
```

```{code-cell} python
:label: ch45-mini-rag
handbook = [
    ("CARD-01", "Penicillin allergy: People with a serious penicillin allergy should avoid amoxicillin because both belong to the same antibiotic family. Ask a doctor for a safe alternative."),
    ("CARD-02", "Children with viral fever: Do not give aspirin to children or teenagers who have a viral fever (like flu) because of a rare serious complication. Use acetaminophen or ibuprofen instead."),
    ("CARD-03", "Severe allergic reaction: If someone has trouble breathing or throat swelling from a peanut or bee sting allergy, use an epinephrine auto-injector (EpiPen) right away and call emergency services."),
    ("CARD-04", "Low blood sugar: If a person with diabetes feels shaky and blood sugar drops below 70, give 15 grams of fast-acting sugar (like juice or glucose tablets) and recheck after 15 minutes."),
    ("CARD-05", "Heat exhaustion: Move the person to a cool shaded spot, loosen tight clothing, apply cool wet towels, and offer small sips of water."),
    ("CARD-06", "Blood thinners and pain pills: People taking blood thinners like warfarin should avoid ibuprofen because combining them raises stomach bleeding risk."),
]

IGNORE = {"a", "an", "the", "is", "are", "for", "of", "to", "in", "on", "with", "and", "or", "what", "how", "should", "can", "someone", "person", "do"}

def keywords(text):
    clean = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return set(clean.split()) - IGNORE

def open_book_lookup(question, min_matches=2):
    q_words = keywords(question)
    best_card, best_hits = None, 0

    for card_id, text in handbook:
        hits = len(q_words & keywords(text))     # count shared meaningful words!
        if hits > best_hits:
            best_hits, best_card = hits, (card_id, text)

    print(f"QUESTION: {question}")
    if best_hits < min_matches:
        print(f"  -> [NOT IN HANDBOOK | only {best_hits} matching word(s)]")
        print("  -> SAFE ANSWER: I don't have a verified guideline for this — please consult a doctor or pharmacist.\n")
    else:
        print(f"  -> [FOUND {best_card[0]} | {best_hits} matching keywords]")
        print(f"  -> GROUNDED FACT: {best_card[1]}\n")

open_book_lookup("Can someone with a penicillin allergy take amoxicillin?")
open_book_lookup("How should you treat low blood sugar below 70?")
open_book_lookup("What is the safe dosage of zolbetuximab for stomach tumors?")
```

Look how clean and safe that is! On Questions 1 and 2, Mini-RAG pulls the exact verified card and prints its citation ID (`CARD-01`, `CARD-04`). On Question 3, because the handbook has **zero** matching cards, the system **refuses to guess** instead of making up a dangerous dose!

## Four habits for using AI safely on health or science topics

1. **Give it the text:** Paste the verified article or guideline and say *"Answer using ONLY the text below."*
2. **Invite "I don't know":** Tell the AI explicitly that saying *"this isn't covered"* is better than guessing.
3. **Check every citation and number:** Never trust a generated study title or medicine dose without checking the original source [@rajpurkar2022ai].
4. **Never paste private medical info:** Public chatbots are **not** private medical vaults (more in Unit 5!).

## Exercises (Modify → Make → Reason)

```{admonition} Exercise 4.5.1 (Modify) — Spot a weakness of simple word matching!
:class: hint
Call `open_book_lookup("What medicine helps a teenager with a viral flu temperature?")`. Notice it uses `"temperature"` instead of `"fever"`. Does our simple word-overlap matcher still find `CARD-02`, and how would **embeddings** (Chapter 4.4) make lookup smarter?
```

```{admonition} Solution
:class: dropdown
```python
open_book_lookup("What medicine helps a teenager with a viral flu temperature?")
```
It matches `CARD-02` only weakly (shared `"viral"`, `"flu"`), and would miss completely if we asked *"What helps a kid's high temperature?"* because `"kid"` ≠ `"children"` and `"temperature"` ≠ `"fever"`! Real clinical RAG systems search using **embedding meaning maps** (Chapter 4.4) so synonyms match automatically!
```

```{admonition} Exercise 4.5.2 (Make) — Add a new card to the handbook
:class: hint
Add `("CARD-07", "Sprained ankle first aid: Rest the ankle, apply an ice pack wrapped in cloth for 15 minutes, gently wrap for support, and elevate the foot.")` to `handbook`, and test `open_book_lookup("How should I care for a sprained ankle with ice?")`.
```

```{admonition} Solution
:class: dropdown
```python
handbook_v2 = handbook + [("CARD-07", "Sprained ankle first aid: Rest the ankle, apply an ice pack wrapped in cloth for 15 minutes, gently wrap for support, and elevate the foot.")]
q_words = keywords("How should I care for a sprained ankle with ice?")
for cid, txt in handbook_v2:
    if len(q_words & keywords(txt)) >= 2:
        print(cid, "->", txt)
```
We just updated our AI's medical knowledge in 2 seconds **without retraining a single weight!**
```

```{admonition} Exercise 4.5.3 (Critical Reasoning) — Can an open-book AI still make mistakes?
:class: hint
Even when a hospital chatbot uses RAG connected to verified doctor guidelines, name **one way** it could still give a harmful answer.
```

```{admonition} Solution
:class: dropdown
Two common risks: (1) the retriever pulls the *wrong* card (for example, an adult guideline for a 6-year-old child), or (2) the chatbot misreads a word like *"avoid"* when rewriting the card! That is why a qualified human stays in charge of medical decisions.
```

## Key takeaways & look ahead

```{admonition} Key takeaways
:class: note
- **Hallucinations** happen because language models predict *likely* words, not verified facts — **fluency ≠ truth**.
- **RAG (open-book AI)** searches verified guidelines first and refuses to guess when evidence is missing.
- Always supply grounded context, invite *"I don't know,"* and verify citations independently.
```

*Next up:* What happens when scientists point these exact Transformer and AI-generation ideas not at human English, but at **proteins, whole genomes, and living cells**? Welcome to the frontier of biological AI in [](4_6-foundation-models-of-life.md)!
