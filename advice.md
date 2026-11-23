---
title: 'How to Learn from This Book'
short_title: 'How to learn'
---

# How to Learn from This Book

Reading a computational textbook passively — nodding along at the text and glancing at finished charts — feels smooth, but leaves almost nothing behind a week later. Real understanding happens in the moment between *expecting* one outcome and *seeing* another.

Every chapter in **TAIHLS** is designed around an evidence-based learning cycle from computing education research called **PRIMM**: **P**redict, **R**un, **I**nvestigate, **M**odify, **M**ake.

```{mermaid}
flowchart LR
    P["1. Predict<br/>Guess the output<br/>before running"] --> R["2. Run<br/>Execute the cell<br/>and observe"]
    R --> I["3. Investigate<br/>Unpack why the<br/>result happened"]
    I --> M["4. Modify<br/>Tweak one knob<br/>and compare"]
    M --> K["5. Make<br/>Apply the pattern<br/>to a new problem"]
    K -. "Next concept" .-> P
```

## The five habits that make ideas stick

```{admonition} Habit 1 — Never skip the "Predict before you run" box
:class: tip
Before every major code lab, you will see a **Predict before you run** prompt. Pause for 15 seconds and commit to an answer in your head (or on scratch paper). Even when your guess is wrong — *especially* when it is wrong — your brain flags the surprise and rewires your mental model far more deeply than passive reading ever could.
```

- **Habit 2 — Read code like a lab protocol, not a novel.** You don't need to memorize Python syntax. Focus on *inputs*, *transformations*, and *outputs*. Ask: *Which line loads the patients? Which line trains the model? Which line grades it?*
- **Habit 3 — Break the code on purpose.** Once a cell runs cleanly, change one parameter — set a learning rate 10× higher, shrink a subgroup to 2% of the training set, move a decision threshold from `0.50` to `0.15` — and watch what breaks.
- **Habit 4 — Try the exercises before opening the dropdown.** Every chapter ends with tiered exercises (**Modify**, **Make**, and **Critical Reasoning**). Struggling productively for three minutes before revealing a solution is where durable memory is built.
- **Habit 5 — Treat confusion as data, not failure.** Every expert in bioinformatics and medical AI hits moments where a plot makes no sense. Write down *precisely* what contradicts your expectation, check the [Glossary](back/glossary.md) for any fuzzy terms, and trace one single example by hand.

## A note on using AI assistants while learning

You probably have access to AI chatbots and coding assistants as you read this book. Used well, they are wonderful tutors; used poorly, they rob you of the exact mental workout you came here for.

::::{tab-set}

:::{tab-item} Good ways to use an AI tutor
- Ask *"Explain line 7 of this NumPy snippet using a 3-patient toy example."*
- Ask *"Here is my prediction for Exercise 2.1 and why I think so — point out any flaw in my reasoning without giving away the answer."*
- Ask *"What biological assumptions does the SIR model leave out?"*
:::

:::{tab-item} Traps that short-circuit learning
- Pasting an exercise prompt straight into a chatbot and copying the code without tracing it.
- Skipping the **Predict** phase because the AI already told you what the graph will look like.
- Trusting medical or biological facts from an LLM without checking the primary source (see [](content/4_3-hallucination-grounding-rag.md)!).
:::

::::

## Reading the visual conventions

Throughout the book, you will notice consistent cues to keep cognitive load low:

- **Blue definition boxes** introduce formal terms (all compiled in the [Glossary](back/glossary.md)).
- **Side margin notes** offer bite-sized intuition, history, or cross-links without interrupting the main story.
- **Colorblind-safe figures** use a unified palette (ocean blue `#0072B2`, vermilion `#D55E00`, teal `#009E73`, amber `#E69F00`) so every comparison stays readable for every eye.
- **Open in Colab badges** mark chapters that also offer a free GPU companion notebook for deeper experiments.
