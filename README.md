# TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences

An interactive online textbook (Jupyter Book 2 / MyST) for advanced high-school
students and first-year undergraduates. Every example runs — in the browser, in
the cloud, or locally.

> **Status:** 21 chapters across 5 parts, complete and copyedited, plus 5
> companion Colab notebooks for heavyweight/GPU work.

## Course contents

**Part A — Seeing patterns in biological data**
- A1. How fast does an infection grow?
- A2. What does the data look like?
- A3. Do two things go together?
- A4. Reading a medical statistic

**Part B — Teaching a computer to decide**
- B1. Is this tumor benign or malignant?
- B2. How do we know a model is any good?
- B3. The cost of being wrong
- B4. Drawing the line

**Part C — Learning from examples at scale**
- C1. Fitting the best line
- C2. What is a neural network?
- C3. Reading medical images
- C4. How models learn

**Part D — Life as sequences, language as data**
- D1. DNA and proteins as text
- D2. Comparing sequences
- D3. Machines that read and write
- D4. When AI makes things up
- D5. Foundation models for biology

**Part E — Using AI responsibly in health**
- E1. When AI gets it wrong for some people
- E2. Privacy, consent, and your data
- E3. Keeping humans in the loop
- E4. Capstone: your turn

**Companion Colab notebooks** (`notebooks/`, heavyweight/GPU — not part of the
book build): a first neural net (C2), a medical-image CNN (C3), running an LLM
(D3), protein foundation models (D5), and a capstone starter (E4).

## Project layout

| Path | What it is |
|------|------------|
| `myst.yml` | Project config (title, authors, license, JupyterLite). **Don't put the TOC here.** |
| `toc.yml` | Table of contents — edit structure here (referenced via `extends:`). |
| `index.md`, `preface.md`, `advice.md` | Front matter. |
| `content/` | The 21 chapters, grouped into Parts A–E (see Course contents above). |
| `notebooks/` | Companion Colab notebooks for heavyweight/GPU exercises. |
| `back/` | Reproducibility, references, acknowledgements, license. |
| `references.bib` | Bibliography. |
| `requirements.txt` | Execution deps for Binder/Colab (pyproject.toml + uv.lock are the source of truth). |
| `.github/workflows/deploy.yml` | Builds and deploys to GitHub Pages. |

## Local development

Requires [`uv`](https://docs.astral.sh/uv/). Node is **not** required.

```bash
uv sync                              # install pinned deps from uv.lock
uv run jupyter book start            # live preview at http://localhost:3000
uv run jupyter book start --execute  # same, but run the code cells
```

Build the static HTML site (what gets deployed):

```bash
uv run jupyter book build --html --execute
# output → _build/html/
```

## Running the code (for readers)

1. **In-browser, no install** — JupyterLite is enabled (`project.jupyter.lite: true`);
   run code cells right in the page.
2. **Binder** — full cloud env from `requirements.txt`.
3. **Locally** — `uv sync` then the commands above.

See `back/reproducibility.md` for details.

## Publishing

1. In repo **Settings → Pages**, set **Source = GitHub Actions**.
2. Push to `main`. The workflow builds (executing the code) and publishes automatically.

## Authoring a new chapter

Copy the **structure** of `content/1_a1-infection-growth.md`: learning objectives →
concept-through-example → small runnable code cell → exercise with a `dropdown`
solution → key takeaways. Add the file to `toc.yml`. Code cells need a
`kernelspec` in the frontmatter (see any chapter).
