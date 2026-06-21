# TAIHLS — Topics in Artificial Intelligence for Health and Life Sciences

An interactive online textbook (Jupyter Book 2 / MyST) for advanced high-school
students and first-year undergraduates. Every example runs — in the browser, in
the cloud, or locally.

> **Status:** scaffold + house style + one fully-worked exemplar chapter
> (`content/1_a1-placeholder.md`). All other content is marked `<!-- PLACEHOLDER -->`
> for the author to fill in.

## Project layout

| Path | What it is |
|------|------------|
| `myst.yml` | Project config (title, authors, license, JupyterLite). **Don't put the TOC here.** |
| `toc.yml` | Table of contents — edit structure here (referenced via `extends:`). |
| `index.md`, `preface.md`, `advice.md` | Front matter. |
| `content/` | Chapters. `1_a1-placeholder.md` is the worked exemplar; the rest are stubs. |
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

1. Replace the `PLACEHOLDER` GitHub URL/owner in `myst.yml` and the launch badges
   in `content/1_a1-placeholder.md`.
2. Push to `main`. In repo **Settings → Pages**, set **Source = GitHub Actions**.
3. The workflow builds (executing the code) and publishes automatically.

## Authoring a new chapter

Copy the **structure** of `content/1_a1-placeholder.md`: learning objectives →
concept-through-example → small runnable code cell → exercise with a `dropdown`
solution → key takeaways. Add the file to `toc.yml`. Code cells need a
`kernelspec` in the frontmatter (see any chapter).
