# TAIHLS — project notes

Full-semester (16-week) interactive textbook for **advanced high-school students** and intro undergraduates, built with **Jupyter Book 2 (MyST engine)** — *not* v1.

## Don't use Jupyter Book v1 patterns
- Config is `myst.yml`; TOC is `toc.yml` (loaded via `extends:`).
- CLI: `uv run jupyter book start --execute`, `uv run jupyter book build --html --execute --strict`.
- Keep `pyproject.toml`, `uv.lock`, and `requirements.txt` in sync (`uv sync --frozen` runs in CI).

## High-School Pedagogical Guardrails
- **One big idea per chapter** (28 short lessons across Units 0–5, paced for ~2 lessons/week + 2-week capstone).
- **Hand-worked example before code:** Walk through 2–4 concrete numbers in plain English before running Python.
- **Zero unexplained jargon:** Define every technical and biological term in plain language on first use and keep it in `back/glossary.md`. No med-school alphabet soup (`eGFR`, `INR`, `IgE-mediated`, etc.).
- **High-school-readable Python:** Short, clean code cells (8–20 lines); avoid hidden linear-algebra tricks (`@`, `[:, :, None]`, `np.triu`, `np.ogrid`, `np.c_`) and heavy plotting boilerplate. Use `plt.show()` (never trailing bare `fig`) so figures never duplicate.
- **High ceiling via optional dropdowns:** Put advanced math formulas or extra technical topics inside collapsible `Going deeper` (`:class: dropdown`) admonitions so curious students can explore without overwhelming the main flow.
- **PRIMM cycle:** Every major code lab starts with a `Predict before you run` box, followed by Run, Investigate, and 3 tiered exercises (*Modify*, *Make*, *Critical Reasoning*).
