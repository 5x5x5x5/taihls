# TAIHLS — project notes

Interactive textbook built with **Jupyter Book 2 (MyST engine)** — *not* v1.

## Don't use Jupyter Book v1 patterns
- Config is a single `myst.yml` (no `_config.yml`).
- TOC lives in `toc.yml`, pulled in via `extends:` (no `_toc.yml`, no Sphinx).
- CLI is `jupyter book <verb>`: `init`, `start`, `build`, `clean`.
- Content files are `.md` with MyST directives.

## Workflow
- `uv` only:
  - `uv sync`
  - `uv run jupyter book start --execute`
  - `uv run jupyter book build --html --execute --strict`
- Static HTML output → `_build/html/` (client-rendered MyST SPA).
- CI enforces `--strict` and `uv sync --frozen` — keep `pyproject.toml`, `uv.lock`, and `requirements.txt` in sync.

## Architecture & Two-Tier Execution
- **Tier 1 (in-book `.md` code cells):** Pure scientific Python (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`) using built-in or self-contained biomedical datasets so every chapter executes offline, rapidly, and in-browser via JupyterLite (Pyodide).
- **Tier 2 (`notebooks/*.ipynb`):** Six GPU/heavyweight Colab companion notebooks (PyTorch neural networks, MedMNIST chest X-ray CNN, Hugging Face clinical LLM next-token probabilities, ESM-2 protein embeddings, fairness/abstention audit, and biomedical capstone starter). Intentionally excluded from `toc.yml` so static builds stay fast.

## Pedagogical House Style (PRIMM + Dual Coding)
- Audience: advanced high-school students and first-year undergraduates (algebra + basic Python; no calculus or linear algebra assumed).
- Every chapter follows: metadata banner → learning objectives → biomedical story → conceptual diagram (Mermaid) / tabs / margin notes → definitions (also in `back/glossary.md`) → PRIMM interactive code labs (**Predict** before **Run**, then **Investigate**) → real-world clinical/scientific bridge (2024–2026 citations) → tiered exercises (**Modify**, **Make**, **Critical Reasoning**) with `:class: dropdown` solutions → key takeaways & look-ahead.
- Colorblind-safe Okabe–Ito palette (`#0072B2`, `#D55E00`, `#009E73`, `#E69F00`, `#CC79A7`, `#56B4E9`) across all figures.
