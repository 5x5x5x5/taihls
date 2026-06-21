# TAIHLS — project notes for Claude

Interactive textbook built with **Jupyter Book 2 (MyST engine)** — *not* v1.

## Don't use Jupyter Book v1 patterns
- Config is a single `myst.yml` (no `_config.yml`).
- TOC lives in `toc.yml`, pulled in via `extends:` (no `_toc.yml`, no Sphinx).
- CLI is `jupyter book <verb>`: `init`, `start`, `build`, `clean`.
- Content files are `.md` with MyST directives.

## Workflow
- `uv` only. `uv sync`; `uv run jupyter book start [--execute]`; `uv run jupyter book build --html --execute`.
- HTML output → `_build/html/` (a client-rendered SPA: per-page `index.html` + `.json`).

## Gotchas
- **Execution hangs under a sandbox.** `--execute` starts a local Jupyter server +
  websocket on localhost; if loopback is blocked it hangs at "Starting WebSocket".
  Run the build with the sandbox disabled.
- Code cells need a `kernelspec` in the page frontmatter (see any chapter).
- JupyterLite (`project.jupyter.lite: true`) runs Pyodide — keep example deps to
  packages Pyodide ships (numpy/matplotlib/pandas OK).
- `project.binder` must be a full URL; exercise/solution use `:class: dropdown`
  (no sphinx-exercise plugin).

## House style (see content/1_a1-placeholder.md, the exemplar)
Per chapter: learning-objectives admonition → concept-through-example → small
runnable code-cell → exercise + dropdown solution → key takeaways. Audience is
advanced HS / first-year undergrad: assume algebra, no calculus/linear algebra,
define every term on first use, warm first-person-plural voice.
