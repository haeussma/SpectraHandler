# SpectraHandler

Spectral deconvolution for reaction data. This is a **library for other researchers**, not
an internal analysis repo — public API, docstrings and docs pages are part of the product.

## Numerical stack

- **JAX** (`jax.numpy as jnp`) for all array maths and models. Models are pure functions,
  JIT-compilable, differentiable.
- **NumPyro** for inference — priors, MCMC/NUTS, SVI. Report posteriors, not bare point
  estimates, wherever the science allows it.
- **NumPy** (`numpy as np`) only at the boundaries: file IO, interop, and `numpy.testing`
  in tests. Compute happens in `jnp`.
- **Do not import SciPy.** It arrives transitively via JAX; that is not permission to use
  it. If you need an optimiser or special function, take the JAX equivalent
  (`jax.scipy`, `optax`, `jaxopt`) and say so.

### JAX rules

- **float64 everywhere.** `jax.config.update("jax_enable_x64", True)` — float32 silently
  destroys least-squares residuals over spectra. Tests enable it in `conftest.py`.
- **Explicit PRNG keys.** Thread `jax.random.key(...)` through; never a global seed, never
  `numpy.random`.
- No Python-side loops over array axes — `vmap` over spectra and timepoints.
- No in-place mutation; use `.at[].set()`.
- Annotate arrays as `jax.Array`. Note the shape and units in the docstring.

## Toolchain — uv only

`uv` owns the Python environment. Never `pip`, never a bare `python`, never
`source .venv/bin/activate`.

```bash
uv sync                      # install (incl. dev group)
uv run pytest                # tests
uv run ruff format .         # format
uv run ruff check --fix .    # lint
uv run ty check              # type check
uv add <pkg>                 # runtime dep
uv add --dev <pkg>           # dev dep
```

`uv.lock` **is committed.** It does not constrain anyone who installs the library — pip and
uv resolve fresh from `[project.dependencies]` — it only pins CI and contributor
environments. With JAX in the tree that pinning is worth having.

## Code standards

- **Full type annotations.** Every function and method annotates all parameters *and* the
  return type — including `-> None`. Enforced by ruff `ANN`; correctness by `ty`.
- **Google-style docstrings** on every public module, class and function. Enforced by
  ruff `D` with `convention = "google"`.
- **Ruff is the only linter and formatter.** Config lives in `pyproject.toml`; don't add
  black, isort, flake8 or mypy alongside it.
- Line length 100.
- No bare `except:`; catch the specific exception.
- Prefer the stdlib, then an already-installed dependency, before adding a new one.

## Third-party libraries: Context7 first

**Before writing code against any third-party library, fetch its current docs via the
Context7 MCP server.** Not optional, and it applies hardest to exactly the libraries in
this repo: JAX and NumPyro move fast and their APIs have churned. Training data lags.

1. `resolve-library-id` with the library name and the actual question
2. `query-docs` with the resolved `/org/project` id and the full question
3. Write the code from what the docs return

Skip it only for refactoring, business logic, or general Python.

## Documentation — Astro Starlight in `docs/`

User-facing documentation lives in `docs/` as an Astro Starlight site. It is **part of the
definition of done**: a public API change that doesn't update the docs is incomplete.

```bash
cd docs && npm install    # once
npm run dev               # local preview
npm run build             # what CI runs
```

- Pages are Markdown/MDX under `docs/src/content/docs/`, with `title` and `description`
  frontmatter on every page.
- `guides/` for narrative and how-to, `reference/` for API surface. The sidebar
  autogenerates from those directories; control order with `sidebar.order` in frontmatter.
- Write for a researcher who has reaction spectra and no JAX experience. Every guide gets
  a runnable code block.
- Docstrings stay the source of truth for API detail; docs pages explain *when and why*.
- Deployed to GitHub Pages by `.github/workflows/docs.yml` on push to `main`.

## Where things get written

`docs/` is a **published website**. Everything under `docs/src/content/docs/` goes public
on GitHub Pages. Internal working material never goes there.

| Document | Goes in | Public | Lifetime |
|---|---|---|---|
| Implementation plan — sequenced work | `plans/NNN-slug.md` | no | delete when done |
| ADR — a decision with lasting consequence | `docs/decisions/NNNN-slug.md` | no | forever |
| The science: methods, concepts, papers | `~/brain/wiki/methods/<slug>.md` | no | forever |
| Which methods this repo relies on | `~/brain/projects/spectrahandler.md` | no | forever |
| User-facing explanation and API docs | `docs/src/content/docs/` | **yes** | forever |
| Test fixture provenance and traps | `tests/data/README.md` | no | with the data |

The split that matters: **a plan is what we are going to do; an ADR is why we chose it.**
When a plan contains a choice that would be expensive to reverse, lift it into an ADR and
link to it. Plans get deleted. ADRs do not.

`docs/decisions/` sits outside `docs/src/content/docs/`, so Astro does not build it — it
is versioned with the code and invisible to the site. `/spec-link` writes these and wires
the matching brain page in the other direction.

**Do not start building a feature without a plan.** If one does not exist for the work at
hand, write it first — `/brainstorm` if the shape is still open, `/write-plan` if it is not.

## Tests

- pytest, tests in `tests/`, mirroring `src/spectrahandler/`.
- Non-trivial logic ships with a test — especially fitting, baseline correction and peak
  models. Numerical assertions use `pytest.approx` or `numpy.testing`, never `==`.
- Use the `key` fixture for anything stochastic; results must be reproducible.
- Reference spectra and regression fixtures go in `tests/data/`, small and committed.

## Quality gates

| Layer | When | What |
|---|---|---|
| Claude hook | after every file edit | `ruff format` + `ruff check --fix` on that file |
| Claude hook | end of each turn | `ty check` + `pytest` |
| pre-commit | `git commit` | ruff, `uv lock --check`, `ty`, whitespace/YAML/TOML |
| GitHub Actions | push / PR | the whole set on a clean checkout |

`pre-commit install` once per clone. The Claude hooks exist because pre-commit only fires
at commit time — they catch a broken edit within seconds instead of twenty minutes later.

## Scientific-code rules

- Physical units belong in the name or the docstring (`wavelength_nm`, `time_s`).
- Never silently drop or interpolate data points; log or raise.
- Fitting code exposes its tuning knobs (initial guesses, priors, bounds, tolerances) as
  arguments with defaults — real spectra need tuning a clean model can't anticipate.
- Keep notebooks out of `src/`.

## Vault link

When a method implemented here also has a page in `~/brain/wiki/methods/`, run
`/spec-link` — it fills the page's `implements:` frontmatter, writes the matching ADR
under `docs/decisions/`, and logs the operation. Science and code stay linked in both
directions or the link rots.
