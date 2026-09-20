# SpectraHandler

Spectral deconvolution for reaction data, built on [JAX](https://docs.jax.dev) and
[NumPyro](https://num.pyro.ai).

## Install

```bash
uv add spectrahandler
```

## Develop

```bash
uv sync
pre-commit install
uv run pytest
```

Documentation lives in [`docs/`](docs/) (Astro Starlight):

```bash
cd docs && npm install && npm run dev
```

Code standards and agent rules: [CLAUDE.md](CLAUDE.md).
