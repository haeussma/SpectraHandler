# SpectraHandler

Spectral deconvolution for reaction data, built on [JAX](https://docs.jax.dev). It
reports the band of every spectra and concentration-profile split the data allow,
rather than one answer.

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
