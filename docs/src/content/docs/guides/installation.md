---
title: Installation
description: Install SpectraHandler with uv or pip.
sidebar:
  order: 1
---

SpectraHandler requires Python 3.13 or newer.

## With uv

```bash
uv add spectrahandler
```

## With pip

```bash
pip install spectrahandler
```

## Plotting

`spectrahandler.plot` draws data, deconvolutions, residuals and diagnostics with
matplotlib, which is an optional extra:

```bash
uv add "spectrahandler[plot]"
# or: pip install "spectrahandler[plot]"
```

## Enable float64

JAX defaults to 32-bit floats. Least-squares residuals over spectra lose meaningful
precision at that width, so enable 64-bit before importing anything that builds arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## GPU

The default install pulls CPU-only `jaxlib`. For CUDA, install the matching JAX wheel
first, following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then add SpectraHandler.

## From source

```bash
git clone https://github.com/haeussma/SpectraHandler
cd SpectraHandler
uv sync
uv run pytest
```
