---
title: Installation
description: Install SpectraHandler with uv or pip.
sidebar:
  order: 1
---

SpectraHandler requires Python 3.13 or newer. It is not on PyPI yet; install it from GitHub.

## With uv

```bash
uv add git+https://github.com/haeussma/SpectraHandler
```

## With pip

```bash
pip install git+https://github.com/haeussma/SpectraHandler
```

Plotting (`spectrahandler.plot`) uses matplotlib, which is installed with it.

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
