"""Shared pytest fixtures."""

from pathlib import Path

import jax
import pytest
from jax import Array

# Spectral fitting in float32 silently loses precision in least-squares residuals.
jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> Path:
    """Directory holding reference spectra and regression fixtures."""
    return DATA_DIR


@pytest.fixture
def key() -> Array:
    """A fixed JAX PRNG key, so every test is reproducible."""
    return jax.random.key(0)
