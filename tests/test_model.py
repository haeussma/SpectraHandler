"""Shapes and prior predictive sanity, before any data is fitted."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpyro.infer import Predictive

from spectrahandler.curve_resolution import curve_resolution_model, make_easy_dataset


@pytest.fixture
def prior_draws(key: Array) -> dict[str, Array]:
    dataset, _, _ = make_easy_dataset(key)
    predictive = Predictive(curve_resolution_model, num_samples=64)
    draws: dict[str, Array] = predictive(
        jax.random.key(1),
        absorbance=None,
        mask=dataset.mask,
        n_wavelength=dataset.n_wavelength,
        n_species=3,
        tau=0.01,
        sigma_scale=0.01,
    )
    return draws


def test_deterministic_shapes(prior_draws: dict[str, Array]) -> None:
    assert prior_draws["spectra"].shape == (64, 3, 64)
    assert prior_draws["concentrations"].shape == (64, 1, 30, 3)


def test_spectra_are_non_negative(prior_draws: dict[str, Array]) -> None:
    assert bool((prior_draws["spectra"] >= 0).all())


def test_spectra_are_scale_fixed(prior_draws: dict[str, Array]) -> None:
    """Mean one per spectrum. Without this the posterior has a free scale direction."""
    assert jnp.allclose(prior_draws["spectra"].mean(axis=-1), 1.0, atol=1e-6)


def test_spectra_are_smooth(prior_draws: dict[str, Array]) -> None:
    """Prior draws should look like UV/Vis bands, not noise. With the walk running per
    channel this is the test that pins tau -- see step 4."""
    curvature = jnp.diff(prior_draws["spectra"], n=2, axis=-1)
    assert float(jnp.abs(curvature).mean()) < 0.05


def test_concentrations_are_non_negative(prior_draws: dict[str, Array]) -> None:
    assert bool((prior_draws["concentrations"] >= 0).all())
