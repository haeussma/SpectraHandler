"""The synthetic generator produces what the recovery test assumes."""

import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution import make_easy_dataset


def test_shapes_and_truth_line_up(key: Array) -> None:
    ds, spectra, concentrations = make_easy_dataset(key)
    assert spectra.shape == (ds.n_species, ds.n_wavelength)
    assert concentrations.shape == (ds.n_run, ds.n_time, ds.n_species)
    assert ds.species == ("a", "b", "c")


def test_truth_reconstructs_the_data(key: Array) -> None:
    ds, spectra, concentrations = make_easy_dataset(key, noise=0.0)
    expected = jnp.einsum("rtk,kw->rtw", concentrations, spectra)
    assert jnp.allclose(ds.absorbance, expected, atol=1e-10)


def test_spectra_are_well_separated(key: Array) -> None:
    """'Easy' means the bands barely overlap; the recovery test relies on it."""
    _, spectra, _ = make_easy_dataset(key)
    normed = spectra / jnp.linalg.norm(spectra, axis=1, keepdims=True)
    gram = normed @ normed.T
    off_diagonal = gram - jnp.diag(jnp.diag(gram))
    assert float(off_diagonal.max()) < 0.3


def test_is_deterministic_given_a_key(key: Array) -> None:
    first, _, _ = make_easy_dataset(key)
    second, _, _ = make_easy_dataset(key)
    assert jnp.array_equal(first.absorbance, second.absorbance)
