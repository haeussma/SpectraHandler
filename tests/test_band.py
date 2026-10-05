"""Invariants of resolve_band that hold on any data, independent of coverage."""

import jax
import jax.numpy as jnp
import pytest

from spectrahandler.curve_resolution import (
    FeasibleBand,
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

#: The realistic set's noise, and the concentration (uM) of the simulated reference scan.
SIGMA, REF_CONC = 0.002, 50.0


def _with_reference(dataset: SpectralDataset, spectrum_a: jax.Array) -> SpectralDataset:
    """The same data plus a reference spectrum for species ``a`` only."""
    refs = jnp.full((dataset.n_species, dataset.n_wavelength), jnp.nan).at[0].set(spectrum_a)
    return SpectralDataset.create(
        absorbance=dataset.absorbance,
        time=dataset.time,
        wavelength=dataset.wavelength,
        species=dataset.species,
        initial_state=dataset.initial_state,
        run_ids=dataset.run_ids,
        reference_spectra=refs,
        reference_sigma=jnp.array([SIGMA / REF_CONC, jnp.nan, jnp.nan]),
    )


@pytest.fixture(scope="module")
def resolved() -> tuple[SpectralDataset, FeasibleBand]:
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, spectra[0])
    return data, resolve_band(data, jax.random.key(1), n_iter=2000)


def test_shapes(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    data, band = resolved
    c_shape = (data.n_run, data.n_time, data.n_species)
    s_shape = (data.n_species, data.n_wavelength)
    assert band.concentration_lower.shape == band.concentration_upper.shape == c_shape
    assert band.spectra_lower.shape == band.spectra_upper.shape == s_shape
    assert band.concentration_ambiguity.shape == (2, *c_shape)
    assert band.spectra_ambiguity.shape == (2, *s_shape)
    assert band.concentration_draws.shape[1:] == c_shape
    assert band.spectra_draws.shape[1:] == s_shape


def test_one_reference_leaves_four_free_numbers(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    """9 entries of T, minus 3 for closure, minus 3 for the reference, plus 1 shared."""
    assert resolved[1].n_free == 4


def test_no_reference_leaves_six_free_numbers() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    assert resolve_band(data, jax.random.key(1), n_iter=200, n_burn=100).n_free == 6


def test_sigma_is_estimated(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    assert resolved[1].sigma == pytest.approx(SIGMA, rel=0.1)


def test_band_brackets_ambiguity_and_is_non_negative(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    _, band = resolved
    assert bool((band.concentration_lower >= 0).all())
    assert bool((band.spectra_lower >= 0).all())
    # The ambiguity extremes may sit a little below zero (the noise slack); the band is
    # clipped at zero, so compare against the clipped extreme.
    floor = jnp.clip(band.concentration_ambiguity[0], 0.0)
    assert bool((band.concentration_lower <= floor + 1e-12).all())
    assert bool((band.concentration_upper >= band.concentration_ambiguity[1]).all())
    assert bool((band.spectra_upper >= band.spectra_ambiguity[1]).all())


def test_draws_respect_closure(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Closure holds in least squares over the rows, so single rows scatter at noise level.

    Measured: overall mean 12.497 uM, rows between 12.31 and 12.71 uM.
    """
    data, band = resolved
    totals = band.concentration_draws.sum(axis=-1).mean(axis=0)
    total = float(data.initial_state.sum())
    assert float(totals.mean()) == pytest.approx(total, abs=0.02)
    assert float(jnp.abs(totals - total).max()) < 0.3


def test_draws_reproduce_the_data(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Every split reproduces the data to the noise: residual rms close to sigma."""
    data, band = resolved
    predicted = jnp.einsum("drtk,dkw->rtw", band.concentration_draws, band.spectra_draws)
    predicted = predicted / band.concentration_draws.shape[0]
    rms = float(jnp.sqrt(((predicted - data.absorbance) ** 2).mean()))
    assert rms < 1.5 * SIGMA


def test_reference_lies_inside_its_band(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    data, band = resolved
    ref = data.reference_spectra[0]
    assert bool(((ref >= band.spectra_lower[0]) & (ref <= band.spectra_upper[0])).all())


def test_requires_fully_measured_runs() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    masked = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        mask=data.mask.at[0, 3].set(False),
    )
    with pytest.raises(NotImplementedError, match="fully measured"):
        resolve_band(masked, jax.random.key(1))


def test_contradictory_reference_is_reported() -> None:
    """A negative reference cannot be a non-negative spectrum: no split is feasible."""
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, -spectra[0])
    with pytest.raises(ValueError, match="no feasible split"):
        resolve_band(data, jax.random.key(1), n_restart=2, n_search=2000)
