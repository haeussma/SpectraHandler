"""The band with backgrounds: does it still contain the truth when the baseline is off?

Two realistic a -> b -> c runs on 96 wavelengths, the second starting from b. Every run gets
its own offset, and all share a fixed instrument wiggle of 0.002 AU that no background shape
can fit. No species absorbs above 660 nm. That window is too narrow to pin a tilt per run
(see test_background.py), so the gate uses offsets only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from spectrahandler.curve_resolution import (
    Background,
    CredibleBand,
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

SIGMA, REF_CONC, TOTAL = 0.002, 50.0, 12.5
WINDOW = ((660.0, 700.0),)
OFFSETS = (Background.offset(),)
OFFSET = jnp.array([0.005, -0.005])


def _data(seed: int, wiggle: float = 0.002) -> tuple[SpectralDataset, Array, Array]:
    """The dataset with an off baseline, the true spectra and the true concentrations."""
    clean, spectra, from_a = make_realistic_dataset(jax.random.key(0), noise=0.0, n_wavelength=96)
    wavelength, time = clean.wavelength, clean.time[0]
    decay = jnp.exp(-0.4 * time)
    from_b = TOTAL * jnp.stack([jnp.zeros_like(time), decay, 1.0 - decay], axis=-1)
    concentrations = jnp.concatenate([from_a, from_b[None]])
    baseline = OFFSET[:, None] + wiggle * jnp.sin(2 * jnp.pi * wavelength / 15.0)
    k_data, k_ref = jax.random.split(jax.random.key(1000 + seed))
    absorbance = jnp.einsum("rtk,kw->rtw", concentrations, spectra) + baseline[:, None, :]
    reference = spectra[0] + SIGMA / REF_CONC * jax.random.normal(k_ref, spectra[0].shape)
    data = SpectralDataset.create(
        absorbance=absorbance + SIGMA * jax.random.normal(k_data, absorbance.shape),
        time=jnp.tile(time, (2, 1)),
        wavelength=wavelength,
        species=("a", "b", "c"),
        initial_state=jnp.array([[TOTAL, 0.0, 0.0], [0.0, TOTAL, 0.0]]),
        run_ids=("from a", "from b"),
        reference_spectra=jnp.full((3, 96), jnp.nan).at[0].set(reference),
        reference_sigma=jnp.array([SIGMA / REF_CONC, jnp.nan, jnp.nan]),
    )
    return data, spectra, concentrations


def _coverage(band: CredibleBand, spectra: Array, concentrations: Array) -> tuple[float, float]:
    """Fraction of true amounts and of true spectrum values inside the band.

    Only ``a`` has a reference, so ``b`` and ``c`` come back in either order: they are
    matched by spectrum.
    """
    fitted = band.spectra_draws.mean(axis=0)[0]
    swapped = float(fitted[1] @ spectra[2]) > float(fitted[1] @ spectra[1])
    p = jnp.array([0, 2, 1]) if swapped else jnp.arange(3)
    c_in = (concentrations[..., p] >= band.concentration_lower) & (
        concentrations[..., p] <= band.concentration_upper
    )
    s_in = (spectra[p] >= band.spectra_lower[0]) & (spectra[p] <= band.spectra_upper[0])
    return float(c_in.mean()), float(s_in.mean())


@pytest.fixture(scope="module")
def resolved() -> tuple[CredibleBand, Array, Array]:
    data, spectra, concentrations = _data(0)
    band = resolve_band(data, jax.random.key(0), backgrounds=OFFSETS, zero_windows=WINDOW)
    return band, spectra, concentrations


def test_band_contains_the_truth_despite_the_baseline(
    resolved: tuple[CredibleBand, Array, Array],
) -> None:
    band, spectra, concentrations = resolved
    amounts, values = _coverage(band, spectra, concentrations)
    assert amounts >= 0.95, f"band holds {amounts:.3f} of the true amounts"
    assert values >= 0.95, f"band holds {values:.3f} of the true spectrum values"


def test_the_result_records_what_was_assumed(
    resolved: tuple[CredibleBand, Array, Array],
) -> None:
    band, _, _ = resolved
    assert band.background_names == ("offset",)
    assert band.background_amplitude.shape == (2, 1)
    assert band.zero_source == "zero windows 660-700 nm, tolerance measured"
    # The wiggle is 0.002 AU; noise smoothed over 20 spectra adds a little. Outside the
    # window an offset has leverage 1, so the tolerance doubles there.
    inside, outside = float(band.baseline_tolerance[-1]), float(band.baseline_tolerance[0])
    assert 0.001 < inside < 0.004
    assert outside == pytest.approx(2 * inside)


def test_offsets_are_recovered(resolved: tuple[CredibleBand, Array, Array]) -> None:
    """The wiggle's mean over the window goes into the offset, so the tolerance bounds the error."""
    band, _, _ = resolved
    error = jnp.abs(band.background_amplitude[:, 0] - OFFSET)
    assert float(error.max()) < float(band.baseline_tolerance[-1])


def test_without_backgrounds_the_off_baseline_is_not_resolved() -> None:
    """An offset of -0.005 AU makes every spectrum negative somewhere: no split fits."""
    data, _, _ = _data(0)
    with pytest.raises(ValueError, match="no feasible split"):
        resolve_band(data, jax.random.key(0), n_restart=2, n_search=2000)


def test_a_tolerance_below_the_measured_one_is_an_error() -> None:
    data, _, _ = _data(0)
    with pytest.raises(ValueError, match=r"baseline_tolerance 0.0005 is below the .* 'from"):
        resolve_band(
            data,
            jax.random.key(0),
            backgrounds=OFFSETS,
            zero_windows=WINDOW,
            baseline_tolerance=5e-4,
        )


def test_backgrounds_and_windows_go_together() -> None:
    data, _, _ = _data(0)
    with pytest.raises(ValueError, match="give both or neither"):
        resolve_band(data, jax.random.key(0), backgrounds=OFFSETS)


def test_a_given_tolerance_without_windows_is_recorded() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    band = resolve_band(data, jax.random.key(1), baseline_tolerance=0.001, n_iter=200, n_burn=100)
    np.testing.assert_allclose(band.baseline_tolerance, 0.001)
    assert band.zero_source == "instrument zero (assumed)"
    assert band.background_amplitude.shape == (1, 0)


def test_a_tolerance_per_wavelength_is_used_as_given() -> None:
    """One number is carried outwards by the fit's leverage; an array is taken as it is."""
    data, _, _ = _data(0)
    given = jnp.full(96, 0.004)
    band = resolve_band(
        data,
        jax.random.key(0),
        backgrounds=OFFSETS,
        zero_windows=WINDOW,
        baseline_tolerance=given,
        n_iter=200,
        n_burn=100,
    )
    np.testing.assert_allclose(band.baseline_tolerance, given)
    assert band.zero_source == "zero windows 660-700 nm, tolerance given"
