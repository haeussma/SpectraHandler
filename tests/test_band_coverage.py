"""Coverage: does the band contain the truth?

The data are realistic a -> b -> c runs with the first scan dropped, one reference scan of
``a`` at 50 uM, and sigma 0.002. The "harder" set lifts every spectrum by 0.003 AU/uM so
that no true value is exactly zero, which makes it the clean test of the split.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from spectrahandler.curve_resolution import (
    FeasibleBand,
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

SIGMA, REF_CONC = 0.002, 50.0
LIFT = {"realistic": 0.0, "harder": 0.003}


def _profiles(k1: float, k2: float, time: Array) -> Array:
    """Consecutive a -> b -> c at 12.5 uM total, shape ``(n_time, 3)``."""
    a = jnp.exp(-k1 * time)
    b = k1 / (k2 - k1) * (jnp.exp(-k1 * time) - jnp.exp(-k2 * time))
    return 12.5 * jnp.stack([a, b, 1.0 - a - b], axis=-1)


def _dataset(
    key: Array, spectra: Array, concentrations: Array, initial_state: Array
) -> SpectralDataset:
    """Noisy runs plus a noisy 50 uM reference scan of ``a``, as a per-uM spectrum."""
    k_data, k_ref = jax.random.split(key)
    n_run, n_time, _ = concentrations.shape
    clean = jnp.einsum("rtk,kw->rtw", concentrations, spectra)
    ref = spectra[0] + SIGMA / REF_CONC * jax.random.normal(k_ref, spectra[0].shape)
    _, _, n_wavelength = clean.shape
    return SpectralDataset.create(
        absorbance=clean + SIGMA * jax.random.normal(k_data, clean.shape),
        time=jnp.tile(jnp.linspace(0.0, 10.0, n_time + 1)[1:], (n_run, 1)),
        wavelength=jnp.linspace(340.0, 700.0, n_wavelength),
        species=("a", "b", "c"),
        initial_state=initial_state,
        run_ids=tuple(f"run{i}" for i in range(n_run)),
        reference_spectra=jnp.full((3, n_wavelength), jnp.nan).at[0].set(ref),
        reference_sigma=jnp.array([SIGMA / REF_CONC, jnp.nan, jnp.nan]),
    )


def _single_run(seed: int, lift: float) -> tuple[SpectralDataset, Array, Array]:
    """Single-run data for one noise draw."""
    _, spectra, concentrations = make_realistic_dataset(jax.random.key(0), noise=0.0)
    spectra, concentrations = spectra + lift, concentrations[:, 1:]
    initial = jnp.array([[12.5, 0.0, 0.0]])
    return (
        _dataset(jax.random.key(1000 + seed), spectra, concentrations, initial),
        spectra,
        concentrations,
    )


def _match(fitted: Array, truth: Array) -> Array:
    """For each true species, the fitted index with the most similar spectrum.

    Only ``a`` has a reference, so ``b`` and ``c`` come back in either order.
    """
    normed_truth = truth / jnp.linalg.norm(truth, axis=-1, keepdims=True)
    normed_fit = fitted / jnp.linalg.norm(fitted, axis=-1, keepdims=True)
    taken: list[int] = []
    for row in normed_truth @ normed_fit.T:
        taken.append(next(int(i) for i in jnp.argsort(-row) if int(i) not in taken))
    return jnp.asarray(taken)


def _coverage(band: FeasibleBand, spectra: Array, concentrations: Array) -> tuple[float, float]:
    """Fraction of true amounts and of true spectrum values inside the band."""
    p = _match(band.spectra_draws.mean(axis=0)[0], spectra)
    c_in = (concentrations >= band.concentration_lower[..., p]) & (
        concentrations <= band.concentration_upper[..., p]
    )
    s_in = (spectra >= band.spectra_lower[0, p]) & (spectra <= band.spectra_upper[0, p])
    return float(c_in.mean()), float(s_in.mean())


def _flat_coverage(band: FeasibleBand, spectra: Array, concentrations: Array) -> float:
    """Fraction of true amounts inside the central 95% of the flat draws."""
    p = _match(band.spectra_draws.mean(axis=0)[0], spectra)
    lo, hi = jnp.percentile(band.concentration_draws, jnp.array([2.5, 97.5]), axis=0)
    inside = (concentrations >= lo[..., p]) & (concentrations <= hi[..., p])
    return float(inside.mean())


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dataset", ["realistic", "harder"])
def test_band_contains_the_truth(dataset: str, seed: int) -> None:
    """The band holds at least 95% of the true amounts and of the true spectrum values."""
    data, spectra, concentrations = _single_run(seed, LIFT[dataset])
    band = resolve_band(data, jax.random.key(seed))
    amounts, values = _coverage(band, spectra, concentrations)
    assert amounts >= 0.95, f"band holds {amounts:.3f} of the true amounts"
    assert values >= 0.95, f"band holds {values:.3f} of the true spectrum values"


@pytest.mark.parametrize("seed", [0, 1])
def test_band_width_matches_the_single_run(seed: int) -> None:
    """The median amount band is about 4.5 uM wide; the tolerance covers the seed-to-seed spread."""
    data, _, _ = _single_run(seed, LIFT["harder"])
    band = resolve_band(data, jax.random.key(seed))
    width = float(jnp.median(band.concentration_upper - band.concentration_lower))
    assert width == pytest.approx(4.5, abs=0.4)


@pytest.mark.parametrize("chain_seed", [1, 11])
def test_flat_draws_cover_an_interior_truth(chain_seed: int) -> None:
    """A chain that starts in a gap of the feasible region must still cover an interior truth.

    Chain keys 1 and 11 are start points where the draws can stay infeasible; the flat
    summary of the draws must hold at least 90% of the true amounts.
    """
    data, spectra, concentrations = _single_run(1, LIFT["harder"])
    band = resolve_band(data, jax.random.key(chain_seed), n_iter=32000)
    assert _flat_coverage(band, spectra, concentrations) >= 0.9


def test_a_run_lacking_a_species_narrows_the_band() -> None:
    """Shared spectra help when a run changes which species are present.

    A second run starting from ``b`` has no ``a`` at any time, and the non-negativity
    of that zero cuts the region. A second run that only changes the rates does not help.
    """
    _, spectra, _ = make_realistic_dataset(jax.random.key(0), noise=0.0)
    spectra = spectra + LIFT["harder"]
    time = jnp.linspace(0.0, 10.0, 30)[1:]
    from_a = _profiles(0.6, 0.4, time)
    b_decay = jnp.exp(-0.4 * time)
    from_b = 12.5 * jnp.stack([jnp.zeros_like(time), b_decay, 1.0 - b_decay], axis=-1)

    def width(concentrations: Array, initial: Array) -> tuple[float, float]:
        data = _dataset(jax.random.key(5), spectra, concentrations, initial)
        band = resolve_band(data, jax.random.key(2))
        amounts, _ = _coverage(band, spectra, concentrations)
        first_run = band.concentration_upper[0] - band.concentration_lower[0]
        return float(jnp.median(first_run)), amounts

    one, one_cov = width(from_a[None], jnp.array([[12.5, 0.0, 0.0]]))
    two, two_cov = width(jnp.stack([from_a, from_b]), jnp.array([[12.5, 0, 0], [0, 12.5, 0.0]]))
    assert two < 0.8 * one, f"band {one:.2f} -> {two:.2f} uM; expected a clear narrowing"
    assert min(one_cov, two_cov) >= 0.95
