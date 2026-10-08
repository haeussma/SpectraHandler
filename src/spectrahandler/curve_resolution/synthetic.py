"""Synthetic datasets with known ground truth.

Shipped with the package so that methods can be tried, and checked against a known answer,
without measured data.
"""

import jax
import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset

__all__ = ["make_easy_dataset", "make_realistic_dataset"]

#: Per species, its Gaussian bands as (centre nm, width nm, peak height).
type _Bands = tuple[tuple[tuple[float, float, float], ...], ...]

#: One well-separated band per species; heights are arbitrary, absorbance peaks near 12.
_EASY_BANDS: _Bands = (
    ((380.0, 18.0, 0.9),),
    ((470.0, 22.0, 0.7),),
    ((550.0, 20.0, 1.0),),
)

#: Heights in absorbance per uM per cm. a and c carry a strong near-UV band plus a weaker
#: visible one; b is unimodal. a and c overlap (cosine similarity 0.47).
_REALISTIC_BANDS: _Bands = (
    ((355.0, 16.0, 0.026), (525.0, 32.0, 0.009)),
    ((470.0, 24.0, 0.011),),
    ((388.0, 18.0, 0.028), (555.0, 28.0, 0.010)),
)


def _make_dataset(
    key: Array,
    bands: _Bands,
    n_time: int,
    n_wavelength: int,
    noise: float,
    total_concentration: float,
) -> tuple[SpectralDataset, Array, Array]:
    """Build a consecutive a -> b -> c dataset from per-species Gaussian bands."""
    wavelength = jnp.linspace(340.0, 700.0, n_wavelength)
    spectra = jnp.stack(
        [
            sum(h * jnp.exp(-0.5 * ((wavelength - c) / w) ** 2) for c, w, h in species_bands)
            for species_bands in bands
        ]
    )

    time = jnp.linspace(0.0, 10.0, n_time)
    # Consecutive first-order a -> b -> c, k1 = 0.6, k2 = 0.4 per hour. The closed form
    # needs no ODE solver.
    k1, k2 = 0.6, 0.4
    frac_a = jnp.exp(-k1 * time)
    frac_b = k1 / (k2 - k1) * (jnp.exp(-k1 * time) - jnp.exp(-k2 * time))
    concentrations = (
        total_concentration
        * jnp.stack([frac_a, frac_b, 1.0 - frac_a - frac_b], axis=-1)[None, :, :]
    )

    clean = jnp.einsum("rtk,kw->rtw", concentrations, spectra)
    absorbance = clean + noise * jax.random.normal(key, clean.shape)

    dataset = SpectralDataset.create(
        absorbance=absorbance,
        time=time[None, :],
        wavelength=wavelength,
        species=("a", "b", "c"),
        initial_state=jnp.array([[total_concentration, 0.0, 0.0]]),
        run_ids=("synthetic",),
    )
    return dataset, spectra, concentrations


def make_easy_dataset(
    key: Array,
    *,
    n_time: int = 30,
    n_wavelength: int = 64,
    noise: float = 0.002,
    total_concentration: float = 12.5,
) -> tuple[SpectralDataset, Array, Array]:
    """Generate a three-species dataset in the easy regime.

    Three well-separated Gaussian bands and a consecutive a -> b -> c time course, which
    is a smoke test for any method. Nothing here is meant to be difficult: failure on
    this data means the method is broken. Note that it is cleaner than any
    instrument: absorbance peaks near 12 and the signal is about 6000 times the noise.

    Args:
        key: PRNG key for the noise draw.
        n_time: Number of timepoints, spread evenly over 10 h.
        n_wavelength: Channels between 340 and 700 nm.
        noise: Standard deviation of the added Gaussian noise, in absorbance units.
        total_concentration: Sum over species at every timepoint, in uM.

    Returns:
        A tuple of the dataset, the true spectra with shape
        ``(n_species, n_wavelength)``, and the true concentrations with shape
        ``(n_run, n_time, n_species)``.
    """
    return _make_dataset(key, _EASY_BANDS, n_time, n_wavelength, noise, total_concentration)


def make_realistic_dataset(
    key: Array,
    *,
    n_time: int = 30,
    n_wavelength: int = 64,
    noise: float = 0.002,
    total_concentration: float = 12.5,
) -> tuple[SpectralDataset, Array, Array]:
    """Generate a three-species dataset at instrument-like signal and noise.

    Same a -> b -> c time course as :func:`make_easy_dataset`, but species a and c have
    two bands each (a strong near-UV band and a weaker visible one) and b has one, the
    absorbance peaks near 0.33 AU, and the signal is about 160 times the noise. The noise
    is one Gaussian standard deviation shared by every channel and timepoint, which is
    exactly what resolve_band assumes. Spectra of a and c overlap, so some rotational
    ambiguity is expected here and a wide band is not by itself a fault.

    Args:
        key: PRNG key for the noise draw.
        n_time: Number of timepoints, spread evenly over 10 h.
        n_wavelength: Channels between 340 and 700 nm.
        noise: Standard deviation of the added Gaussian noise, in absorbance units.
        total_concentration: Sum over species at every timepoint, in uM.

    Returns:
        A tuple of the dataset, the true spectra in absorbance per uM per cm with shape
        ``(n_species, n_wavelength)``, and the true concentrations in uM with shape
        ``(n_run, n_time, n_species)``.
    """
    return _make_dataset(key, _REALISTIC_BANDS, n_time, n_wavelength, noise, total_concentration)
