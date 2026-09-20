"""Synthetic datasets with known ground truth.

Lives in the package rather than in tests because the suite must run before any real
fixture is reachable, and because coverage checks reuse it. See ``spec.md`` §5.
"""

import jax
import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset

__all__ = ["make_easy_dataset"]

#: Band centre (nm), width (nm), and peak molar absorptivity for each species.
_BANDS = ((380.0, 18.0, 0.9), (470.0, 22.0, 0.7), (550.0, 20.0, 1.0))


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
    is what v0 is validated against. Nothing here is meant to be difficult: failure on
    this data means the model or the sampler is broken.

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
    wavelength = jnp.linspace(340.0, 700.0, n_wavelength)
    centres, widths, heights = (jnp.asarray(v) for v in zip(*_BANDS, strict=True))
    spectra = heights[:, None] * jnp.exp(
        -0.5 * ((wavelength[None, :] - centres[:, None]) / widths[:, None]) ** 2
    )

    time = jnp.linspace(0.0, 10.0, n_time)
    # Consecutive first-order a -> b -> c, k1 = 0.6, k2 = 0.4 per hour. Closed form, so
    # no ODE solver is pulled in at this stage.
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
