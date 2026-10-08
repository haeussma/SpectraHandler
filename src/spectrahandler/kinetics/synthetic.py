"""Replicate time courses of a first-order scheme with known truth, for tests and docs."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.kinetics.scheme import Scheme, Step, concentrations

__all__ = ["make_kinetic_replicates"]


def make_kinetic_replicates(
    key: Array,
    scheme: Scheme,
    rates: Mapping[Step, float],
    *,
    initial: Mapping[str, float],
    n_replicates: int = 4,
    between_sd_log: float = 0.0,
    noise: float = 0.002,
    n_time: int = 40,
    n_wavelength: int = 30,
    condition: str = "synthetic",
) -> tuple[SpectralDataset, Array, Array]:
    """Replicate runs of one condition, each with its own true rates.

    Each replicate's log rates are the given log rates plus Gaussian scatter with
    standard deviation ``between_sd_log``, the replicate-to-replicate variation of a real
    experiment. Species spectra are Gaussian bands (width 30 nm, height 0.02 absorbance
    per concentration unit) with centres spread from 380 to 560 nm. Time runs from 0 to
    five times the slowest step's time constant, in seconds.

    Args:
        key: PRNG key for the rate scatter and the noise.
        scheme: The reaction scheme.
        rates: True rate per step, in 1/s; the geometric centre of the replicates.
        initial: Initial concentration per species; species left out start at 0.
        n_replicates: Number of runs.
        between_sd_log: Standard deviation of log rates between replicates.
        noise: Standard deviation of the added Gaussian noise, in absorbance units.
        n_time: Timepoints per run.
        n_wavelength: Channels between 340 and 700 nm.
        condition: Condition label of every run.

    Returns:
        The dataset, the true log rates with shape ``(n_replicates, n_steps)``, and the
        true spectra with shape ``(n_species, n_wavelength)`` in the dataset's species
        order.

    Raises:
        ValueError: If ``rates`` does not name exactly the scheme's steps.
    """
    if set(rates) != set(scheme.steps):
        raise ValueError(f"rates must name exactly the steps {scheme.steps}")
    species = scheme.species
    steps = scheme.index_steps(species)
    key_rates, key_noise = jax.random.split(key)
    centre = jnp.log(jnp.asarray([rates[step] for step in scheme.steps]))
    log_rates = centre + between_sd_log * jax.random.normal(
        key_rates, (n_replicates, scheme.n_steps)
    )

    wavelength = jnp.linspace(340.0, 700.0, n_wavelength)
    centres = jnp.linspace(380.0, 560.0, len(species))
    spectra = 0.02 * jnp.exp(-0.5 * ((wavelength[None, :] - centres[:, None]) / 30.0) ** 2)
    time = jnp.linspace(0.0, 5.0 / min(rates.values()), n_time)
    c0 = jnp.asarray([float(initial.get(name, 0.0)) for name in species])
    profiles = jax.vmap(lambda lr: concentrations(time, jnp.exp(lr), c0, steps))(log_rates)
    absorbance = profiles @ spectra + noise * jax.random.normal(
        key_noise, (n_replicates, n_time, n_wavelength)
    )

    dataset = SpectralDataset.create(
        absorbance=absorbance,
        time=jnp.broadcast_to(time, (n_replicates, n_time)),
        wavelength=wavelength,
        species=species,
        initial_state=jnp.broadcast_to(c0, (n_replicates, len(species))),
        run_ids=tuple(f"replicate{i + 1}" for i in range(n_replicates)),
        conditions=(condition,) * n_replicates,
        time_unit="s",
    )
    return dataset, log_rates, spectra
