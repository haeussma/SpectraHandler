"""The v0 curve resolution model.

Deliberately the simplest thing that can identify spectra and profiles at all: one
shared noise scalar, no baseline, no reference terms, no smoothness on the
concentrations. See ``plans/001-bayesian-curve-resolution/spec.md`` §4 for what is
missing and why each omission is safe on easy data.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.nn import softplus

__all__ = ["curve_resolution_model"]


def curve_resolution_model(
    absorbance: Array | None,
    mask: Array,
    n_wavelength: int,
    n_species: int,
    *,
    tau: float,
    sigma_scale: float,
) -> None:
    """NumPyro model for absorbance as concentrations times pure spectra.

    Each spectrum is a softplus-transformed second order random walk over the
    wavelength channels, rescaled to mean one. Non-negativity keeps the fit physical;
    the rescaling removes the ``C * a, S / a`` ambiguity that would otherwise leave the
    posterior with a free direction. There is no basis expansion -- the walk runs per
    channel, which is why the grid must be binned. See ``spec.md`` section 4.

    Args:
        absorbance: Observations, shape ``(n_run, n_time, n_wavelength)``, or ``None``
            to draw from the prior.
        mask: True where measured, shape ``(n_run, n_time)``.
        n_wavelength: Number of wavelength channels, after binning.
        n_species: Number of components to resolve.
        tau: Fixed scale of the random walk, **per channel**. It sets the curvature
            directly, and the initial slope as ``tau * sqrt(n_wavelength)`` -- the
            spread the walk itself accumulates over the grid, so that no channel is
            privileged and ``tau`` is the single knob for how far a spectrum may depart
            from flat. As a curvature it scales with the square of the bin width:
            rebinning changes it.
        sigma_scale: Scale of the ``HalfNormal`` prior on the noise standard deviation.
    """
    n_run, n_time = mask.shape

    # Second order random walk, non-centred: a level, an initial slope, and per-channel
    # curvature. The initial slope is given the spread the walk itself accumulates over
    # the grid, ``tau * sqrt(n_wavelength)``, so no channel is privileged and ``tau`` is
    # the single knob for how far a spectrum may depart from flat. Left at unit scale it
    # is ~1.4 per channel, which ramps theta across +-90 over 64 channels: every prior
    # draw is then a softplus hinge rather than an absorption band, at any tau.
    theta_init = jnp.asarray(
        numpyro.sample("theta_init", dist.Normal(0.0, 1.0).expand([n_species, 2]).to_event(2))
    )
    curvature = jnp.asarray(
        numpyro.sample(
            "curvature",
            dist.Normal(0.0, 1.0).expand([n_species, n_wavelength - 2]).to_event(2),
        )
    )
    # Truly non-centred: the site is a standardised curvature and tau scales it here, so
    # every latent sits at unit prior scale whatever tau is. Sampling the site at scale
    # tau instead is the same distribution but measurably worse geometry.
    initial_slope = jnp.sqrt(n_wavelength) * theta_init[:, 1:2]
    slope = tau * jnp.cumsum(jnp.concatenate([initial_slope, curvature], axis=1), axis=1)
    theta = jnp.concatenate(
        [theta_init[:, 0:1], theta_init[:, 0:1] + jnp.cumsum(slope, axis=1)], axis=1
    )

    unnormalised = softplus(theta)
    spectra = numpyro.deterministic(
        "spectra", unnormalised / unnormalised.mean(axis=-1, keepdims=True)
    )

    c_raw = numpyro.sample(
        "c_raw", dist.Normal(0.0, 1.0).expand([n_run, n_time, n_species]).to_event(3)
    )
    concentrations = numpyro.deterministic("concentrations", softplus(c_raw))

    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_scale))
    predicted = jnp.einsum("rtk,kw->rtw", concentrations, spectra)
    with numpyro.handlers.mask(mask=mask[:, :, None]):
        numpyro.sample("absorbance", dist.Normal(predicted, sigma), obs=absorbance)
