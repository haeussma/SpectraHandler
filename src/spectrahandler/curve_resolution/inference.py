"""NUTS driver.

Non-dimensionalises absorbance before sampling, because NUTS geometry is much better
when the observations are O(1) -- see ``spec.md`` section 8.
"""

import jax.numpy as jnp
from jax import Array
from numpyro.infer import MCMC, NUTS

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.model import curve_resolution_model

__all__ = ["fit"]


def fit(
    dataset: SpectralDataset,
    n_species: int,
    key: Array,
    *,
    tau: float = 0.01,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 2,
) -> MCMC:
    """Sample the v0 curve resolution posterior with NUTS.

    Absorbance is divided by its largest absolute value before sampling. Because each
    spectrum is normalised to mean one, sampled ``concentrations`` are in those
    non-dimensional units, not in ``dataset.concentration_unit``.

    Args:
        dataset: Validated observations. ``v0`` requires ``mask.all()``.
        n_species: Number of components to resolve. Given, not inferred.
        key: PRNG key for the sampler.
        tau: Fixed random walk curvature scale, per wavelength channel.
        num_warmup: Warmup iterations. 200 is the development setting.
        num_samples: Post-warmup draws per chain.
        num_chains: Chains; two is the minimum that lets R-hat mean anything.

    Returns:
        The completed ``MCMC`` object. Call ``print_summary()`` for R-hat, ESS and the
        divergence count.

    Raises:
        NotImplementedError: If ``dataset.mask`` has any False entry. Ragged runs are
            a later step; the contract carries the machinery, v0 does not use it.
    """
    if not bool(dataset.mask.all()):
        raise NotImplementedError("v0 requires fully measured runs; masking is step 4+")

    scale = float(jnp.nanmax(jnp.abs(dataset.absorbance)))
    kernel = NUTS(curve_resolution_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(
        key,
        absorbance=dataset.absorbance / scale,
        mask=dataset.mask,
        n_wavelength=dataset.n_wavelength,
        n_species=n_species,
        tau=tau,
        sigma_scale=0.01,
        extra_fields=("diverging", "num_steps"),
    )
    return mcmc
