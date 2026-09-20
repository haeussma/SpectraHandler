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


def _absorbance_scale(dataset: SpectralDataset) -> float:
    """The factor absorbance is divided by to bring it to O(1).

    Private on purpose. A caller comparing fitted concentrations against known ones
    needs this number, but whether it belongs in the public API is still open -- see the
    concern recorded in the Task 2 report.
    """
    return float(jnp.nanmax(jnp.abs(dataset.absorbance)))


def fit(
    dataset: SpectralDataset,
    n_species: int,
    key: Array,
    *,
    tau: float = 0.01,
    sigma_scale: float = 0.01,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 2,
    max_tree_depth: int = 10,
) -> MCMC:
    """Sample the v0 curve resolution posterior with NUTS.

    Absorbance is divided by its largest absolute value before sampling. Because each
    spectrum is normalised to mean one, sampled ``concentrations`` are in those
    non-dimensional units, not in ``dataset.concentration_unit``.

    **These defaults do not converge** on the easy synthetic set: R-hat 4.4 and an ESS
    of 1, with the tree depth pinned at its maximum. They are the plan's development
    settings, kept because the one configuration that did converge costs about eight
    minutes. Check ``print_summary()`` before believing any width this returns, and see
    ``plans/001-bayesian-curve-resolution/spec.md`` section 7 for the full evidence and
    for which settings did converge.

    Args:
        dataset: Validated observations. ``v0`` requires ``mask.all()``.
        n_species: Number of components to resolve. Given, not inferred.
        key: PRNG key for the sampler.
        tau: Fixed random walk scale for the spectra, per wavelength channel. Sets both
            the curvature and the initial slope; see :func:`curve_resolution_model`.
        sigma_scale: Scale of the ``HalfNormal`` prior on the noise standard deviation,
            in the non-dimensionalised units absorbance is scaled to.
        num_warmup: Warmup iterations. 200 is the development setting.
        num_samples: Post-warmup draws per chain.
        num_chains: Chains; two is the minimum that lets R-hat mean anything.
        max_tree_depth: NUTS tree depth cap. At the default of 10 the sampler saturates
            it on this model; 14 is what converged, at roughly fifteen times the cost.

    Returns:
        The completed ``MCMC`` object. Call ``print_summary()`` for R-hat, ESS and the
        divergence count.

    Raises:
        NotImplementedError: If ``dataset.mask`` has any False entry. Ragged runs are
            a later step; the contract carries the machinery, v0 does not use it.
    """
    if not bool(dataset.mask.all()):
        raise NotImplementedError("v0 requires fully measured runs; masking is step 4+")

    kernel = NUTS(curve_resolution_model, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(
        key,
        absorbance=dataset.absorbance / _absorbance_scale(dataset),
        mask=dataset.mask,
        n_wavelength=dataset.n_wavelength,
        n_species=n_species,
        tau=tau,
        sigma_scale=sigma_scale,
        extra_fields=("diverging", "num_steps"),
    )
    return mcmc
