"""Credible-band curve resolution.

The data fix the product ``C @ S`` but not its split into ``C`` and ``S``. The work is done
in two stages that keep the two apart:

1. **What the data identify.** A NumPyro posterior of the rank-K product ``X @ Y`` and of one
   noise standard deviation per wavelength, with independent normal noise. ``X`` and ``Y``
   are held in a fixed gauge, so this posterior has one mode and is close to normal.
2. **What they do not.** Every split is a re-mixing ``C = X @ T``, ``S = inv(T) @ Y``.
   Closure and references pin some entries of ``T``; non-negativity bounds the rest to a
   feasible region. No prior is put on ``T``.

A re-mixing is feasible unless it makes some value negative beyond what the posterior of
stage 1 allows. The band is the range, over the whole feasible region, of the posterior
interval of every concentration and spectrum value. Flat draws over the region are a
labelled typical-solution summary, not the result. See ADR 0007.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.scipy.stats import norm
from numpyro.diagnostics import split_gelman_rubin
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, init_to_value

from spectrahandler.curve_resolution.background import (
    Background,
    align_baselines,
    background_shapes,
    fit_backgrounds,
)
from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.noise import NoiseCorrelation

__all__ = ["CredibleBand", "resolve_band"]

#: theta -> (concentrations, spectra, concentration posterior sd, spectra posterior sd).
type _Parts = Callable[[Array], tuple[Array, Array, Array, Array]]
#: theta -> summed non-negativity violation in posterior sd; zero means feasible.
type _Violation = Callable[[Array], Array]
type _State = tuple[Array, Array, Array, Array, Array]

#: Search and edge lengths are fractions of tau = |t_particular|, the scale of T. The step,
#: spread, floor, cap and determinant constants below are absolute values at the reference
#: scale tau = 16, and are rescaled by tau / 16.
_TAU_REFERENCE = 16.0

#: Above this split R-hat the posterior of the identified part is reported as not converged.
_R_HAT_WARN = 1.05


@dataclass(frozen=True, eq=False)
class CredibleBand:
    """Everything the data allow, for every concentration and spectrum value.

    Attributes:
        concentration_lower: Band lower edge, clipped at zero, shape
            ``(n_run, n_time, n_species)``, in ``concentration_unit``.
        concentration_upper: Band upper edge, same shape and unit.
        spectra_lower: Band lower edge, clipped at zero, shape
            ``(1, n_species, n_wavelength)``, in absorbance per ``concentration_unit``.
            The run axis has size 1: all runs are resolved with one set of spectra, and
            it broadcasts against per-run arrays.
        spectra_upper: Band upper edge, same shape and unit.
        concentration_ambiguity: Extremes of the posterior mean over the feasible region,
            shape ``(2, n_run, n_time, n_species)``: the rotational ambiguity without the
            noise part. The region tolerates values slightly below zero, so these extremes
            may dip below zero.
        spectra_ambiguity: Same, shape ``(2, 1, n_species, n_wavelength)``.
        concentration_draws: Flat draws over the region plus posterior noise, shape
            ``(n_draw, n_run, n_time, n_species)``. A typical-solution summary under an
            explicit flat prior on the free re-mixing numbers, not a calibrated interval.
        spectra_draws: Same, shape ``(n_draw, 1, n_species, n_wavelength)``.
        sigma: Posterior mean of the noise standard deviation per wavelength, shape
            ``(n_wavelength,)``, in absorbance units.
        sigma_draws: Posterior draws of the same, shape ``(n_sample, n_wavelength)``. One
            repeated row if ``sigma`` was given.
        noise_exponent_draws: Posterior draws of the growth of the noise with absorbance,
            shape ``(n_sample,)``; one repeated value if ``noise_exponent`` was given. With
            a non-zero exponent ``sigma`` is the noise at each wavelength's mean absorbance.
        level: Credibility level of the band.
        n_free: Free re-mixing numbers left after closure and references.
        r_hat: Largest split R-hat over the identified part; close to 1 when the chains
            agree.
        background_names: Names of the backgrounds that were fitted and subtracted.
        background_amplitude: The amplitudes removed from each run, shape
            ``(n_run, n_background)``, in absorbance units: the common part from the zero
            windows plus the posterior mean of each run's part relative to the first run.
        background_draws: Posterior draws of each run's amplitudes relative to the first
            run, around the starting alignment, shape ``(n_sample, n_run, n_background)``,
            in absorbance units; zero for the first run.
        baseline_tolerance: By how much zero absorbance may be off at each wavelength, shape
            ``(n_wavelength,)``, in absorbance units. Spectra may dip below zero by this
            much per smallest run total, and the spectrum band is widened by the same
            amount. Outside the zero windows it grows with the distance a tilt is
            extrapolated over.
        zero_source: Where zero absorbance came from, in words.
    """

    concentration_lower: Array
    concentration_upper: Array
    spectra_lower: Array
    spectra_upper: Array
    concentration_ambiguity: Array
    spectra_ambiguity: Array
    concentration_draws: Array
    spectra_draws: Array
    sigma: Array
    sigma_draws: Array
    noise_exponent_draws: Array
    level: float
    n_free: int
    r_hat: float
    background_names: tuple[str, ...]
    background_amplitude: Array
    background_draws: Array
    baseline_tolerance: Array
    zero_source: str


class _Identified(NamedTuple):
    """Posterior draws of what the data identify, chains merged."""

    x: Array  # (n_sample, n_row, n_species): row coordinates of the time course
    x_ref: Array  # (n_sample, n_ref, n_species): row coordinates of the references
    y: Array  # (n_sample, n_species, n_wavelength): spectral patterns
    sigma: Array  # (n_sample, n_wavelength): noise sd per wavelength
    exponent: Array  # (n_sample,): growth of the noise with absorbance
    baseline: Array  # (n_sample, n_run, n_background): amplitudes relative to the first run
    r_hat: float


def _sample_identified(
    time_course: Array,
    ref_rows: Array,
    ref_sd: Array,
    k: int,
    key: Array,
    *,
    sigma: Array | None,
    noise_scale: float,
    power: Array | None,
    exponent: float | None,
    absorbance_scale: float,
    shapes: Array,
    n_run: int,
    reference_baseline: bool,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
) -> _Identified:
    """Posterior of the rank-``k`` product and the noise per wavelength, in a fixed gauge.

    The model is ``time_course ~ Normal(X @ Y, noise)`` with one ``sigma`` per wavelength;
    each reference row is ``Normal(x_ref @ Y, ref_sd)``. The gauge is
    ``Y = (y0 + B @ y_perp) * sigma0``: ``y0`` holds the leading right singular vectors
    of the noise-whitened data, ``y_perp`` their complement. Every rank-``k`` matrix near the
    data has exactly one such factorisation, so ``X``, ``B`` and ``sigma`` are identified.
    Priors: flat on ``X``, ``B`` and ``x_ref``; half-normal on ``sigma``; normal with mean
    0.5 and sd 0.5 on the exponent when it is sampled.

    The noise standard deviation of one point is ``sigma * 10 ** (exponent * (A - A_mean))``,
    with ``A`` the modelled absorbance and ``A_mean`` the mean absorbance at that wavelength,
    so ``sigma`` is the noise at the average signal. Noise is independent between spectra.
    Along wavelength it is independent if ``power`` is ``None``; otherwise each scaled
    residual spectrum has the power spectrum ``power``, evaluated on a circular grid (the
    Whittle likelihood).

    Every run after the first gets one amplitude per background, with a flat prior: its
    baseline relative to the first run. With ``reference_baseline`` every reference gets
    such amplitudes too. The runs share ``Y``, so a
    difference between runs that is constant in time and has a background's shape is
    identified.

    Args:
        time_course: Data, shape ``(n_row, n_wavelength)``.
        ref_rows: Reference spectra, shape ``(n_ref, n_wavelength)``; ``n_ref`` may be 0.
        ref_sd: Standard deviation of each reference value, shape
            ``(n_ref, n_wavelength)``.
        k: Rank, the number of species.
        key: PRNG key for NUTS.
        sigma: Known noise sd per wavelength, shape ``(n_wavelength,)``; ``None`` samples it.
        noise_scale: Scale of the half-normal prior on ``sigma``.
        power: Power spectrum of unit-variance noise along wavelength, shape
            ``(n_wavelength,)`` in ``jnp.fft.fft`` order with mean 1; ``None`` for
            independent noise.
        exponent: Known growth of the noise with absorbance; ``None`` samples it.
        absorbance_scale: Absorbance units per unit of ``time_course``, so that the
            exponent refers to absorbance.
        shapes: Background shapes, shape ``(n_wavelength, n_background)``; may have no
            column.
        n_run: Number of runs; the rows of ``time_course`` are run after run, each with
            the same number of spectra.
        reference_baseline: Give every reference background amplitudes of its own.
        num_warmup: NUTS warmup steps per chain.
        num_samples: NUTS draws per chain.
        num_chains: Chains, run vectorised.
        target_accept_prob: NUTS target acceptance probability.

    Returns:
        The draws and the largest split R-hat over the sampled sites.
    """
    n_row, n_w = time_course.shape
    n_ref = ref_rows.shape[0]
    sigma0 = jnp.ones(n_w) if sigma is None else sigma
    # For a given noise the noise-scaled SVD is the maximum likelihood fit, and its residual
    # gives the noise: alternate a few times for a start and a well-scaled gauge.
    for _ in range(3 if sigma is None else 0):
        u, s, vt = jnp.linalg.svd(time_course / sigma0, full_matrices=False)
        resid = time_course - ((u[:, :k] * s[:k]) @ vt[:k]) * sigma0
        sigma0 = jnp.sqrt((resid**2).sum(axis=0) / (n_row - k))
    # The gauge is set up in coordinates where the noise is white along wavelength, so that
    # the latents are equally well determined and NUTS mixes as with independent noise.
    whiten = unwhiten = jnp.eye(n_w)
    if power is not None:
        basis = jnp.fft.fft(jnp.eye(n_w), axis=1)
        whiten = jnp.fft.ifft(basis / jnp.sqrt(power), axis=1).real
        unwhiten = jnp.fft.ifft(basis * jnp.sqrt(power), axis=1).real
    _, s, vt = jnp.linalg.svd((time_course / sigma0) @ whiten, full_matrices=True)
    y0, y_perp, b_scale = vt[:k] @ unwhiten, vt[k:] @ unwhiten, 1.0 / s[:k, None]
    to_x = whiten @ vt[:k].T
    level = time_course.mean(axis=0)
    n_time, n_background = n_row // n_run, shapes.shape[1]
    relative = n_run > 1 and n_background > 0
    # Roughly the posterior sd of one amplitude, so that the latent has unit scale.
    baseline_scale = 1.0 / jnp.sqrt(n_time * ((shapes / sigma0[:, None]) ** 2).sum(axis=0))
    ref_scale = 1.0 / jnp.sqrt(((shapes[None] / ref_sd[:, :, None]) ** 2).sum(axis=1))

    def model() -> None:
        x = jnp.asarray(numpyro.sample("x", dist.ImproperUniform(constraints.real, (), (n_row, k))))
        b = jnp.asarray(
            numpyro.sample("b", dist.ImproperUniform(constraints.real, (), (k, n_w - k)))
        )
        # b is scaled by the singular values so that every latent has a posterior sd near 1.
        y = jnp.asarray(numpyro.deterministic("y", (y0 + (b_scale * b) @ y_perp) * sigma0))
        noise = (
            sigma
            if sigma is not None
            else numpyro.sample("sigma", dist.HalfNormal(noise_scale).expand([n_w]).to_event(1))
        )
        mean = x @ y
        if relative:
            shift = jnp.asarray(
                numpyro.sample(
                    "baseline",
                    dist.ImproperUniform(constraints.real, (), (n_run - 1, n_background)),
                )
            )
            amplitude = jnp.concatenate([jnp.zeros((1, n_background)), baseline_scale * shift])
            mean = mean + jnp.repeat(amplitude @ shapes.T, n_time, axis=0)
        if exponent != 0.0:
            gamma = (
                exponent
                if exponent is not None
                else numpyro.sample("exponent", dist.Normal(0.5, 0.5))
            )
            noise = noise * 10.0 ** (gamma * absorbance_scale * (mean - level))
        if power is None:
            numpyro.sample("absorbance", dist.Normal(mean, noise).to_event(2), obs=time_course)
        else:
            # Whittle: the scaled residual spectra are independent across frequency, each
            # frequency with variance `power`. Terms that do not depend on a latent are left out.
            scaled = (time_course - mean) / noise
            energy = (jnp.abs(jnp.fft.fft(scaled, axis=-1)) ** 2 / power).sum() / n_w
            log_scale = jnp.log(jnp.broadcast_to(noise, mean.shape)).sum()
            numpyro.factor("absorbance", -0.5 * energy - log_scale)
        if n_ref:
            x_ref = jnp.asarray(
                numpyro.sample("x_ref", dist.ImproperUniform(constraints.real, (), (n_ref, k)))
            )
            ref_mean = x_ref @ y
            if reference_baseline and n_background:
                # A reference is a measurement with a baseline of its own.
                ref_shift = jnp.asarray(
                    numpyro.sample(
                        "reference_baseline",
                        dist.ImproperUniform(constraints.real, (), (n_ref, n_background)),
                    )
                )
                ref_mean = ref_mean + (ref_scale * ref_shift) @ shapes.T
            numpyro.sample("reference", dist.Normal(ref_mean, ref_sd).to_event(2), obs=ref_rows)

    init = {
        "x": (time_course / sigma0) @ to_x,
        "b": jnp.zeros((k, n_w - k)),
        "sigma": sigma0,
        "x_ref": (ref_rows / sigma0) @ to_x,
        "exponent": jnp.asarray(0.5),
        "baseline": jnp.zeros((n_run - 1, n_background)),
        "reference_baseline": jnp.zeros((n_ref, n_background)),
    }
    kernel = NUTS(
        model, init_strategy=init_to_value(values=init), target_accept_prob=target_accept_prob
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(key)
    chains = mcmc.get_samples(group_by_chain=True)
    r_hat = max(
        float(split_gelman_rubin(chains[site]).max())
        for site in ("x", "b", "sigma", "exponent", "baseline", "reference_baseline")
        if site in chains
    )
    draws = mcmc.get_samples()
    n_sample = num_chains * num_samples
    return _Identified(
        x=draws["x"],
        x_ref=draws["x_ref"] if n_ref else jnp.zeros((n_sample, 0, k)),
        y=draws["y"],
        sigma=draws["sigma"] if sigma is None else jnp.tile(sigma, (n_sample, 1)),
        exponent=draws["exponent"] if exponent is None else jnp.full(n_sample, exponent),
        baseline=jnp.concatenate(
            [
                jnp.zeros((n_sample, 1, n_background)),
                baseline_scale * draws["baseline"]
                if relative
                else jnp.zeros((n_sample, n_run - 1, n_background)),
            ],
            axis=1,
        ),
        r_hat=r_hat,
    )


def resolve_band(
    dataset: SpectralDataset,
    key: Array,
    *,
    level: float = 0.95,
    sigma: float | Array | None = None,
    noise_scale: float | None = None,
    noise_correlation: NoiseCorrelation | None = None,
    noise_exponent: float | None = 0.0,
    backgrounds: tuple[Background, ...] | None = None,
    zero_windows: tuple[tuple[float, float], ...] = (),
    baseline_tolerance: float | Array | None = None,
    baseline_smoothing: int = 20,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    target_accept_prob: float = 0.8,
    n_iter: int = 8000,
    n_burn: int = 500,
    thin: int = 2,
    n_als: int = 200,
    n_restart: int = 8,
    n_search: int = 30000,
) -> CredibleBand:
    """Resolve ``dataset`` into the credible band of every spectra/profile split it allows.

    The noise is independent between all points, normal, with one standard deviation per
    wavelength. Closure: in every run, species sum to ``initial_state[run].sum()`` at every
    time. Every species with a finite ``reference_spectra`` row is pinned by it; species
    without one are resolved only up to relabelling among themselves. The work is done
    with concentrations divided by the mean run total and absorbance by its RMS, so every
    tolerance acts on a problem of unit scale and the result does not depend on the units;
    every output is in the dataset's units.

    Two normal quantiles follow from ``level``. A re-mixing is feasible unless some value
    is below zero by more than ``z_region`` posterior sd, with ``z_region`` chosen so that
    the true split passes all ``n_value`` tests together with probability ``level``:
    ``level ** (1 / n_value)`` per test. The band then takes, over the feasible region, the
    central ``level`` posterior interval of every value.

    Baselines come in two parts. How the runs' baselines differ from the first run's is
    inferred in the fit, one amplitude per background and run: the runs share their
    spectra, so such a difference cannot be chemistry. What all runs have in common
    cannot be told from the species spectra, so it is fitted inside ``zero_windows`` and
    subtracted (:func:`fit_backgrounds`). What is left in the windows is the baseline
    tolerance; outside the windows it is multiplied by one plus the leverage of the fit
    there. Every spectrum may dip below zero by the tolerance divided by the smallest
    run total, and the spectrum band is widened by the same amount on both sides, because
    a baseline error that is constant in time ends up in the species spectra.

    Without zero windows the data and the reference spectra are all taken to share the
    instrument's zero. With zero windows the data's zero is the windows', and every
    reference gets background amplitudes of its own, so a reference need not be
    baseline-corrected beforehand.

    Args:
        dataset: Validated observations; ``mask`` must be all True.
        key: PRNG key for NUTS, the start search, the hit-and-run chain and the noise draws.
        level: Credibility level of the band, between 0 and 1.
        sigma: Known noise standard deviation in absorbance units, one number or one per
            wavelength. ``None`` infers one per wavelength.
        noise_scale: Scale of the half-normal prior on each wavelength's noise standard
            deviation, in absorbance units. ``None`` takes the RMS absorbance, which is weak.
        noise_correlation: How the noise of neighbouring channels is correlated, from
            :func:`measure_noise_correlation` on data with the same channel spacing.
            ``None`` takes the noise as independent between channels.
        noise_exponent: Growth of the noise with absorbance: the noise standard deviation
            is multiplied by ``10 ** (noise_exponent * A)``. 0 is noise that does not depend
            on the signal, 0.5 is shot noise. ``None`` infers it, with a normal prior of
            mean 0.5 and sd 0.5.
        backgrounds: Known spectral shapes of the baseline. ``None`` takes an offset and a
            tilt; ``()`` assumes that no run has a baseline. With several runs, each run
            gets one amplitude per shape relative to the first run.
        zero_windows: Wavelength intervals ``(low, high)``, in the dataset's unit, where no
            chemical species absorbs: they fix the baseline that all runs share. An
            assumption; the measured tolerance tells how well it holds. Without them the
            common zero is the first run's instrument zero.
        baseline_tolerance: By how much zero absorbance may be off, in absorbance units.
            One number is the tolerance inside the zero windows, carried outwards by the
            fit's leverage (a worst case), or the tolerance everywhere if there are no
            windows. One number per wavelength is used as given, e.g. measured on a blank
            that went through :func:`fit_backgrounds`. ``None`` takes the value measured in
            the windows, or 0 without windows. A value below the measured one is an error.
        baseline_smoothing: Number of consecutive spectra averaged when the tolerance is
            measured.
        num_warmup: NUTS warmup steps per chain.
        num_samples: NUTS draws per chain.
        num_chains: NUTS chains.
        target_accept_prob: NUTS target acceptance probability.
        n_iter: Hit-and-run iterations.
        n_burn: Iterations discarded before draws are kept.
        thin: Keep every ``thin``-th draw after burn-in.
        n_als: MCR-ALS sweeps in the reduced space that produce the starting split.
        n_restart: Parallel restarts of the feasible-start search.
        n_search: Maximum steps per restart of the feasible-start search.

    Returns:
        The band, its ambiguity part, flat draws and the noise posterior.

    Raises:
        RuntimeError: If JAX float64 is not enabled.
        NotImplementedError: If ``dataset.mask`` has any False entry.
        ValueError: If an argument is out of range, if ``zero_windows`` are given without
            any background, for the errors of :func:`fit_backgrounds`, if
            ``baseline_tolerance`` is below the measured one, if there are not more spectra
            and wavelengths than species, if the mean run total or the absorbance RMS is not
            finite and positive, or if no feasible split is found: the rank, closure or
            references contradict the data, or the search failed numerically.

    Warns:
        UserWarning: If the chains of the identified part disagree (split R-hat above 1.05).
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "resolve_band needs float64: call jax.config.update('jax_enable_x64', True) "
            "before creating any arrays. spectrahandler never enables it itself."
        )
    if sigma is not None and not bool((jnp.isfinite(sigma) & (jnp.asarray(sigma) > 0)).all()):
        raise ValueError(f"sigma must be finite and positive, got {sigma}")
    for name, value, ok, rule in [
        ("level", level, 0 < level < 1, "> 0 and < 1"),
        ("noise_scale", noise_scale, noise_scale is None or noise_scale > 0, "> 0"),
        (
            "baseline_tolerance",
            baseline_tolerance,
            baseline_tolerance is None or bool((jnp.asarray(baseline_tolerance) >= 0).all()),
            ">= 0",
        ),
        ("num_warmup", num_warmup, num_warmup >= 1, ">= 1"),
        ("num_samples", num_samples, num_samples >= 4, ">= 4"),
        ("num_chains", num_chains, num_chains >= 1, ">= 1"),
        ("n_burn", n_burn, 0 <= n_burn < n_iter, f">= 0 and < n_iter = {n_iter}"),
        ("thin", thin, thin >= 1, ">= 1"),
        ("n_restart", n_restart, n_restart >= 1, ">= 1"),
        ("n_als", n_als, n_als >= 0, ">= 0"),
        ("n_search", n_search, n_search >= 1, ">= 1"),
    ]:
        if not ok:
            raise ValueError(f"{name} must be {rule}, got {value}")
    if not bool(dataset.mask.all()):
        raise NotImplementedError("resolve_band requires fully measured runs")
    if backgrounds is None:
        backgrounds = (Background.offset(), Background.tilt())
    if zero_windows and not backgrounds:
        raise ValueError("zero_windows need at least one background to fit there")
    shapes = background_shapes(backgrounds, dataset.wavelength)
    # The runs' baselines relative to the first: a start here, sampled in the fit.
    amplitude = jnp.zeros((dataset.n_run, shapes.shape[1]))
    if dataset.n_run > 1 and backgrounds:
        amplitude = align_baselines(dataset.absorbance, shapes, dataset.n_species)
        dataset = replace(
            dataset, absorbance=dataset.absorbance - (amplitude @ shapes.T)[:, None, :]
        )
    given = None if baseline_tolerance is None else jnp.asarray(baseline_tolerance)
    delta = jnp.broadcast_to(0.0 if given is None else given, (dataset.n_wavelength,))
    zero_source = "no zero window: zero is the instrument zero of the first run (assumed)"
    if zero_windows:
        fit = fit_backgrounds(
            dataset.absorbance,
            dataset.wavelength,
            backgrounds=backgrounds,
            zero_windows=zero_windows,
            sigma=sigma,
            smoothing=baseline_smoothing,
            shared=True,
        )
        if given is not None and fit.tolerance > float(given.min()):
            run, where = fit.tolerance_at
            raise ValueError(
                f"baseline_tolerance {float(given.min()):g} is below the {fit.tolerance:.3g} "
                f"measured in the zero windows (run {dataset.run_ids[run]!r}, "
                f"{where:g} {dataset.wavelength_unit})"
            )
        if given is None or given.ndim == 0:
            # One number is the tolerance inside the windows; the fit's leverage carries it out.
            delta = (fit.tolerance if given is None else given) * fit.tolerance_factor
        dataset = replace(dataset, absorbance=fit.corrected)
        amplitude = amplitude + fit.amplitude
        windows = ", ".join(f"{low:g}-{high:g}" for low, high in zero_windows)
        zero_source = (
            f"zero windows {windows} {dataset.wavelength_unit}, tolerance "
            f"{'measured' if baseline_tolerance is None else 'given'}"
        )
    dataset, c_scale, a_scale = _to_unit_scale(dataset)
    s_scale = a_scale / c_scale
    k = dataset.n_species
    n_w = dataset.n_wavelength
    n_row = dataset.n_run * dataset.n_time
    if min(n_row, n_w) <= k:
        raise ValueError(f"need more spectra ({n_row}) and wavelengths ({n_w}) than species ({k})")
    time_course = dataset.absorbance.reshape(n_row, n_w)
    has_ref = jnp.isfinite(dataset.reference_spectra).all(axis=1)
    ref_idx = jnp.flatnonzero(has_ref)
    ref_comp = jnp.eye(k)[ref_idx]

    totals = jnp.repeat(dataset.initial_state.sum(axis=1), dataset.n_time)
    # A baseline error up to delta, constant in time, sits in the spectra as delta / total.
    tol_s = delta / a_scale / totals.min()

    k_post, k_search, k_chain, k_noise = jax.random.split(key, 4)
    post = _sample_identified(
        time_course,
        dataset.reference_spectra[ref_idx],
        # Beyond its own baseline, a reference may disagree with the data by the pattern
        # that no background fits: the tolerance inside the zero windows.
        jnp.sqrt(dataset.reference_sigma[ref_idx, None] ** 2 + tol_s.min() ** 2) * jnp.ones(n_w),
        k,
        k_post,
        sigma=None if sigma is None else jnp.broadcast_to(jnp.asarray(sigma) / a_scale, (n_w,)),
        noise_scale=1.0 if noise_scale is None else noise_scale / a_scale,
        power=None if noise_correlation is None else noise_correlation.spectrum(n_w),
        exponent=noise_exponent,
        absorbance_scale=a_scale,
        shapes=shapes,
        n_run=dataset.n_run,
        # A zero window moves the data's zero, so a reference need not share it any more.
        reference_baseline=bool(zero_windows),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
    )
    if post.r_hat > _R_HAT_WARN:
        warnings.warn(
            f"the chains of the identified part disagree (split R-hat {post.r_hat:.2f}); "
            "raise num_warmup or num_samples",
            stacklevel=2,
        )
    # Posterior mean and, per row of X and per column of Y, the K x K covariance: all that a
    # single concentration or spectrum value needs under any re-mixing.
    x_time, y = post.x.mean(axis=0), post.y.mean(axis=0)
    x = jnp.concatenate([x_time, post.x_ref.mean(axis=0)])
    cov_x = jax.vmap(lambda draws: jnp.cov(draws.T), in_axes=1)(post.x)
    cov_y = jax.vmap(lambda draws: jnp.cov(draws.T), in_axes=2)(post.y)
    # A reference pins its species' spectrum to x_ref @ Y whatever the re-mixing, so that
    # spectrum's spread is the spread of this product, which includes the uncertainty of
    # x_ref and with it of the reference's own baseline.
    sd_ref = jnp.einsum("dnk,dkw->dnw", post.x_ref, post.y).std(axis=0)
    y_pinv = jnp.linalg.pinv(y)
    z_region = float(norm.ppf(level ** (1.0 / (n_row * k + k * n_w))))
    z_margin = float(norm.ppf(0.5 + level / 2))

    # Equalities on vec(T), row-major. Closure: X @ T @ 1 = row totals, so T @ 1 = t_star.
    # References: x_ref @ T = the unit composition of the reference's species.
    row_sums = jnp.concatenate([totals, ref_comp.sum(axis=1)])
    t_star = jnp.linalg.lstsq(x, row_sums)[0]
    eq = jnp.concatenate([jnp.kron(jnp.eye(k), jnp.ones(k)), jnp.kron(x[n_row:], jnp.eye(k))])
    rhs = jnp.concatenate([t_star, ref_comp.ravel()])
    t_particular = jnp.linalg.lstsq(eq, rhs)[0]
    _, eq_sv, eq_vt = jnp.linalg.svd(eq)
    rank = int((eq_sv > 1e-8 * eq_sv[0]).sum())
    null = eq_vt[rank:]
    tau = jnp.linalg.norm(t_particular)

    def parts(theta: Array) -> tuple[Array, Array, Array, Array]:
        t = (t_particular + theta @ null).reshape(k, k)
        r = jnp.linalg.inv(t)
        c: Array = x_time @ t
        sp: Array = r @ y
        sd_c: Array = jnp.sqrt(jnp.einsum("ak,iab,bk->ik", t, cov_x, t))
        sd_s: Array = jnp.sqrt(jnp.einsum("ka,wab,kb->kw", r, cov_y, r))
        return c, sp, sd_c, sd_s.at[ref_idx].set(sd_ref)

    def violation(theta: Array) -> Array:
        t = (t_particular + theta @ null).reshape(k, k)
        c, sp, sd_c, sd_s = parts(theta)
        v = (jnp.clip(-(c + z_region * sd_c), 0.0) / sd_c).sum() + (
            jnp.clip(-(sp + z_region * sd_s + tol_s), 0.0) / sd_s
        ).sum()
        return jnp.where(
            jnp.abs(jnp.linalg.det(t)) < 1e-12 * (tau / _TAU_REFERENCE) ** k, jnp.inf, v
        )

    def to_theta(t: Array) -> Array:
        """The nearest split satisfying closure and references, in free coordinates."""
        theta: Array = (t.ravel() - t_particular) @ null.T
        return theta

    def alternate(_: int, theta: Array) -> Array:
        """One MCR-ALS sweep in the reduced space: clip C and refit, clip S and refit."""
        t = (t_particular + theta @ null).reshape(k, k)
        t = jnp.linalg.lstsq(x_time, jnp.clip(x_time @ t, 0.0))[0]
        t = (t_particular + to_theta(t) @ null).reshape(k, k)
        t = jnp.linalg.inv(jnp.clip(jnp.linalg.inv(t) @ y, 0.0) @ y_pinv)
        return to_theta(t)

    guess = _initial_guess(x, x_time, totals, ref_comp, ref_idx, t_particular, null, k)
    theta0 = jax.lax.fori_loop(0, n_als, alternate, guess)
    theta0 = jnp.where(jnp.isfinite(theta0).all(), theta0, guess)
    theta, v = _find_feasible(violation, theta0, tau, k_search, n_restart, n_search)
    if not math.isfinite(float(v)):
        raise ValueError(
            f"no feasible split found (violation {float(v)}): numerical failure in the "
            "search, not evidence that the data contradict the model"
        )
    if not float(v) == 0.0:
        raise ValueError(
            f"no feasible split found (violation {float(v):.3g}): the rank, closure or "
            "references contradict the data"
        )

    lo, hi, draws = _hit_and_run(parts, violation, theta, tau, k_chain, n_iter, z_margin)
    c, sp, sd_c, sd_s = (d[n_burn::thin] for d in draws)
    shape = (dataset.n_run, dataset.n_time, k)
    n_c = n_row * k

    def split(values: Array) -> tuple[Array, Array]:
        """All values as concentrations and spectra, in the dataset's units."""
        return c_scale * values[:n_c].reshape(shape), s_scale * values[n_c:].reshape(1, k, n_w)

    (c_mean_lo, s_mean_lo), (c_lo, s_lo), _ = (split(row) for row in lo)
    (c_mean_hi, s_mean_hi), _, (c_hi, s_hi) = (split(row) for row in hi)
    kc, ks = jax.random.split(k_noise)
    sigma_draws = post.sigma * a_scale
    return CredibleBand(
        concentration_lower=jnp.clip(c_lo, 0.0),
        concentration_upper=c_hi,
        spectra_lower=jnp.clip(s_lo - s_scale * tol_s, 0.0),
        spectra_upper=s_hi + s_scale * tol_s,
        concentration_ambiguity=jnp.stack([c_mean_lo, c_mean_hi]),
        spectra_ambiguity=jnp.stack([s_mean_lo, s_mean_hi]),
        concentration_draws=c_scale
        * (c + sd_c * jax.random.normal(kc, c.shape)).reshape(-1, *shape),
        spectra_draws=s_scale * (sp + sd_s * jax.random.normal(ks, sp.shape))[:, None],
        sigma=sigma_draws.mean(axis=0),
        sigma_draws=sigma_draws,
        noise_exponent_draws=post.exponent,
        level=level,
        n_free=int(null.shape[0]),
        r_hat=post.r_hat,
        background_names=tuple(b.name for b in backgrounds),
        background_amplitude=amplitude + a_scale * post.baseline.mean(axis=0),
        background_draws=a_scale * post.baseline,
        baseline_tolerance=delta,
        zero_source=zero_source,
    )


def _to_unit_scale(dataset: SpectralDataset) -> tuple[SpectralDataset, float, float]:
    """The dataset in units where the mean run total and the absorbance RMS are 1.

    Args:
        dataset: Validated observations, fully measured.

    Returns:
        The rescaled dataset, ``c_scale`` and ``a_scale``: concentrations were divided by
        ``c_scale``, absorbance by ``a_scale``, so spectra by ``a_scale / c_scale``.

    Raises:
        ValueError: If either scale is not finite and positive.
    """
    c_scale = float(dataset.initial_state.sum(axis=1).mean())
    a_scale = float(jnp.sqrt((dataset.absorbance**2).mean()))
    if not (math.isfinite(c_scale) and c_scale > 0):
        raise ValueError(f"mean run total of initial_state must be positive, got {c_scale}")
    if not (math.isfinite(a_scale) and a_scale > 0):
        raise ValueError(f"absorbance RMS must be positive, got {a_scale}")
    s_scale = a_scale / c_scale
    scaled = replace(
        dataset,
        absorbance=dataset.absorbance / a_scale,
        initial_state=dataset.initial_state / c_scale,
        reference_spectra=dataset.reference_spectra / s_scale,
        reference_sigma=dataset.reference_sigma / s_scale,
    )
    return scaled, c_scale, a_scale


def _initial_guess(
    x: Array,
    x_time: Array,
    totals: Array,
    ref_comp: Array,
    ref_idx: Array,
    t_particular: Array,
    null: Array,
    k: int,
) -> Array:
    """A starting split: the most distinct rows taken as pure species.

    Reference rows are pure by construction; the remaining species are assigned, in
    order, to the time rows least explained by the rows picked so far. The result is
    projected onto the equality constraints.
    """
    picked = x[x_time.shape[0] :]
    rows, comps = [picked], [ref_comp]
    free_species = [i for i in range(k) if i not in {int(j) for j in ref_idx}]
    # Greedy farthest-row picking; the Python loop runs over species (K steps), not data.
    for species in free_species:
        basis = jnp.concatenate(rows)
        resid = x_time - x_time @ jnp.linalg.pinv(basis) @ basis if basis.shape[0] else x_time
        row = int(jnp.argmax(jnp.linalg.norm(resid, axis=1)))
        rows.append(x_time[row : row + 1])
        comps.append(totals[row] * jnp.eye(k)[species][None, :])
    t0 = jnp.linalg.solve(jnp.concatenate(rows), jnp.concatenate(comps))
    theta0: Array = (t0.ravel() - t_particular) @ null.T
    return theta0


def _find_feasible(
    violation: _Violation,
    theta0: Array,
    tau: Array,
    key: Array,
    n_restart: int,
    n_search: int,
) -> tuple[Array, Array]:
    """Adaptive random search for a point with zero violation, restarts in parallel.

    Step lengths are in units of ``tau``, the scale of the re-mixing matrix.
    """
    unit = tau / _TAU_REFERENCE

    def search(start: Array, key: Array) -> tuple[Array, Array]:
        def cond(state: _State) -> Array:
            _, v, _, i, _ = state
            return (v > 0) & (i < n_search)

        def body(state: _State) -> _State:
            theta, v, step, i, key = state
            key, sub = jax.random.split(key)
            cand = theta + step * jax.random.normal(sub, theta.shape)
            vc = violation(cand)
            better = vc < v
            out: _State = (
                jnp.where(better, cand, theta),
                jnp.where(better, vc, v),
                jnp.where(better, step * 1.3, jnp.maximum(step * 0.97, 1e-5 * unit)),
                i + 1,
                key,
            )
            return out

        init = (start, violation(start), 3.0 * unit, jnp.asarray(0), key)
        theta: Array
        v: Array
        theta, v, _, _, _ = jax.lax.while_loop(cond, body, init)
        return theta, v

    k_start, k_search = jax.random.split(key)
    spread = 3.0 * unit * jnp.arange(n_restart)[:, None]
    starts = theta0 + spread * jax.random.normal(k_start, (n_restart, theta0.size))
    thetas, vs = jax.jit(jax.vmap(search))(starts, jax.random.split(k_search, n_restart))
    best = jnp.argmin(vs)
    best_theta: Array = thetas[best]
    best_v: Array = vs[best]
    return best_theta, best_v


def _shrink(
    violation: _Violation,
    theta: Array,
    u: Array,
    lower: Array,
    upper: Array,
    key: Array,
    max_tries: int = 50,
) -> Array:
    """A uniform feasible position on ``[lower, upper]`` along ``u``, by shrinkage.

    The region is not convex: a line through it can cross a gap, so a uniform position on
    the bracket can be infeasible. An infeasible proposal shrinks the bracket towards the
    current point (Neal 2003, slice sampling) and is redrawn. Without this a chain that
    lands in a gap stays there, because every bracket from an infeasible point is empty.
    """

    def cond(state: _State) -> Array:
        _, _, pos, i, _ = state
        return (violation(theta + pos * u) > 0) & (i < max_tries)

    def body(state: _State) -> _State:
        lo, hi, pos, i, key = state
        lo, hi = jnp.where(pos < 0, pos, lo), jnp.where(pos > 0, pos, hi)
        key, sub = jax.random.split(key)
        out: _State = (lo, hi, jax.random.uniform(sub, minval=lo, maxval=hi), i + 1, key)
        return out

    key, sub = jax.random.split(key)
    first = jax.random.uniform(sub, minval=lower, maxval=upper)
    _, _, pos, i, _ = jax.lax.while_loop(cond, body, (lower, upper, first, jnp.asarray(0), key))
    accepted: Array = jnp.where(i < max_tries, pos, 0.0)
    return accepted


def _hit_and_run(
    parts: _Parts,
    violation: _Violation,
    theta: Array,
    tau: Array,
    key: Array,
    n_iter: int,
    z_margin: float,
) -> tuple[Array, Array, tuple[Array, Array, Array, Array]]:
    """Uniform draws over the region, plus the extremes of every value over the region.

    The extremes have shape ``(3, n_value)``: of the posterior mean, of the interval's lower
    edge ``mean - z_margin * sd`` and of its upper edge ``mean + z_margin * sd``. Edge
    searches start and stop at fixed multiples of ``tau``, the scale of the re-mixing matrix.
    """
    unit = tau / _TAU_REFERENCE

    def edge(theta: Array, u: Array) -> Array:
        """Distance along ``u`` to the region's boundary: double, then bisect."""
        t = jax.lax.while_loop(
            lambda t: (violation(theta + t * u) == 0) & (t < 1e4 * unit),
            lambda t: 2 * t,
            1e-3 * unit,
        )

        def bisect(_: int, ab: tuple[Array, Array]) -> tuple[Array, Array]:
            a, b = ab
            mid = 0.5 * (a + b)
            inside = violation(theta + mid * u) == 0
            return jnp.where(inside, mid, a), jnp.where(inside, b, mid)

        distance: Array = jax.lax.fori_loop(0, 25, bisect, (jnp.asarray(0.0), t))[0]
        return distance

    def flat(theta: Array) -> Array:
        c, sp, sd_c, sd_s = parts(theta)
        mean = jnp.concatenate([c.ravel(), sp.ravel()])
        margin = z_margin * jnp.concatenate([sd_c.ravel(), sd_s.ravel()])
        return jnp.stack([mean, mean - margin, mean + margin])

    def step(
        carry: tuple[Array, Array, Array], key: Array
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array]]:
        theta, lo, hi = carry
        k_dir, k_pos = jax.random.split(key)
        u = jax.random.normal(k_dir, theta.shape)
        u = u / jnp.linalg.norm(u)
        t_plus, t_minus = edge(theta, u), edge(theta, -u)
        ends = jnp.stack([flat(theta + t_plus * u), flat(theta - t_minus * u)])
        lo, hi = jnp.minimum(lo, ends.min(axis=0)), jnp.maximum(hi, ends.max(axis=0))
        moved: Array = theta + _shrink(violation, theta, u, -t_minus, t_plus, k_pos) * u
        return (moved, lo, hi), parts(moved)

    start = flat(theta)
    final, outputs = jax.jit(lambda c, ks: jax.lax.scan(step, c, ks))(
        (theta, start, start), jax.random.split(key, n_iter)
    )
    lo: Array = final[1]
    hi: Array = final[2]
    draws: tuple[Array, Array, Array, Array] = tuple(outputs)
    return lo, hi, draws
