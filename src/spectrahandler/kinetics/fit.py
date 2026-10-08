"""Maximum likelihood per run for a first-order scheme.

Per run ``r``: ``D_r(t, λ) = Σ_s c_rs(t) S_rs(λ) + ε``, with ``c_r(t)`` from the scheme,
spectra free, and independent Gaussian noise with its own level per wavelength. Given
the rates, the spectra are the least-squares solution and each wavelength's noise
variance is ``RSS_λ / n``; both are solved exactly, leaving the profile log likelihood
``-n/2 Σ_λ [log(2π RSS_λ / n) + 1]`` over the log rates.

Runs are fitted independently. What runs share is decided afterwards, by condition, in
:func:`spectrahandler.kinetics.summary.summarize`.

Two things the data cannot tell apart are absorbed rather than fitted: a
static offset and a shift of the time axis both end up in the species spectra. A shift of
the time axis leaves the rates and the spectra of end products unchanged; the spectra of
the starting species and of every intermediate absorb it and describe the state at the
first time label rather than at the true reaction start.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.kinetics.scheme import IndexStep, Scheme, Step, concentrations

if TYPE_CHECKING:
    from spectrahandler.kinetics.summary import ConditionSummary

__all__ = ["KineticFit", "fit_kinetics"]

#: Newton state: log rates, objective, damping, iteration, done.
type _State = tuple[Array, Array, Array, Array, Array]

#: Rate permutations are checked for equally good fits up to this many steps (6! = 720).
_MAX_STEPS_FOR_SWAP_CHECK = 6


@dataclass(frozen=True)
class KineticFit:
    """Per-run maximum likelihood fit of a first-order scheme.

    Attributes:
        scheme: The fitted scheme.
        species: Species in dataset order; every species axis follows it.
        run_ids: One per run.
        conditions: One per run; runs sharing a label are replicates.
        rates: Shape ``(n_run, n_steps)``, in 1 / ``time_unit``, step order of
            ``scheme.steps``.
        log_rate_sd: Shape ``(n_run, n_steps)``. Within-run standard deviation of the
            log rates from the curvature of the likelihood; ``NaN`` where the curvature
            is not positive definite. A diagnostic only: it assumes independent noise and
            is typically too small.
        spectra: Shape ``(n_run, n_species, n_wavelength)``, in absorbance per
            concentration unit. Includes any static offset of the run.
        concentrations: Shape ``(n_run, n_time, n_species)``; ``NaN`` where not measured.
        residuals: Shape ``(n_run, n_time, n_wavelength)``, data minus model, in
            absorbance; ``NaN`` where not measured.
        sigma: Shape ``(n_run, n_wavelength)``. Fitted noise standard deviation per
            wavelength, in absorbance.
        log_likelihood: Shape ``(n_run,)``, at the fitted rates.
        converged: One per run.
        ambiguities: ``(run_id, rates)`` for every permutation of a run's rates that fits
            as well or better than the fitted rates (within ``ambiguity_tolerance``), as
            in the rate swap of A -> B -> C. A permutation that fits better means the fit
            sits at a local optimum. Not resolved: the data cannot choose. The check is
            skipped for schemes with more than 6 steps.
        time_unit: Unit of time; rates are per this unit.
    """

    scheme: Scheme
    species: tuple[str, ...]
    run_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    rates: Array
    log_rate_sd: Array
    spectra: Array
    concentrations: Array
    residuals: Array
    sigma: Array
    log_likelihood: Array
    converged: tuple[bool, ...]
    ambiguities: tuple[tuple[str, tuple[float, ...]], ...]
    time_unit: str

    def summary(self, level: float = 0.95, min_ratio: float = 3.0) -> dict[str, "ConditionSummary"]:
        """Replicates pooled per condition; see :func:`~spectrahandler.kinetics.summarize`.

        Args:
            level: Coverage of the reported intervals.
            min_ratio: Warn when the between-replicate spread of a log rate is less than
                this multiple of its within-run standard deviation.

        Returns:
            One summary per condition, in order of first appearance.
        """
        from spectrahandler.kinetics.summary import summarize

        return summarize(self, level=level, min_ratio=min_ratio)


def _profile(
    log_rates: Array, time: Array, absorbance: Array, initial: Array, steps: tuple[IndexStep, ...]
) -> tuple[Array, Array, Array, Array]:
    """Profile log likelihood, concentrations, spectra and residual at ``log_rates``."""
    c = concentrations(time, jnp.exp(log_rates), initial, steps)
    spectra = jnp.linalg.lstsq(c, absorbance)[0]
    residual = absorbance - c @ spectra
    rss = (residual**2).sum(axis=0)
    n = absorbance.shape[0]
    log_lik = -0.5 * n * (jnp.log(2.0 * jnp.pi * rss / n) + 1.0).sum()
    return log_lik, c, spectra, residual


_evaluate = jax.jit(_profile, static_argnames=("steps",))


@partial(jax.jit, static_argnames=("steps", "max_iter"))
def _maximise(
    log_rates0: Array,
    time: Array,
    absorbance: Array,
    initial: Array,
    steps: tuple[IndexStep, ...],
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Damped Newton (Levenberg-Marquardt) on the negative profile log likelihood.

    With a handful of log rates and exact JAX gradient and Hessian, Newton converges in
    tens of steps where BFGS line searches can stall. Each step is capped at a factor e²
    in any rate. Converged means the Hessian is positive definite and the
    Newton decrement, the gain still available, is below 1e-6 log-likelihood units.

    Returns:
        Log rates, whether converged, and the Hessian of the negative log likelihood.
    """
    scale = absorbance.size  # objective per observation keeps the damping scale-free

    def objective(lr: Array) -> Array:
        return -_profile(lr, time, absorbance, initial, steps)[0] / scale

    grad, hess = jax.grad(objective), jax.hessian(objective)
    eye = jnp.eye(log_rates0.shape[0])

    def keep_going(state: _State) -> Array:
        _, _, _, iteration, done = state
        going: Array = (iteration < max_iter) & ~done
        return going

    def step(state: _State) -> _State:
        lr, value, damping, iteration, _ = state
        g, h = grad(lr), hess(lr)
        move = -jnp.linalg.solve(h + damping * eye, g)
        move = move * jnp.minimum(1.0, 2.0 / jnp.maximum(jnp.abs(move).max(), 1e-300))
        candidate = lr + move
        candidate_value = objective(candidate)
        better = jnp.isfinite(candidate_value) & (candidate_value < value)
        lr = jnp.where(better, candidate, lr)
        value = jnp.where(better, candidate_value, value)
        damping = jnp.where(better, damping / 3.0, damping * 4.0)
        done = (jnp.abs(grad(lr)).max() < 1e-10) | (damping > 1e12)
        return lr, value, damping, iteration + 1, done

    initial_state = (
        log_rates0,
        objective(log_rates0),
        jnp.asarray(1e-3),
        jnp.asarray(0),
        jnp.asarray(False),
    )
    lr: Array = jax.lax.while_loop(keep_going, step, initial_state)[0]
    g, h = grad(lr), hess(lr)
    decrement = g @ jnp.linalg.solve(h, g) * scale
    converged: Array = jnp.all(jnp.linalg.eigvalsh(h) > 0) & (decrement < 1e-6)
    curvature: Array = h * scale
    return lr, converged, curvature


def _start(
    dataset: SpectralDataset, scheme: Scheme, initial_rates: Mapping[Step, float] | None
) -> Array:
    """Starting log rates: given, or 1 / median run duration spaced by factors of 3."""
    if initial_rates is not None:
        missing = set(scheme.steps) ^ set(initial_rates)
        if missing:
            raise ValueError(f"initial_rates must name exactly the steps; mismatch: {missing}")
        start = jnp.asarray([float(initial_rates[step]) for step in scheme.steps])
        if not bool((start > 0).all()):
            raise ValueError("initial_rates must be positive")
        return jnp.log(start)
    measured_max = jnp.where(dataset.mask, dataset.time, -jnp.inf).max(axis=1)
    measured_min = jnp.where(dataset.mask, dataset.time, jnp.inf).min(axis=1)
    duration = float(jnp.median(measured_max - measured_min))
    if duration <= 0:
        raise ValueError("runs have zero duration; pass initial_rates")
    # Distinct starting rates: equal ones would leave A -> B -> C on its symmetric line.
    return jnp.log(1.0 / duration) - jnp.log(3.0) * jnp.arange(scheme.n_steps)


def fit_kinetics(
    dataset: SpectralDataset,
    scheme: Scheme,
    *,
    initial_rates: Mapping[Step, float] | None = None,
    t_offset: float = 0.0,
    max_iter: int = 200,
    ambiguity_tolerance: float = 1e-3,
    rank_tolerance: float = 1e-10,
) -> KineticFit:
    """Fit a first-order scheme to every run, each run on its own.

    Each run's initial concentrations come from ``dataset.initial_state``. Combine
    replicates with :meth:`KineticFit.summary`.

    Args:
        dataset: The runs; ``dataset.species`` must equal the scheme's species.
        scheme: The reaction scheme.
        initial_rates: Starting rate per step, in 1 / ``dataset.time_unit``. Defaults to
            1 / (median run duration) for the first step and a factor 3 slower for each
            further step. Give your own for schemes with more than one step.
        t_offset: Added to every time before fitting. A shift of the time axis leaves the
            rates and the spectra of end products unchanged; the spectra of the starting
            species and of every intermediate absorb it and describe the state at the first
            time label rather than at the true reaction start. Use it when the instrument's
            dead time is known from a calibration.
        max_iter: Newton iterations per run.
        ambiguity_tolerance: Log-likelihood shortfall below which a permutation of the
            fitted rates counts as fitting as well; a permutation that fits better is
            always recorded.
        rank_tolerance: Smallest allowed ratio of the smallest to the largest singular
            value of a run's concentration profiles.

    Returns:
        The fit, per run.

    Raises:
        RuntimeError: If JAX is not in float64 mode.
        ValueError: If the scheme's species differ from the dataset's, ``initial_rates``
            is malformed, a run has no more timepoints than species, or a run's
            concentration profiles are rank-deficient at the fitted rates.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "fit_kinetics needs float64: call jax.config.update('jax_enable_x64', True) "
            "before creating any arrays. spectrahandler never enables it itself."
        )
    steps = scheme.index_steps(dataset.species)
    log_start = _start(dataset, scheme, initial_rates)
    n_time, n_species = dataset.n_time, dataset.n_species

    results: list[dict[str, Array]] = []
    converged: list[bool] = []
    ambiguities: list[tuple[str, tuple[float, ...]]] = []
    for r, run_id in enumerate(dataset.run_ids):  # runs differ in length; a handful of them
        measured = np.asarray(dataset.mask[r])
        time = dataset.time[r][measured] + t_offset
        absorbance = dataset.absorbance[r][measured]
        initial = dataset.initial_state[r]
        if time.shape[0] <= n_species:
            raise ValueError(
                f"run {run_id!r}: {time.shape[0]} measured timepoints for {n_species} species"
            )

        log_rates, ok, hessian = _maximise(log_start, time, absorbance, initial, steps, max_iter)
        log_lik, c, spectra, residual = _evaluate(log_rates, time, absorbance, initial, steps)
        _, singular, vt = jnp.linalg.svd(c, full_matrices=False)
        if float(singular.min()) <= rank_tolerance * float(singular.max()):
            weight = np.abs(np.asarray(vt[-1]))
            involved = [dataset.species[i] for i in np.flatnonzero(weight > 0.1 * weight.max())]
            raise ValueError(
                f"run {run_id!r}: the concentration profiles are rank-deficient at the fitted "
                f"rates, so the spectra of {involved} are not identified "
                "(a species is never populated, or two species are always formed in a fixed ratio, "
                "as in parallel steps A -> B and A -> C)"
            )

        positive = bool((jnp.linalg.eigvalsh(hessian) > 0).all())
        sd = (
            jnp.sqrt(jnp.diag(jnp.linalg.inv(hessian)))
            if positive
            else jnp.full_like(log_rates, jnp.nan)
        )

        if scheme.n_steps <= _MAX_STEPS_FOR_SWAP_CHECK:
            for order in itertools.permutations(range(scheme.n_steps)):
                alternative = log_rates[jnp.asarray(order)]
                if float(jnp.abs(alternative - log_rates).max()) < 1e-6:
                    continue
                alt_lik = _evaluate(alternative, time, absorbance, initial, steps)[0]
                if float(alt_lik - log_lik) >= -ambiguity_tolerance:
                    ambiguities.append((run_id, tuple(float(x) for x in jnp.exp(alternative))))

        index = jnp.flatnonzero(jnp.asarray(measured))
        results.append(
            {
                "rates": jnp.exp(log_rates),
                "log_rate_sd": sd,
                "spectra": spectra,
                "concentrations": jnp.full((n_time, n_species), jnp.nan).at[index].set(c),
                "residuals": jnp.full((n_time, dataset.n_wavelength), jnp.nan)
                .at[index]
                .set(residual),
                "sigma": jnp.sqrt((residual**2).mean(axis=0)),
                "log_likelihood": log_lik,
            }
        )
        converged.append(bool(ok))

    stacked = {name: jnp.stack([res[name] for res in results]) for name in results[0]}
    return KineticFit(
        scheme=scheme,
        species=dataset.species,
        run_ids=dataset.run_ids,
        conditions=dataset.conditions,
        converged=tuple(converged),
        ambiguities=tuple(ambiguities),
        time_unit=dataset.time_unit,
        **stacked,
    )
