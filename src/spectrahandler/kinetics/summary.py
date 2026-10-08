"""Pool replicate runs per condition.

Rates are combined on the log scale: the geometric mean, with a Student-t interval on
``n - 1`` degrees of freedom from the spread between replicates. For normally
distributed replicate log rates with flat priors on their mean and on the log of their
spread, and negligible within-run error, this interval is also the posterior interval of
the mean log rate. Spectra are combined per wavelength on the linear scale.

The interval covers only what varies between the declared replicates: consecutive shots
from one loading do not cover preparation-to-preparation or day-to-day variation.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import betainc
from jax.scipy.stats import t as student_t
from jax.typing import ArrayLike

from spectrahandler.kinetics.scheme import Step

if TYPE_CHECKING:
    from spectrahandler.kinetics.fit import KineticFit

__all__ = ["ConditionSummary", "RateEstimate", "student_t_quantile", "summarize"]


@dataclass(frozen=True)
class RateEstimate:
    """One rate constant of one condition.

    Attributes:
        value: Geometric mean over replicates, in 1 / time unit.
        lower: Lower interval bound; ``None`` with one replicate.
        upper: Upper interval bound; ``None`` with one replicate.
        between_sd_log: Standard deviation of the log rate between replicates; ``None``
            with one replicate.
        within_sd_log: Mean within-run standard deviation of the log rate (diagnostic).
        n: Number of replicates.
    """

    value: float
    lower: float | None
    upper: float | None
    between_sd_log: float | None
    within_sd_log: float
    n: int

    def density(self, rate: ArrayLike) -> Array:
        """Posterior density of the rate, the distribution behind ``lower`` and ``upper``.

        The log rate follows a Student-t on ``n - 1`` degrees of freedom, centred on
        ``log(value)`` with scale ``between_sd_log / sqrt(n)``; dividing by the rate turns
        it into a density over the rate itself.

        Args:
            rate: Rates at which to evaluate, positive, in 1 / time unit.

        Returns:
            The density, same shape as ``rate``, in time unit.

        Raises:
            ValueError: With one replicate: there is no spread to build it from.
        """
        if self.between_sd_log is None:
            raise ValueError("one replicate: no replicate-based density")
        k = jnp.asarray(rate)
        scale = self.between_sd_log / math.sqrt(self.n)
        log_density = student_t.logpdf(
            jnp.log(k), self.n - 1, loc=math.log(self.value), scale=scale
        )
        density: Array = jnp.exp(log_density) / k
        return density


@dataclass(frozen=True)
class ConditionSummary:
    """Replicates of one condition, pooled.

    Attributes:
        condition: The label.
        run_ids: The replicate runs.
        rates: One estimate per step.
        spectrum_mean: Shape ``(n_species, n_wavelength)``, mean over replicates.
        spectrum_lower: Same shape, lower interval bound; ``None`` with one replicate.
        spectrum_upper: Same shape, upper interval bound; ``None`` with one replicate.
        warnings: Everything the numbers need read alongside them.
    """

    condition: str
    run_ids: tuple[str, ...]
    rates: dict[Step, RateEstimate]
    spectrum_mean: Array
    spectrum_lower: Array | None
    spectrum_upper: Array | None
    warnings: tuple[str, ...]


def student_t_quantile(p: float, df: float) -> float:
    """Quantile of Student's t distribution, for ``p`` in (0.5, 1).

    Bisection on the CDF ``1 - I_{df/(df+x²)}(df/2, 1/2) / 2``, using
    ``jax.scipy.special.betainc``; no SciPy.

    Args:
        p: Probability, in (0.5, 1).
        df: Degrees of freedom, positive.

    Returns:
        ``x`` with ``P(T ≤ x) = p``.

    Raises:
        ValueError: If ``p`` or ``df`` is out of range.
    """
    if not 0.5 < p < 1.0 or df <= 0:
        raise ValueError(f"need 0.5 < p < 1 and df > 0, got p={p}, df={df}")

    def cdf(x: Array) -> Array:
        return 1.0 - 0.5 * betainc(df / 2.0, 0.5, df / (df + x * x))

    def halve(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < p
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 200, halve, (jnp.asarray(0.0), jnp.asarray(1e6)))
    return float(0.5 * (lo + hi))


def summarize(
    fit: "KineticFit", *, level: float = 0.95, min_ratio: float = 3.0
) -> dict[str, ConditionSummary]:
    """Pool the replicate runs of every condition.

    Args:
        fit: From :func:`~spectrahandler.kinetics.fit_kinetics`.
        level: Coverage of the reported intervals, in (0, 1).
        min_ratio: Warn when a log rate's between-replicate standard deviation is below
            this multiple of its mean within-run standard deviation: the interval then
            ignores within-run uncertainty that is not small.

    Returns:
        One summary per condition, in order of first appearance.

    Raises:
        ValueError: If ``level`` is not in (0, 1).
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    summaries: dict[str, ConditionSummary] = {}
    for condition in dict.fromkeys(fit.conditions):
        index = jnp.asarray([i for i, c in enumerate(fit.conditions) if c == condition])
        n = int(index.shape[0])
        log_rates = jnp.log(fit.rates[index])
        within = jnp.nanmean(fit.log_rate_sd[index], axis=0)
        spectra = fit.spectra[index]
        run_ids = tuple(fit.run_ids[int(i)] for i in index)
        warnings: list[str] = []

        if n == 1:
            rates = {
                step: RateEstimate(
                    float(jnp.exp(log_rates[0, j])), None, None, None, float(within[j]), 1
                )
                for j, step in enumerate(fit.scheme.steps)
            }
            lower = upper = None
            warnings.append(
                "one run: no replicate-based uncertainty; within_sd_log assumes independent "
                "noise and is typically too small"
            )
        else:
            q = student_t_quantile(0.5 + level / 2.0, n - 1)
            mean, sd = log_rates.mean(axis=0), log_rates.std(axis=0, ddof=1)
            half = q * sd / jnp.sqrt(n)
            rates = {
                step: RateEstimate(
                    value=float(jnp.exp(mean[j])),
                    lower=float(jnp.exp(mean[j] - half[j])),
                    upper=float(jnp.exp(mean[j] + half[j])),
                    between_sd_log=float(sd[j]),
                    within_sd_log=float(within[j]),
                    n=n,
                )
                for j, step in enumerate(fit.scheme.steps)
            }
            for j, (a, b) in enumerate(fit.scheme.steps):
                if not float(sd[j]) >= min_ratio * float(within[j]):
                    warnings.append(
                        f"{a} -> {b}: the spread between replicates ({float(sd[j]):.2g} in log "
                        f"rate) is not clearly larger than the fit precision "
                        f"({float(within[j]):.2g}); the interval ignores within-run uncertainty"
                    )
            spread = q * spectra.std(axis=0, ddof=1) / jnp.sqrt(n)
            lower, upper = spectra.mean(axis=0) - spread, spectra.mean(axis=0) + spread

        not_converged = [run_ids[i] for i in range(n) if not fit.converged[int(index[i])]]
        if not_converged:
            warnings.append(f"not converged: {not_converged}")
        swapped = sorted({run for run, _ in fit.ambiguities if run in run_ids})
        if swapped:
            warnings.append(
                f"rate permutations fit equally well in {swapped}: the rates are not "
                "separately identified (see KineticFit.ambiguities)"
            )
        summaries[condition] = ConditionSummary(
            condition=condition,
            run_ids=run_ids,
            rates=rates,
            spectrum_mean=spectra.mean(axis=0),
            spectrum_lower=lower,
            spectrum_upper=upper,
            warnings=tuple(warnings),
        )
    return summaries
