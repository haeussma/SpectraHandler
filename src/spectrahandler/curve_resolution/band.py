"""Feasible-band curve resolution.

The data fix the product ``C @ S`` but not its split into ``C`` and ``S``. Every split is
a re-mixing ``C = X @ T``, ``S = inv(T) @ Y`` of the data's own rank-K patterns. Closure
and references pin some entries of ``T``; non-negativity bounds the rest to a feasible
region. This module reports that region's projection onto every concentration and
spectrum value -- the band -- plus a noise margin, and flat draws over the region as a
labelled typical-solution summary. See ``docs/decisions/0003-feasible-band-not-posterior.md``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset

__all__ = ["FeasibleBand", "resolve_band"]

#: theta -> (concentrations, spectra, concentration noise sd, spectra noise sd).
type _Parts = Callable[[Array], tuple[Array, Array, Array, Array]]
#: theta -> summed non-negativity violation in noise sd; zero means feasible.
type _Violation = Callable[[Array], Array]
type _State = tuple[Array, Array, Array, Array, Array]


@dataclass(frozen=True)
class FeasibleBand:
    """Everything the data allow, for every concentration and spectrum value.

    Attributes:
        concentration_lower: Band lower edge incl. noise margin, clipped at zero, shape
            ``(n_run, n_time, n_species)``, in ``concentration_unit``.
        concentration_upper: Band upper edge incl. noise margin, same shape and unit.
        spectra_lower: Band lower edge incl. noise margin, clipped at zero, shape
            ``(n_species, n_wavelength)``, in absorbance per ``concentration_unit``.
        spectra_upper: Band upper edge incl. noise margin, same shape and unit.
        concentration_ambiguity: Extremes over the feasible region without the noise
            margin, shape ``(2, n_run, n_time, n_species)``: the part no amount of
            repeating the same measurement removes.
        spectra_ambiguity: Same, shape ``(2, n_species, n_wavelength)``.
        concentration_draws: Flat draws over the region plus propagated noise, shape
            ``(n_draw, n_run, n_time, n_species)``. A typical-solution summary under an
            explicit flat prior on the free re-mixing numbers, not a calibrated interval.
        spectra_draws: Same, shape ``(n_draw, n_species, n_wavelength)``.
        sigma: Noise standard deviation used, in absorbance units.
        n_free: Free re-mixing numbers left after closure and references.
    """

    concentration_lower: Array
    concentration_upper: Array
    spectra_lower: Array
    spectra_upper: Array
    concentration_ambiguity: Array
    spectra_ambiguity: Array
    concentration_draws: Array
    spectra_draws: Array
    sigma: float
    n_free: int


def _estimate_sigma(matrix: Array, rank: int) -> float:
    """Noise standard deviation from the residual after a rank-``rank`` SVD.

    Args:
        matrix: Data, shape ``(n_row, n_col)``.
        rank: Number of components kept.

    Returns:
        ``sqrt(sum of discarded singular values squared / ((n_row - rank) * (n_col - rank)))``.
    """
    s = jnp.linalg.svd(matrix, compute_uv=False)
    n_row, n_col = matrix.shape
    return float(jnp.sqrt((s[rank:] ** 2).sum() / ((n_row - rank) * (n_col - rank))))


def resolve_band(
    dataset: SpectralDataset,
    key: Array,
    *,
    sigma: float | None = None,
    z_slack: float = 3.5,
    n_iter: int = 8000,
    n_burn: int = 500,
    thin: int = 2,
    n_als: int = 200,
    n_restart: int = 8,
    n_search: int = 30000,
) -> FeasibleBand:
    """Resolve ``dataset`` into the band of every spectra/profile split the data allow.

    Closure: in every run, species sum to ``initial_state[run].sum()`` at every time.
    Every species with a finite ``reference_spectra`` row is pinned by it; species
    without one are resolved only up to relabelling among themselves.

    Args:
        dataset: Validated observations; ``mask`` must be all True.
        key: PRNG key for the start search, the hit-and-run chain and the noise draws.
        sigma: Noise standard deviation in absorbance units. ``None`` estimates it from
            the rank-``n_species`` residual of the time course.
        z_slack: Non-negativity is tested at ``-z_slack`` noise sd, and the band gets a
            ``z_slack`` sd noise margin. 3.5 keeps ~100 true zeros inside ~99% of the time.
        n_iter: Hit-and-run iterations.
        n_burn: Iterations discarded before draws are kept.
        thin: Keep every ``thin``-th draw after burn-in.
        n_als: MCR-ALS sweeps in the reduced space that produce the starting split.
        n_restart: Parallel restarts of the feasible-start search.
        n_search: Maximum steps per restart of the feasible-start search.

    Returns:
        The band, its ambiguity part, and flat draws.

    Raises:
        NotImplementedError: If ``dataset.mask`` has any False entry.
        ValueError: If no feasible split is found: the rank, closure or references
            contradict the data.
    """
    if not bool(dataset.mask.all()):
        raise NotImplementedError("resolve_band requires fully measured runs")
    k = dataset.n_species
    n_row = dataset.n_run * dataset.n_time
    time_course = dataset.absorbance.reshape(n_row, dataset.n_wavelength)
    sigma = _estimate_sigma(time_course, k) if sigma is None else sigma

    # Each reference becomes one more data row, weighted so its noise is sigma too, with
    # composition `weight` of its own species and zero of every other.
    has_ref = jnp.isfinite(dataset.reference_spectra).all(axis=1)
    ref_idx = jnp.flatnonzero(has_ref)
    weight = sigma / dataset.reference_sigma[ref_idx]
    ref_rows = weight[:, None] * dataset.reference_spectra[ref_idx]
    ref_comp = weight[:, None] * jnp.eye(k)[ref_idx]

    data = jnp.concatenate([time_course, ref_rows])
    u, s, vt = jnp.linalg.svd(data, full_matrices=False)
    x, y, s_k = u[:, :k] * s[:k], vt[:k], s[:k]

    # Equalities on vec(T), row-major. Closure: X @ T @ 1 = row totals, so T @ 1 = t_star.
    # References: x_ref @ T = composition.
    totals = jnp.repeat(dataset.initial_state.sum(axis=1), dataset.n_time)
    row_sums = jnp.concatenate([totals, ref_comp.sum(axis=1)])
    t_star = jnp.linalg.lstsq(x, row_sums)[0]
    eq = jnp.concatenate([jnp.kron(jnp.eye(k), jnp.ones(k)), jnp.kron(x[n_row:], jnp.eye(k))])
    rhs = jnp.concatenate([t_star, ref_comp.ravel()])
    t_particular = jnp.linalg.lstsq(eq, rhs)[0]
    _, eq_sv, eq_vt = jnp.linalg.svd(eq)
    rank = int((eq_sv > 1e-8 * eq_sv[0]).sum())
    null = eq_vt[rank:]
    x_time = x[:n_row]

    def parts(theta: Array) -> tuple[Array, Array, Array, Array]:
        t = (t_particular + theta @ null).reshape(k, k)
        r = jnp.linalg.inv(t)
        c: Array = x_time @ t
        sp: Array = r @ y
        sd_c: Array = sigma * jnp.linalg.norm(t, axis=0)
        sd_s: Array = sigma * jnp.sqrt(((r / s_k) ** 2).sum(axis=1))
        return c, sp, sd_c, sd_s

    def violation(theta: Array) -> Array:
        t = (t_particular + theta @ null).reshape(k, k)
        c, sp, sd_c, sd_s = parts(theta)
        v = (jnp.clip(-(c + z_slack * sd_c), 0.0) / sd_c).sum() + (
            jnp.clip(-(sp + z_slack * sd_s[:, None]), 0.0) / sd_s[:, None]
        ).sum()
        return jnp.where(jnp.abs(jnp.linalg.det(t)) < 1e-12, jnp.inf, v)

    def to_theta(t: Array) -> Array:
        """The nearest split satisfying closure and references, in free coordinates."""
        theta: Array = (t.ravel() - t_particular) @ null.T
        return theta

    def alternate(_: int, theta: Array) -> Array:
        """One MCR-ALS sweep in the reduced space: clip C and refit, clip S and refit.

        ``y`` has orthonormal rows, so ``S = inv(T) @ y`` inverts as ``inv(S @ y.T)``.
        """
        t = (t_particular + theta @ null).reshape(k, k)
        t = jnp.linalg.lstsq(x_time, jnp.clip(x_time @ t, 0.0))[0]
        t = (t_particular + to_theta(t) @ null).reshape(k, k)
        t = jnp.linalg.inv(jnp.clip(jnp.linalg.inv(t) @ y, 0.0) @ y.T)
        return to_theta(t)

    guess = _initial_guess(x, x_time, totals, ref_comp, ref_idx, t_particular, null, k)
    theta0 = jax.lax.fori_loop(0, n_als, alternate, guess)
    k_search, k_chain, k_noise = jax.random.split(key, 3)
    theta, v = _find_feasible(violation, theta0, k_search, n_restart, n_search)
    if not float(v) == 0.0:
        raise ValueError(
            f"no feasible split found (violation {float(v):.3g}): the rank, closure or "
            "references contradict the data"
        )

    lo, hi, draws = _hit_and_run(parts, violation, theta, k_chain, n_iter)
    c, sp, sd_c, sd_s = (d[n_burn::thin] for d in draws)
    shape = (dataset.n_run, dataset.n_time, k)

    margin_c, margin_s = z_slack * jnp.median(sd_c, axis=0), z_slack * jnp.median(sd_s, axis=0)
    n_c = n_row * k
    c_min, c_max = lo[:n_c].reshape(n_row, k), hi[:n_c].reshape(n_row, k)
    s_min, s_max = lo[n_c:].reshape(k, -1), hi[n_c:].reshape(k, -1)

    kc, ks = jax.random.split(k_noise)
    c_draws = c + sd_c[:, None, :] * jax.random.normal(kc, c.shape)
    s_draws = sp + sd_s[:, :, None] * jax.random.normal(ks, sp.shape)
    return FeasibleBand(
        concentration_lower=jnp.clip(c_min - margin_c, 0.0).reshape(shape),
        concentration_upper=(c_max + margin_c).reshape(shape),
        spectra_lower=jnp.clip(s_min - margin_s[:, None], 0.0),
        spectra_upper=s_max + margin_s[:, None],
        concentration_ambiguity=jnp.stack([c_min, c_max]).reshape(2, *shape),
        spectra_ambiguity=jnp.stack([s_min, s_max]),
        concentration_draws=c_draws.reshape(-1, *shape),
        spectra_draws=s_draws,
        sigma=sigma,
        n_free=int(null.shape[0]),
    )


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
    # ponytail: greedy farthest-row picking, a K-step Python loop over species, not data.
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
    key: Array,
    n_restart: int,
    n_search: int,
) -> tuple[Array, Array]:
    """Adaptive random search for a point with zero violation, restarts in parallel."""

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
                jnp.where(better, step * 1.3, jnp.maximum(step * 0.97, 1e-5)),
                i + 1,
                key,
            )
            return out

        init = (start, violation(start), jnp.asarray(3.0), jnp.asarray(0), key)
        theta: Array
        v: Array
        theta, v, _, _, _ = jax.lax.while_loop(cond, body, init)
        return theta, v

    k_start, k_search = jax.random.split(key)
    spread = 3.0 * jnp.arange(n_restart)[:, None]
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
    key: Array,
    n_iter: int,
) -> tuple[Array, Array, tuple[Array, Array, Array, Array]]:
    """Uniform draws over the region, plus the extreme value of every quantity seen."""

    def edge(theta: Array, u: Array) -> Array:
        """Distance along ``u`` to the region's boundary: double, then bisect."""
        t = jax.lax.while_loop(
            lambda t: (violation(theta + t * u) == 0) & (t < 1e4), lambda t: 2 * t, 1e-3
        )

        def bisect(_: int, ab: tuple[Array, Array]) -> tuple[Array, Array]:
            a, b = ab
            mid = 0.5 * (a + b)
            inside = violation(theta + mid * u) == 0
            return jnp.where(inside, mid, a), jnp.where(inside, b, mid)

        distance: Array = jax.lax.fori_loop(0, 25, bisect, (jnp.asarray(0.0), t))[0]
        return distance

    def flat(theta: Array) -> Array:
        c, sp, _, _ = parts(theta)
        return jnp.concatenate([c.ravel(), sp.ravel()])

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
