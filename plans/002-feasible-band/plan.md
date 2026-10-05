# Feasible-Band Curve Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve a reaction time course into the band of every pure-spectra /
concentration split the data allow, prove on synthetic data that the band contains the
truth, show whether the chosen number of species is enough, read the JASCO fixtures,
and retire the v0 random-walk model.

**Architecture:** `resolve_band` takes the rank-K SVD patterns of the data, parametrises
every split as a re-mixing matrix `T` restricted by closure and references, starts from
MCR-ALS in that reduced space, and walks the feasible region by hit-and-run with
slice-sampling shrinkage. It reports the extremes of every value plus a noise margin.
Readers are a separate NumPy module that returns arrays with no preprocessing. A
diagnostics module reports what is left after K components and whether it is white
noise; an optional `plot` module draws it.

**Tech Stack:** JAX (`jax.numpy`, `jax.lax.scan` / `while_loop` / `fori_loop`, `vmap`),
NumPy (readers, plotting and tests only), matplotlib (optional `plot` extra, and a
dev dependency for its test), pytest, Astro Starlight for docs.

**Spec:** [`spec.md`](spec.md) — read it first. Decision:
[ADR 0003](../../docs/decisions/0003-feasible-band-not-posterior.md) (accepted). Data
loading: [ADR 0002](../../docs/decisions/0002-scope-boundary-against-mcrals.md)
(accepted).

## Global Constraints

Copied from [`CLAUDE.md`](../../CLAUDE.md) and [`spec.md`](spec.md) §0. Every task's
requirements implicitly include these.

- Python ≥ 3.13. `uv run ...` for everything; never bare `python`, never `pip`.
- **Do not import SciPy.** NumPy only in `jasco.py` (an IO boundary) and in tests.
- float64 is enabled in `tests/conftest.py`. Never force it at library import.
- Full annotations on every parameter and return, including `-> None`. Where `ty`
  reports `unsound-return-statement` on a JAX result, bind it to an annotated name first
  (`out: Array = ...; return out`) — that is the repo's idiom.
- Google-style docstring on every public module, class and function. Line length 100.
- Explicit PRNG keys. No Python loops over array axes. No in-place mutation.
- Numerical assertions use `pytest.approx` or `numpy.testing`, never `==` on floats
  (exact values read from a file are the exception).
- A step is done when `uv run ruff check . && uv run ty check && uv run pytest` is clean.
- Every code block below was run before this plan was written (spec §5). If a measured
  number in a test docstring does not reproduce, stop and report — do not loosen the
  threshold.

## Preconditions

None blocking. ADRs 0002 and 0003 are accepted. Task 4 (readers) is independent of
Tasks 1–3 and can run in parallel. Task 5 (diagnostics) needs Task 4, because one of its
tests reads a real fixture. Task 6 requires Task 3's gate to pass.

---

### Task 0: Commit the decision record

The ADR cites the bench report by path, and plan 001's directory is deleted at the end
of this plan. Commit the evidence first, so it stays reachable in git history.

**Files:**
- Add: `plans/001-bayesian-curve-resolution/report-spectrum-descriptions.md`
- Add: `plans/001-bayesian-curve-resolution/bench/` (`__pycache__` is gitignored)
- Add: `docs/decisions/0003-feasible-band-not-posterior.md`, `plans/002-feasible-band/`
- Modify (already edited): `docs/decisions/0002-scope-boundary-against-mcrals.md`,
  `plans/001-bayesian-curve-resolution/spec.md`

- [ ] **Step 1: Check what is staged**

Run: `git status --short`
Expected: exactly the paths above, nothing under `src/` or `tests/`.

- [ ] **Step 2: Commit**

```bash
git add plans/001-bayesian-curve-resolution docs/decisions plans/002-feasible-band
git commit -m "docs: decide on the feasible band (ADR 0003) and data loading (ADR 0002)"
```

---

### Task 1: Give reference spectra a checked meaning

**Files:**
- Modify: `src/spectrahandler/curve_resolution/dataset.py` (class docstring, `_validate`)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SpectralDataset` guarantees that every `reference_spectra` row is all finite
  or all `NaN`, and that `reference_sigma[k]` is finite and positive wherever row `k` is
  finite. Task 2 relies on both.

- [ ] **Step 1: Write the failing tests**

In `tests/test_dataset.py`, add three cases to the `test_validation_rejects`
parametrize list, directly after the `initial_state` NaN case (the `+` lines are new;
keep the indentation exactly as shown):

```diff
         ({"initial_state": jnp.array([[jnp.nan, 1.0]])}, "finite"),
+        (
+            {
+                "reference_spectra": jnp.array([[0.0, jnp.nan, 0.0, 0.0], [jnp.nan] * 4]),
+                "reference_sigma": jnp.array([1.0, jnp.nan]),
+            },
+            "all finite or all NaN",
+        ),
+        (
+            {
+                "reference_spectra": jnp.array([[0.0] * 4, [jnp.nan] * 4]),
+                "reference_sigma": jnp.array([jnp.nan, jnp.nan]),
+            },
+            "reference_sigma",
+        ),
+        (
+            {
+                "reference_spectra": jnp.array([[0.0] * 4, [jnp.nan] * 4]),
+                "reference_sigma": jnp.array([0.0, jnp.nan]),
+            },
+            "reference_sigma",
+        ),
     ],
 )
 def test_validation_rejects(overrides: dict[str, Array], message: str) -> None:
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: 3 failures, `Failed: DID NOT RAISE <class 'ValueError'>`.

- [ ] **Step 3: Implement**

In `src/spectrahandler/curve_resolution/dataset.py`, replace the two attribute entries
in the `SpectralDataset` class docstring:

```text
        reference_spectra: Known pure spectra, shape ``(n_species, n_wavelength)``,
            ``NaN`` for species without one. Carried but unused in v0.
        reference_sigma: Uncertainty on each reference, shape ``(n_species,)``, ``NaN``
            where absent.
```

with:

```text
        reference_spectra: Known pure spectra, shape ``(n_species, n_wavelength)``, in
            absorbance per ``concentration_unit`` (a scan divided by the concentration
            it was taken at). A row is all finite, or all ``NaN`` for no reference.
        reference_sigma: Per-channel noise standard deviation of each reference row,
            shape ``(n_species,)``, same units. Finite and positive wherever the row is
            finite; ignored where it is ``NaN``.
```

Then append to the end of `_validate`, inside the function (the `+` lines are new):

```diff
     if not bool((ds.initial_state >= 0).all()):
         raise ValueError("initial_state must be non-negative")
+
+    ref_finite = jnp.isfinite(ds.reference_spectra)
+    has_ref = ref_finite.all(axis=1)
+    if not bool((has_ref | ~ref_finite.any(axis=1)).all()):
+        raise ValueError("each reference_spectra row must be all finite or all NaN")
+    sigma_ok = jnp.isfinite(ds.reference_sigma) & (ds.reference_sigma > 0)
+    if not bool((sigma_ok | ~has_ref).all()):
+        raise ValueError(
+            "reference_sigma must be finite and positive for every species with a reference"
+        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: PASS, all cases.

- [ ] **Step 5: Full gate, then commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
git add src/spectrahandler/curve_resolution/dataset.py tests/test_dataset.py
git commit -m "feat: give reference spectra units and validate them"
```

---

### Task 2: `resolve_band` and its invariants

**Files:**
- Create: `src/spectrahandler/curve_resolution/band.py`
- Modify: `src/spectrahandler/curve_resolution/__init__.py`
- Test: `tests/test_band.py`

**Interfaces:**
- Consumes: `SpectralDataset` (Task 1 guarantees on references), `make_realistic_dataset`.
- Produces:
  - `FeasibleBand` (frozen dataclass): `concentration_lower`, `concentration_upper`
    `(n_run, n_time, n_species)`; `spectra_lower`, `spectra_upper`
    `(n_species, n_wavelength)`; `concentration_ambiguity` `(2, n_run, n_time,
    n_species)`; `spectra_ambiguity` `(2, n_species, n_wavelength)`;
    `concentration_draws` `(n_draw, n_run, n_time, n_species)`; `spectra_draws`
    `(n_draw, n_species, n_wavelength)`; `sigma: float`; `n_free: int`.
  - `resolve_band(dataset: SpectralDataset, key: Array, *, sigma: float | None = None,
    z_slack: float = 3.5, n_iter: int = 8000, n_burn: int = 500, thin: int = 2,
    n_als: int = 200, n_restart: int = 8, n_search: int = 30000) -> FeasibleBand`.
  - Both exported from `spectrahandler.curve_resolution`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_band.py
"""Invariants of resolve_band that hold on any data, independent of coverage."""

import jax
import jax.numpy as jnp
import pytest

from spectrahandler.curve_resolution import (
    FeasibleBand,
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

#: The realistic set's noise, and the concentration (uM) of the simulated reference scan.
SIGMA, REF_CONC = 0.002, 50.0


def _with_reference(dataset: SpectralDataset, spectrum_a: jax.Array) -> SpectralDataset:
    """The same data plus a reference spectrum for species ``a`` only."""
    refs = jnp.full((dataset.n_species, dataset.n_wavelength), jnp.nan).at[0].set(spectrum_a)
    return SpectralDataset.create(
        absorbance=dataset.absorbance,
        time=dataset.time,
        wavelength=dataset.wavelength,
        species=dataset.species,
        initial_state=dataset.initial_state,
        run_ids=dataset.run_ids,
        reference_spectra=refs,
        reference_sigma=jnp.array([SIGMA / REF_CONC, jnp.nan, jnp.nan]),
    )


@pytest.fixture(scope="module")
def resolved() -> tuple[SpectralDataset, FeasibleBand]:
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, spectra[0])
    return data, resolve_band(data, jax.random.key(1), n_iter=2000)


def test_shapes(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    data, band = resolved
    c_shape = (data.n_run, data.n_time, data.n_species)
    s_shape = (data.n_species, data.n_wavelength)
    assert band.concentration_lower.shape == band.concentration_upper.shape == c_shape
    assert band.spectra_lower.shape == band.spectra_upper.shape == s_shape
    assert band.concentration_ambiguity.shape == (2, *c_shape)
    assert band.spectra_ambiguity.shape == (2, *s_shape)
    assert band.concentration_draws.shape[1:] == c_shape
    assert band.spectra_draws.shape[1:] == s_shape


def test_one_reference_leaves_four_free_numbers(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    """9 entries of T, minus 3 for closure, minus 3 for the reference, plus 1 shared."""
    assert resolved[1].n_free == 4


def test_no_reference_leaves_six_free_numbers() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    assert resolve_band(data, jax.random.key(1), n_iter=200, n_burn=100).n_free == 6


def test_sigma_is_estimated(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    assert resolved[1].sigma == pytest.approx(SIGMA, rel=0.1)


def test_band_brackets_ambiguity_and_is_non_negative(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    _, band = resolved
    assert bool((band.concentration_lower >= 0).all())
    assert bool((band.spectra_lower >= 0).all())
    # The ambiguity extremes may sit a little below zero (the noise slack); the band is
    # clipped at zero, so compare against the clipped extreme.
    floor = jnp.clip(band.concentration_ambiguity[0], 0.0)
    assert bool((band.concentration_lower <= floor + 1e-12).all())
    assert bool((band.concentration_upper >= band.concentration_ambiguity[1]).all())
    assert bool((band.spectra_upper >= band.spectra_ambiguity[1]).all())


def test_draws_respect_closure(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Closure holds in least squares over the rows, so single rows scatter at noise level.

    Measured: overall mean 12.497 uM, rows between 12.31 and 12.71 uM.
    """
    data, band = resolved
    totals = band.concentration_draws.sum(axis=-1).mean(axis=0)
    total = float(data.initial_state.sum())
    assert float(totals.mean()) == pytest.approx(total, abs=0.02)
    assert float(jnp.abs(totals - total).max()) < 0.3


def test_draws_reproduce_the_data(resolved: tuple[SpectralDataset, FeasibleBand]) -> None:
    """Every split reproduces the data to the noise: residual rms close to sigma."""
    data, band = resolved
    predicted = jnp.einsum("drtk,dkw->rtw", band.concentration_draws, band.spectra_draws)
    predicted = predicted / band.concentration_draws.shape[0]
    rms = float(jnp.sqrt(((predicted - data.absorbance) ** 2).mean()))
    assert rms < 1.5 * SIGMA


def test_reference_lies_inside_its_band(
    resolved: tuple[SpectralDataset, FeasibleBand],
) -> None:
    data, band = resolved
    ref = data.reference_spectra[0]
    assert bool(((ref >= band.spectra_lower[0]) & (ref <= band.spectra_upper[0])).all())


def test_requires_fully_measured_runs() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    masked = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        mask=data.mask.at[0, 3].set(False),
    )
    with pytest.raises(NotImplementedError, match="fully measured"):
        resolve_band(masked, jax.random.key(1))


def test_contradictory_reference_is_reported() -> None:
    """A negative reference cannot be a non-negative spectrum: no split is feasible."""
    data, spectra, _ = make_realistic_dataset(jax.random.key(0))
    data = _with_reference(data, -spectra[0])
    with pytest.raises(ValueError, match="no feasible split"):
        resolve_band(data, jax.random.key(1), n_restart=2, n_search=2000)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_band.py -q`
Expected: collection error — `cannot import name 'FeasibleBand'`.

- [ ] **Step 3: Write `band.py`**

```python
# src/spectrahandler/curve_resolution/band.py
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
```

Then in `src/spectrahandler/curve_resolution/__init__.py` add the import as the first
import line and the two names to `__all__`, keeping it sorted:

```python
from spectrahandler.curve_resolution.band import FeasibleBand, resolve_band
```

```python
__all__ = [
    "FeasibleBand",
    "SpectralDataset",
    "curve_resolution_model",
    "fit",
    "make_easy_dataset",
    "make_realistic_dataset",
    "resolve_band",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_band.py -q`
Expected: PASS, 10 tests, about 15 s (the first `resolve_band` call compiles).

- [ ] **Step 5: Full gate, then commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
git add src/spectrahandler/curve_resolution tests/test_band.py
git commit -m "feat: resolve_band reports the feasible band of every spectra/profile split"
```

---

### Task 3: The gate — does the band contain the truth?

**Files:**
- Test: `tests/test_band_coverage.py`
- Modify: `plans/002-feasible-band/spec.md` §8

**Interfaces:**
- Consumes: `resolve_band`, `FeasibleBand` (Task 2); `make_realistic_dataset`.
- Produces: nothing new. Task 5 may start only once this passes.

These tests mirror the bench (`plans/001-bayesian-curve-resolution/bench/remix.py`) on
the library code. If one fails, do **not** tune thresholds or defaults to make it pass.
Report the measured number against the docstring's.

- [ ] **Step 1: Write the tests**

```python
# tests/test_band_coverage.py
"""The gate: does the band contain the truth?

Mirrors the bench in ``plans/001-bayesian-curve-resolution/bench`` (``remix.py``):
realistic a -> b -> c data with the first scan dropped, one reference scan of ``a`` at
50 uM, sigma 0.002. "harder" lifts every spectrum by 0.003 AU/uM so no true value is
exactly zero, which makes it the clean test of the split. Numbers measured on the JAX
port before this plan was written are quoted next to each threshold.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from spectrahandler.curve_resolution import (
    FeasibleBand,
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

SIGMA, REF_CONC = 0.002, 50.0
LIFT = {"realistic": 0.0, "harder": 0.003}


def _profiles(k1: float, k2: float, time: Array) -> Array:
    """Consecutive a -> b -> c at 12.5 uM total, shape ``(n_time, 3)``."""
    a = jnp.exp(-k1 * time)
    b = k1 / (k2 - k1) * (jnp.exp(-k1 * time) - jnp.exp(-k2 * time))
    return 12.5 * jnp.stack([a, b, 1.0 - a - b], axis=-1)


def _dataset(
    key: Array, spectra: Array, concentrations: Array, initial_state: Array
) -> SpectralDataset:
    """Noisy runs plus a noisy 50 uM reference scan of ``a``, as a per-uM spectrum."""
    k_data, k_ref = jax.random.split(key)
    n_run, n_time, _ = concentrations.shape
    clean = jnp.einsum("rtk,kw->rtw", concentrations, spectra)
    ref = spectra[0] + SIGMA / REF_CONC * jax.random.normal(k_ref, spectra[0].shape)
    _, _, n_wavelength = clean.shape
    return SpectralDataset.create(
        absorbance=clean + SIGMA * jax.random.normal(k_data, clean.shape),
        time=jnp.tile(jnp.linspace(0.0, 10.0, n_time + 1)[1:], (n_run, 1)),
        wavelength=jnp.linspace(340.0, 700.0, n_wavelength),
        species=("a", "b", "c"),
        initial_state=initial_state,
        run_ids=tuple(f"run{i}" for i in range(n_run)),
        reference_spectra=jnp.full((3, n_wavelength), jnp.nan).at[0].set(ref),
        reference_sigma=jnp.array([SIGMA / REF_CONC, jnp.nan, jnp.nan]),
    )


def _bench(seed: int, lift: float) -> tuple[SpectralDataset, Array, Array]:
    """The bench's single-run data for one noise draw."""
    _, spectra, concentrations = make_realistic_dataset(jax.random.key(0), noise=0.0)
    spectra, concentrations = spectra + lift, concentrations[:, 1:]
    initial = jnp.array([[12.5, 0.0, 0.0]])
    return (
        _dataset(jax.random.key(1000 + seed), spectra, concentrations, initial),
        spectra,
        concentrations,
    )


def _match(fitted: Array, truth: Array) -> Array:
    """For each true species, the fitted index with the most similar spectrum.

    Only ``a`` has a reference, so ``b`` and ``c`` come back in either order.
    """
    normed_truth = truth / jnp.linalg.norm(truth, axis=-1, keepdims=True)
    normed_fit = fitted / jnp.linalg.norm(fitted, axis=-1, keepdims=True)
    taken: list[int] = []
    for row in normed_truth @ normed_fit.T:
        taken.append(next(int(i) for i in jnp.argsort(-row) if int(i) not in taken))
    return jnp.asarray(taken)


def _coverage(band: FeasibleBand, spectra: Array, concentrations: Array) -> tuple[float, float]:
    """Fraction of true amounts and of true spectrum values inside the band."""
    p = _match(band.spectra_draws.mean(axis=0), spectra)
    c_in = (concentrations >= band.concentration_lower[..., p]) & (
        concentrations <= band.concentration_upper[..., p]
    )
    s_in = (spectra >= band.spectra_lower[p]) & (spectra <= band.spectra_upper[p])
    return float(c_in.mean()), float(s_in.mean())


def _flat_coverage(band: FeasibleBand, spectra: Array, concentrations: Array) -> float:
    """Fraction of true amounts inside the central 95% of the flat draws."""
    p = _match(band.spectra_draws.mean(axis=0), spectra)
    lo, hi = jnp.percentile(band.concentration_draws, jnp.array([2.5, 97.5]), axis=0)
    inside = (concentrations >= lo[..., p]) & (concentrations <= hi[..., p])
    return float(inside.mean())


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dataset", ["realistic", "harder"])
def test_band_contains_the_truth(dataset: str, seed: int) -> None:
    """Measured 1.000 / 1.000 on all 16 port runs (8 seeds x 2 sets); the bench, 1.00."""
    data, spectra, concentrations = _bench(seed, LIFT[dataset])
    band = resolve_band(data, jax.random.key(seed))
    amounts, values = _coverage(band, spectra, concentrations)
    assert amounts >= 0.95, f"band holds {amounts:.3f} of the true amounts"
    assert values >= 0.95, f"band holds {values:.3f} of the true spectrum values"


@pytest.mark.parametrize("seed", [0, 1])
def test_band_width_matches_the_bench(seed: int) -> None:
    """Median amount band: 4.47 / 4.61 uM measured, 4.5 uM in the bench report."""
    data, _, _ = _bench(seed, LIFT["harder"])
    band = resolve_band(data, jax.random.key(seed))
    width = float(jnp.median(band.concentration_upper - band.concentration_lower))
    assert width == pytest.approx(4.5, abs=0.4)


@pytest.mark.parametrize("chain_seed", [1, 11])
def test_flat_draws_cover_an_interior_truth(chain_seed: int) -> None:
    """Regression: chain key 1 used to land in a gap of the region and stay there.

    Before the shrinkage step, 89% of its draws were infeasible and the flat summary
    covered 0.25 of the amounts at 32000 iterations. Measured after: 1.000 for both keys.
    """
    data, spectra, concentrations = _bench(1, LIFT["harder"])
    band = resolve_band(data, jax.random.key(chain_seed), n_iter=32000)
    assert _flat_coverage(band, spectra, concentrations) >= 0.9


def test_a_run_lacking_a_species_narrows_the_band() -> None:
    """Shared spectra help when a run changes which species are present.

    A second run starting from ``b`` has no ``a`` at any time, and the non-negativity
    of that zero cuts the region. Measured on exactly this data: 4.51 -> 3.08 uM,
    coverage 1.000 (figure: plans/002-feasible-band/figures/band_port.png). A second run
    that only changes the rates does not help (4.49 -> 4.40 uM, spec section 5).
    """
    _, spectra, _ = make_realistic_dataset(jax.random.key(0), noise=0.0)
    spectra = spectra + LIFT["harder"]
    time = jnp.linspace(0.0, 10.0, 30)[1:]
    from_a = _profiles(0.6, 0.4, time)
    b_decay = jnp.exp(-0.4 * time)
    from_b = 12.5 * jnp.stack([jnp.zeros_like(time), b_decay, 1.0 - b_decay], axis=-1)

    def width(concentrations: Array, initial: Array) -> tuple[float, float]:
        data = _dataset(jax.random.key(5), spectra, concentrations, initial)
        band = resolve_band(data, jax.random.key(2))
        amounts, _ = _coverage(band, spectra, concentrations)
        first_run = band.concentration_upper[0] - band.concentration_lower[0]
        return float(jnp.median(first_run)), amounts

    one, one_cov = width(from_a[None], jnp.array([[12.5, 0.0, 0.0]]))
    two, two_cov = width(jnp.stack([from_a, from_b]), jnp.array([[12.5, 0, 0], [0, 12.5, 0.0]]))
    assert two < 0.8 * one, f"band {one:.2f} -> {two:.2f} uM; expected a clear narrowing"
    assert min(one_cov, two_cov) >= 0.95
```

- [ ] **Step 2: Run the gate**

Run: `uv run pytest tests/test_band_coverage.py -q`
Expected: PASS, 9 tests, about 20 s. These pass on Task 2's code as written. If any
fails, stop: that means Task 2's code diverged from the plan, or the platform changed
the numerics. Compare against spec §5 before changing anything.

- [ ] **Step 3: Full gate**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: all clean. The two v0 `xfail`s in `tests/test_recovery.py` still show as
`xfailed`.

- [ ] **Step 4: Record what the gate showed**

Replace the placeholder line under `## 8. Gate result` in
[`spec.md`](spec.md) with the measured coverage, widths and wall clock from
`uv run pytest tests/test_band_coverage.py -q --durations=0`, and one sentence each on
anything that differed from spec §5.

- [ ] **Step 5: Commit**

```bash
git add tests/test_band_coverage.py plans/002-feasible-band/spec.md
git commit -m "test: the band contains the synthetic truth; record the gate"
```

---

### Task 4: JASCO readers

Independent of Tasks 1–3.

**Files:**
- Create: `src/spectrahandler/jasco.py`
- Test: `tests/test_jasco.py`

**Interfaces:**
- Consumes: the fixtures in `tests/data/` and the `data_dir` fixture from `conftest.py`.
- Produces: `Scan(wavelength_nm, time_min, absorbance, metadata)` (frozen dataclass,
  NumPy arrays, absorbance `(n_time, n_wavelength)`); `read_interval_scan(path) -> Scan`;
  `read_spectrum(path) -> tuple[ndarray, ndarray, dict[str, str]]`;
  `read_spectrum_series(paths) -> Scan`. Import path `spectrahandler.jasco`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_jasco.py
"""The JASCO readers against the committed fixtures and their documented traps."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from spectrahandler.jasco import read_interval_scan, read_spectrum, read_spectrum_series

PROBE_A = "probe_a/20260826_Probe_a.csv"
PROBE_C = "probe_a/20260826_Probe_c_titancitrate-only.csv"
BUFFER = "probe_a/20260827_Puffer_background.csv"


def test_interval_scan_shape_and_order(data_dir: Path) -> None:
    scan = read_interval_scan(data_dir / PROBE_A)
    assert scan.absorbance.shape == (60, 551)
    assert scan.wavelength_nm[0] == 250.0
    assert scan.wavelength_nm[-1] == 800.0
    assert bool(np.all(np.diff(scan.wavelength_nm) > 0))


def test_interval_scan_values_follow_their_wavelength(data_dir: Path) -> None:
    """File row ``800,0.0172124,...``: A(800 nm, first time) after sorting and transposing."""
    scan = read_interval_scan(data_dir / PROBE_A)
    assert scan.absorbance[0, -1] == pytest.approx(0.0172124)


def test_interval_scan_keeps_recorded_minutes(data_dir: Path) -> None:
    """Probe c starts 2.783 min after Probe a; never rebuilt as a uniform grid."""
    scan = read_interval_scan(data_dir / PROBE_C)
    assert scan.time_min[0] == pytest.approx(2.78333)
    assert 570 < scan.time_min[-1] < 600


def test_header_is_kept(data_dir: Path) -> None:
    assert read_interval_scan(data_dir / PROBE_A).metadata["YUNITS"] == "ABSORBANCE"


def test_single_spectrum_stops_before_footer(data_dir: Path) -> None:
    """The buffer reads about -0.194 AU at 250 nm: the deep-UV instrument artefact."""
    wavelength_nm, absorbance, header = read_spectrum(data_dir / BUFFER)
    assert wavelength_nm.shape == absorbance.shape
    assert wavelength_nm[0] == 250.0
    assert absorbance[0] == pytest.approx(-0.194, abs=0.001)
    assert "TITLE" in header


def test_series_is_ordered_by_time_not_name(data_dir: Path) -> None:
    """Lexical order would put 15min before 1h before 20h before 2h."""
    scan = read_spectrum_series(sorted((data_dir / "1a").glob("*.csv")))
    np.testing.assert_array_equal(scan.time_min, [0, 15, 30, 60, 120, 240, 1200])
    assert scan.absorbance.shape == (7, 801)
    assert scan.wavelength_nm[0] == 300.0


def test_series_shows_saturation(data_dir: Path) -> None:
    """``1a`` saturates at 10 AU; the reader must not hide it."""
    scan = read_spectrum_series(sorted((data_dir / "1a").glob("*.csv")))
    assert float(scan.absorbance.max()) == pytest.approx(10.0)


def test_missing_marker_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not_jasco.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="XYDATA"):
        read_interval_scan(path)


def test_series_needs_a_time_in_every_name(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="acquisition time"):
        read_spectrum_series([data_dir / BUFFER])


def test_series_rejects_two_files_at_one_time(data_dir: Path, tmp_path: Path) -> None:
    shutil.copy(data_dir / "1a" / "1a_1h.csv", tmp_path / "1a_1h.csv")
    shutil.copy(data_dir / "1a" / "1a_1h.csv", tmp_path / "1a_60min.csv")
    with pytest.raises(ValueError, match="same acquisition time"):
        read_spectrum_series(sorted(tmp_path.glob("*.csv")))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_jasco.py -q`
Expected: collection error — `No module named 'spectrahandler.jasco'`.

- [ ] **Step 3: Write `jasco.py`**

```python
# src/spectrahandler/jasco.py
"""Readers for JASCO V-730 CSV exports.

Two layouts, both documented with their traps in ``tests/data/README.md``: an interval
scan (one file, every timepoint) and a single spectrum (one file per timepoint, time in
the filename). Readers return the file's numbers unchanged apart from sorting wavelength
ascending; no trimming, blanking or resampling happens here.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["Scan", "read_interval_scan", "read_spectrum", "read_spectrum_series"]

type _Floats = npt.NDArray[np.float64]

#: Filename suffix such as ``_15min`` or ``_2h``. Units are mixed within one series.
_TIME_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)(min|h)$")
_MINUTES_PER = {"min": 1.0, "h": 60.0}


@dataclass(frozen=True)
class Scan:
    """One run as recorded: absorbance over time and wavelength.

    Attributes:
        wavelength_nm: Strictly increasing grid, shape ``(n_wavelength,)``, in nm.
        time_min: Acquisition times as recorded, shape ``(n_time,)``, in minutes. Not
            rebuilt as a uniform grid; runs from one cell changer do not share an origin.
        absorbance: Shape ``(n_time, n_wavelength)``, in absorbance units.
        metadata: The ``KEY,value`` header of the (first) file, values stripped.
    """

    wavelength_nm: _Floats
    time_min: _Floats
    absorbance: _Floats
    metadata: dict[str, str]


def _split(path: Path) -> tuple[dict[str, str], list[list[str]]]:
    """Header as a dict, and the data rows after ``XYDATA`` up to the first blank line."""
    lines = path.read_text(errors="replace").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "XYDATA")
    except StopIteration:
        raise ValueError(f"{path}: no XYDATA marker; not a JASCO CSV export") from None
    header = {}
    for line in lines[:start]:
        key, _, value = line.partition(",")
        header[key.strip()] = value.strip()
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        rows.append(line.strip().split(","))
    return header, rows


def _ascending(wavelength_nm: _Floats, absorbance: _Floats) -> tuple[_Floats, _Floats]:
    """Sort the wavelength axis (last axis of ``absorbance``) ascending; JASCO counts down."""
    order = np.argsort(wavelength_nm)
    wavelength_nm, absorbance = wavelength_nm[order], absorbance[..., order]
    if not np.all(np.diff(wavelength_nm) > 0):
        raise ValueError("duplicate wavelengths in export")
    return wavelength_nm, absorbance


def read_interval_scan(path: str | Path) -> Scan:
    """Read a JASCO interval-scan export: every timepoint of one run in one file.

    The first data row holds the acquisition times **in minutes** behind an empty first
    field; every later row is ``wavelength, A(t0), A(t1), ...``.

    Args:
        path: The ``.csv`` export.

    Returns:
        The run, wavelength ascending, absorbance as ``(n_time, n_wavelength)``.

    Raises:
        ValueError: If the file has no ``XYDATA`` marker or a row has the wrong length.
    """
    path = Path(path)
    header, rows = _split(path)
    time_min = np.array([float(v) for v in rows[0][1:] if v.strip()])
    body = rows[1:]
    if any(len(r) != len(time_min) + 1 for r in body):
        raise ValueError(f"{path}: a data row does not have one value per timepoint")
    table = np.array(body, dtype=np.float64)
    wavelength_nm, absorbance = _ascending(table[:, 0], table[:, 1:].T)
    return Scan(wavelength_nm, time_min, absorbance, header)


def _time_from_name(path: Path) -> float:
    """Acquisition time in minutes from a name such as ``1a_15min`` or ``1a_2h``."""
    match = _TIME_IN_NAME.search(path.stem)
    if match is None:
        raise ValueError(f"{path.name}: no acquisition time like '_15min' or '_2h' in name")
    return float(match.group(1)) * _MINUTES_PER[match.group(2)]


def read_spectrum(path: str | Path) -> tuple[_Floats, _Floats, dict[str, str]]:
    """Read one JASCO single-spectrum export, e.g. a blank or a reference scan.

    Two columns, ``wavelength,absorbance``, then an instrument footer after a blank line.

    Args:
        path: The ``.csv`` export.

    Returns:
        Wavelength in nm ascending with shape ``(n_wavelength,)``, absorbance with the
        same shape, and the ``KEY,value`` header.

    Raises:
        ValueError: If the file has no ``XYDATA`` marker or the rows are not two columns.
    """
    path = Path(path)
    header, rows = _split(path)
    if any(len(r) != 2 for r in rows):
        raise ValueError(f"{path}: expected two columns, wavelength and absorbance")
    table = np.array(rows, dtype=np.float64)
    wavelength_nm, absorbance = _ascending(table[:, 0], table[:, 1])
    return wavelength_nm, absorbance, header


def read_spectrum_series(paths: Sequence[str | Path]) -> Scan:
    """Read single-spectrum exports into one run, ordered by the time in each filename.

    Each file is two columns, ``wavelength,absorbance``, followed by an instrument
    footer after a blank line. The acquisition time is only in the filename, with mixed
    units (``_15min``, ``_1h``); files are ordered by that time, never by name.

    Args:
        paths: One export per timepoint, in any order. All on the same wavelength grid.

    Returns:
        The run, wavelength ascending, with ``metadata`` from the earliest file.

    Raises:
        ValueError: If a filename carries no time, two files share a time, or the
            wavelength grids differ.
    """
    files = sorted((Path(p) for p in paths), key=_time_from_name)
    time_min = np.array([_time_from_name(p) for p in files])
    if len(set(time_min.tolist())) != len(time_min):
        raise ValueError("two files carry the same acquisition time")
    spectra = [read_spectrum(p) for p in files]
    grid = spectra[0][0]
    if any(w.shape != grid.shape or not np.array_equal(w, grid) for w, _, _ in spectra):
        raise ValueError("spectra are not on one wavelength grid")
    absorbance = np.stack([a for _, a, _ in spectra])
    return Scan(grid, time_min, absorbance, spectra[0][2])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_jasco.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Full gate, then commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
git add src/spectrahandler/jasco.py tests/test_jasco.py
git commit -m "feat: read JASCO interval-scan and single-spectrum exports"
```

---

### Task 5: Is the number of species right? Rank and noise diagnostics

Every split in the band reproduces the data through the same best rank-K fit, so the fit
cannot tell splits apart — but it can tell whether K is right. This task reports the
singular values against the white-noise edge, the residual after K components, and the
residual's autocorrelation, and draws them.

**Files:**
- Create: `src/spectrahandler/curve_resolution/diagnostics.py`,
  `src/spectrahandler/plot.py`
- Modify: `src/spectrahandler/curve_resolution/__init__.py`, `pyproject.toml`, `uv.lock`
- Test: `tests/test_diagnostics.py`, `tests/test_plot.py`

**Interfaces:**
- Consumes: `SpectralDataset`; `read_interval_scan` (Task 4) in one test.
- Produces:
  - `NoiseDiagnostics` (frozen dataclass): `singular_values (n_singular,)`,
    `sigma_by_rank (n_singular,)`, `rank: int`, `sigma: float`, `noise_edge: float`,
    `n_above_noise: int`, `residual (n_run, n_time, n_wavelength)`,
    `autocorrelation_wavelength (max_lag + 1,)`, `autocorrelation_time (max_lag + 1,)`.
  - `noise_diagnostics(dataset: SpectralDataset, rank: int | None = None, *,
    max_lag: int = 10) -> NoiseDiagnostics`, exported from
    `spectrahandler.curve_resolution`.
  - `spectrahandler.plot.plot_noise(diagnostics: NoiseDiagnostics, dataset:
    SpectralDataset) -> matplotlib.figure.Figure`. matplotlib is imported inside the
    function, so the library imports without it.

**Measured before this plan was written** (spec §5, figures in `figures/`):

| Data | above the edge | lag-1 autocorrelation, wavelength / time |
| --- | --- | --- |
| synthetic, white noise | 3 | −0.07 / −0.09 |
| synthetic + drifting offset | 4 | 0.19 / 0.13 |
| real Probe a, ≥ 340 nm, rank 6 | 10 | 0.96 / 0.44 |

- [ ] **Step 1: Add matplotlib as an optional extra and a dev dependency**

```bash
uv add --optional plot matplotlib
uv add --dev matplotlib
```

Expected: `pyproject.toml` gains `[project.optional-dependencies] plot = ["matplotlib>=…"]`
and `matplotlib` in the `dev` group; `uv.lock` updates.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_diagnostics.py
"""Rank and noise diagnostics: do they tell noise from structure?"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from spectrahandler.curve_resolution import (
    SpectralDataset,
    make_realistic_dataset,
    noise_diagnostics,
)
from spectrahandler.jasco import read_interval_scan


def _with_absorbance(dataset: SpectralDataset, absorbance: jax.Array) -> SpectralDataset:
    return SpectralDataset.create(
        absorbance=absorbance,
        time=dataset.time,
        wavelength=dataset.wavelength,
        species=dataset.species,
        initial_state=dataset.initial_state,
        run_ids=dataset.run_ids,
    )


def test_white_noise_reads_as_three_components() -> None:
    """Measured: 3 above the edge, sigma 1.98e-3, lag-1 autocorrelation -0.07 / -0.09."""
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    diag = noise_diagnostics(data)
    assert diag.rank == 3
    assert diag.n_above_noise == 3
    assert diag.sigma == pytest.approx(0.002, rel=0.1)
    assert diag.sigma == pytest.approx(float(diag.sigma_by_rank[3]))
    assert float(diag.autocorrelation_wavelength[0]) == pytest.approx(1.0)
    assert abs(float(diag.autocorrelation_wavelength[1])) < 0.15
    assert abs(float(diag.autocorrelation_time[1])) < 0.15
    assert diag.residual.shape == data.absorbance.shape


def test_an_unmodelled_offset_shows_as_a_fourth_component() -> None:
    """A flat offset drifting over time is one more component. Measured: 4 above."""
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    drift = 0.01 * jnp.sin(jnp.linspace(0.0, 3.0, data.n_time))[None, :, None]
    diag = noise_diagnostics(_with_absorbance(data, data.absorbance + drift))
    assert diag.n_above_noise == 4


def test_interpolated_export_is_not_white(data_dir: Path) -> None:
    """The trap in tests/data/README.md: the JASCO export is interpolated along wavelength.

    Measured on Probe a (>= 340 nm, rank 6): lag-1 autocorrelation 0.96 along wavelength.
    White noise would read about 0, so the rank-residual sigma cannot be trusted here.
    """
    scan = read_interval_scan(data_dir / "probe_a" / "20260826_Probe_a.csv")
    keep = scan.wavelength_nm >= 340
    data = SpectralDataset.create(
        absorbance=jnp.asarray(scan.absorbance[None][..., keep]),
        time=jnp.asarray(scan.time_min[None] / 60),
        wavelength=jnp.asarray(scan.wavelength_nm[keep]),
        species=("cob1", "cob2", "co3", "mecbl", "ti3", "tiox"),
        initial_state=jnp.zeros((1, 6)),
        run_ids=("a",),
    )
    diag = noise_diagnostics(data)
    assert float(diag.autocorrelation_wavelength[1]) > 0.9


def test_rank_must_leave_room_for_noise() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    with pytest.raises(ValueError, match="rank"):
        noise_diagnostics(data, rank=30)


def test_requires_fully_measured_runs() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    masked = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        mask=data.mask.at[0, 3].set(False),
    )
    with pytest.raises(NotImplementedError, match="fully measured"):
        noise_diagnostics(masked)
```

```python
# tests/test_plot.py
"""The noise figure renders. What it shows is pinned by test_diagnostics.py."""

import jax
import matplotlib

matplotlib.use("Agg")

from spectrahandler.curve_resolution import make_realistic_dataset, noise_diagnostics
from spectrahandler.plot import plot_noise


def test_plot_noise_has_three_panels() -> None:
    data, _, _ = make_realistic_dataset(jax.random.key(0))
    fig = plot_noise(noise_diagnostics(data), data)
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert len(titles) == 3
    assert titles[0].startswith("what the data hold")
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_diagnostics.py tests/test_plot.py -q`
Expected: collection errors — `cannot import name 'noise_diagnostics'` and
`No module named 'spectrahandler.plot'`.

- [ ] **Step 4: Write `diagnostics.py`**

```python
# src/spectrahandler/curve_resolution/diagnostics.py
"""Is the chosen number of species enough? Rank and noise diagnostics.

Every split the band reports reproduces the data through the same best rank-K fit, so
the fit cannot tell splits apart. What it can tell is whether K is right: if the residual
after K components is white noise at one level, K components are all the data hold. A
structured residual means something is missing -- a species, a baseline, a per-run
offset -- or that the noise is not what the band assumes.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset

__all__ = ["NoiseDiagnostics", "noise_diagnostics"]


@dataclass(frozen=True)
class NoiseDiagnostics:
    """What is left after keeping ``rank`` components, and whether it looks like noise.

    Attributes:
        singular_values: Of the stacked time course, shape ``(n_singular,)``, in
            absorbance units, largest first. ``n_singular = min(n_run * n_time,
            n_wavelength)``.
        sigma_by_rank: Noise standard deviation implied by keeping ``r`` components,
            for ``r = 0 .. n_singular - 1``, in absorbance units.
        rank: Components kept for ``residual``.
        sigma: ``sigma_by_rank[rank]``.
        noise_edge: The largest singular value white noise at ``sigma`` would produce
            in a matrix of this shape, ``sigma * (sqrt(n_row) + sqrt(n_wavelength))``.
            Singular values above it carry signal.
        n_above_noise: Number of singular values above ``noise_edge``.
        residual: Data minus the rank-``rank`` fit, shape
            ``(n_run, n_time, n_wavelength)``, in absorbance units.
        autocorrelation_wavelength: Residual autocorrelation along wavelength for lags
            ``0 .. max_lag``, shape ``(max_lag + 1,)``. White noise: 1, then ~0.
        autocorrelation_time: Same along time, within each run.
    """

    singular_values: Array
    sigma_by_rank: Array
    rank: int
    sigma: float
    noise_edge: float
    n_above_noise: int
    residual: Array
    autocorrelation_wavelength: Array
    autocorrelation_time: Array


def _autocorrelation(x: Array, axis: int, max_lag: int) -> Array:
    """Autocorrelation of ``x`` along ``axis`` for lags ``0 .. max_lag``, pooled."""
    x = jnp.moveaxis(x, axis, -1)
    energy = (x**2).sum()
    # ponytail: a Python loop over lag values (max_lag ~ 10), not over data.
    return jnp.stack(
        [energy / energy]
        + [(x[..., :-lag] * x[..., lag:]).sum() / energy for lag in range(1, max_lag + 1)]
    )


def noise_diagnostics(
    dataset: SpectralDataset, rank: int | None = None, *, max_lag: int = 10
) -> NoiseDiagnostics:
    """Singular values, residual and residual autocorrelation after ``rank`` components.

    Args:
        dataset: Validated observations; ``mask`` must be all True.
        rank: Components to keep. ``None`` keeps ``dataset.n_species``.
        max_lag: Largest lag of the residual autocorrelations, in channels and in
            timepoints.

    Returns:
        The diagnostics; :func:`spectrahandler.plot.plot_noise` draws them.

    Raises:
        NotImplementedError: If ``dataset.mask`` has any False entry.
        ValueError: If ``rank`` leaves no degrees of freedom for the noise.
    """
    if not bool(dataset.mask.all()):
        raise NotImplementedError("noise_diagnostics requires fully measured runs")
    rank = dataset.n_species if rank is None else rank
    n_row = dataset.n_run * dataset.n_time
    n_singular = min(n_row, dataset.n_wavelength)
    if not 0 <= rank < n_singular:
        raise ValueError(f"rank must be in [0, {n_singular}), got {rank}")

    data = dataset.absorbance.reshape(n_row, dataset.n_wavelength)
    u, s, vt = jnp.linalg.svd(data, full_matrices=False)
    kept = jnp.arange(n_singular)
    discarded = jnp.cumsum((s**2)[::-1])[::-1]
    dof = (n_row - kept) * (dataset.n_wavelength - kept)
    sigma_by_rank = jnp.sqrt(discarded / dof)
    sigma = float(sigma_by_rank[rank])
    edge = sigma * (n_row**0.5 + dataset.n_wavelength**0.5)

    residual = (data - (u[:, :rank] * s[:rank]) @ vt[:rank]).reshape(dataset.absorbance.shape)
    return NoiseDiagnostics(
        singular_values=s,
        sigma_by_rank=sigma_by_rank,
        rank=rank,
        sigma=sigma,
        noise_edge=edge,
        n_above_noise=int((s > edge).sum()),
        residual=residual,
        autocorrelation_wavelength=_autocorrelation(residual, 2, max_lag),
        autocorrelation_time=_autocorrelation(residual, 1, max_lag),
    )
```

In `src/spectrahandler/curve_resolution/__init__.py`, add after the `dataset` import:

```python
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics, noise_diagnostics
```

and add `"NoiseDiagnostics"` and `"noise_diagnostics"` to `__all__`, keeping it sorted.

- [ ] **Step 5: Write `plot.py`**

Colours: the reference palette's diverging pair (blue / neutral grey / red) for the
residual, so zero reads as nothing; signal points in the first categorical slot.

```python
# src/spectrahandler/plot.py
"""Figures for checking a resolution by eye. Needs the ``plot`` extra (matplotlib)."""

from typing import TYPE_CHECKING

import numpy as np

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_noise"]

#: Diverging pair with a neutral midpoint: residuals below zero blue, above red.
_DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
_INK, _MUTED, _SIGNAL, _NOISE = "#0b0b0b", "#52514e", "#2a78d6", "#a3a29b"


def plot_noise(diagnostics: NoiseDiagnostics, dataset: SpectralDataset) -> "Figure":
    """Three panels: what the data hold, what is left, and whether what is left is noise.

    Left, the singular values against the white-noise edge: those above it carry
    signal. Middle, the residual after ``diagnostics.rank`` components over time and
    wavelength, scaled to +-3 sigma: white noise looks like even static, structure
    means something is missing. Right, the residual autocorrelation along wavelength
    and time: white noise drops to about zero after lag 0.

    Args:
        diagnostics: From :func:`spectrahandler.curve_resolution.noise_diagnostics`.
        dataset: The data the diagnostics were computed from, for the axes.

    Returns:
        A matplotlib figure with three axes.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as err:
        raise ImportError(
            "plot_noise needs matplotlib: install the 'plot' extra, "
            "e.g. uv add 'spectrahandler[plot]'"
        ) from err

    s = np.asarray(diagnostics.singular_values)
    shown = min(len(s), max(20, 3 * diagnostics.rank))
    fig, (ax_sv, ax_res, ax_ac) = plt.subplots(
        1, 3, figsize=(13, 3.8), width_ratios=(1, 1.5, 1), constrained_layout=True
    )

    index = np.arange(1, shown + 1)
    signal = s[:shown] > diagnostics.noise_edge
    ax_sv.semilogy(index, s[:shown], color=_MUTED, lw=1, zorder=1)
    ax_sv.scatter(
        index[signal],
        s[:shown][signal],
        s=28,
        color=_SIGNAL,
        zorder=2,
        label=f"signal ({int(signal.sum())})",
    )
    ax_sv.scatter(index[~signal], s[:shown][~signal], s=28, color=_NOISE, zorder=2, label="noise")
    ax_sv.axhline(diagnostics.noise_edge, color=_INK, lw=1, ls="--", label="white-noise edge")
    ax_sv.axvline(diagnostics.rank + 0.5, color=_MUTED, lw=0.8, ls=":")
    ax_sv.set(
        xlabel="component",
        ylabel="singular value (AU)",
        title=f"what the data hold\n{diagnostics.n_above_noise} components above noise; "
        f"rank {diagnostics.rank} kept",
    )
    ax_sv.legend(frameon=False, fontsize=8)

    residual = np.asarray(diagnostics.residual).reshape(-1, dataset.n_wavelength)
    limit = 3 * diagnostics.sigma
    cmap = LinearSegmentedColormap.from_list("residual", _DIVERGING)
    wl = np.asarray(dataset.wavelength)
    image = ax_res.imshow(
        residual,
        aspect="auto",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        extent=(wl[0], wl[-1], residual.shape[0] - 0.5, -0.5),
        interpolation="nearest",
    )
    for boundary in range(1, dataset.n_run):
        ax_res.axhline(boundary * dataset.n_time - 0.5, color=_INK, lw=0.8)
    fig.colorbar(image, ax=ax_res, label="residual (AU)")
    ax_res.set(
        xlabel=f"wavelength ({dataset.wavelength_unit})",
        ylabel="timepoint (runs stacked)",
        title=f"what is left after {diagnostics.rank} components\n"
        f"sigma = {diagnostics.sigma:.2g} AU, colour scale ±3 sigma",
    )

    lags = np.arange(len(diagnostics.autocorrelation_wavelength))
    ax_ac.axhline(0, color=_MUTED, lw=0.8)
    ax_ac.plot(
        lags,
        np.asarray(diagnostics.autocorrelation_wavelength),
        marker="o",
        ms=4,
        color=_DIVERGING[0],
        lw=1.5,
        label="along wavelength",
    )
    ax_ac.plot(
        lags,
        np.asarray(diagnostics.autocorrelation_time),
        marker="s",
        ms=4,
        color=_DIVERGING[2],
        lw=1.5,
        label="along time",
    )
    ax_ac.set(
        xlabel="lag (channels or timepoints)",
        ylabel="autocorrelation",
        ylim=(-0.3, 1.05),
        title="is what is left noise?\nwhite noise: 1 at lag 0, then about 0",
    )
    ax_ac.legend(frameon=False, fontsize=8)

    for ax in (ax_sv, ax_res, ax_ac):
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", fontsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig
```

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py tests/test_plot.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 7: Full gate, then commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
git add src/spectrahandler tests/test_diagnostics.py tests/test_plot.py pyproject.toml uv.lock
git commit -m "feat: rank and noise diagnostics, with an optional noise figure"
```

---

### Task 6: Retire v0

Requires Task 3 to pass and Task 5 to be done. Deletion only; no behaviour is added.

**Files:**
- Delete: `src/spectrahandler/curve_resolution/model.py`,
  `src/spectrahandler/curve_resolution/inference.py`, `tests/test_model.py`,
  `tests/test_recovery.py`
- Modify: `src/spectrahandler/curve_resolution/__init__.py`, `tests/conftest.py`,
  `src/spectrahandler/curve_resolution/synthetic.py` (docstrings)

**Interfaces:**
- Consumes: Task 2's exports.
- Produces: `spectrahandler.curve_resolution` exports exactly `FeasibleBand`,
  `NoiseDiagnostics`, `SpectralDataset`, `make_easy_dataset`, `make_realistic_dataset`,
  `noise_diagnostics`, `resolve_band`.
  `fit` and `curve_resolution_model` are gone.

- [ ] **Step 1: Delete the v0 files**

```bash
git rm src/spectrahandler/curve_resolution/model.py src/spectrahandler/curve_resolution/inference.py tests/test_model.py tests/test_recovery.py
```

- [ ] **Step 2: Rewrite `__init__.py`**

```python
# src/spectrahandler/curve_resolution/__init__.py
"""Curve resolution: pure spectra and concentration profiles, with the band the data allow."""

from spectrahandler.curve_resolution.band import FeasibleBand, resolve_band
from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics, noise_diagnostics
from spectrahandler.curve_resolution.synthetic import make_easy_dataset, make_realistic_dataset

__all__ = [
    "FeasibleBand",
    "NoiseDiagnostics",
    "SpectralDataset",
    "make_easy_dataset",
    "make_realistic_dataset",
    "noise_diagnostics",
    "resolve_band",
]
```

- [ ] **Step 3: Drop NumPyro from `tests/conftest.py`**

Delete these lines; nothing samples with multiple chains any more:

```python
import numpyro
```

```python
# Must precede any JAX backend initialisation, or num_chains=2 silently falls back to
# running the chains one after the other.
numpyro.set_host_device_count(2)
```

- [ ] **Step 4: Update the `synthetic.py` docstrings that name v0 or plan 001**

In the module docstring, replace
`fixture is reachable, and because coverage checks reuse it. See ``spec.md`` §5.`
with `fixture is reachable, and because coverage checks reuse it.`

In `make_easy_dataset`, replace
`is what v0 is validated against. Nothing here is meant to be difficult: failure on`
with `is a smoke test for any method. Nothing here is meant to be difficult: failure on`.

In `make_realistic_dataset`, replace
`exactly what the v0 model assumes. Spectra of a and c overlap, so some rotational`
with `exactly what resolve_band assumes. Spectra of a and c overlap, so some rotational`.

In `src/spectrahandler/curve_resolution/dataset.py`, change the module docstring's
first line from `"""The one data structure inference operates on.` to
`"""The one data structure curve resolution operates on.`

- [ ] **Step 5: Verify nothing references v0**

Run: `grep -rn -e "numpyro" -e "curve_resolution_model" -e "import fit" -e "v0" src tests`
Expected: no output.

- [ ] **Step 6: Full gate, then commit**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: all clean, no `xfailed` left (77 passed when this plan was dry-run).

```bash
git add -A src tests
git commit -m "refactor: retire the v0 random-walk model and NUTS fit (ADR 0003)"
```

`numpyro` stays in `pyproject.toml` for the kinetic stage; see spec §7 for the open
question.

---

### Task 7: Documentation

**Files:**
- Create: `docs/src/content/docs/guides/curve-resolution.md`
- Modify: `docs/src/content/docs/guides/quickstart.md`

**Interfaces:**
- Consumes: the public API after Task 5.
- Produces: the user-facing page for `resolve_band`.

- [ ] **Step 1: Write the guide**

````markdown
---
title: Curve resolution with honest uncertainty
description: Resolve a reaction time course into pure spectra and concentration profiles, and report the full range the data allow.
sidebar:
  order: 3
---

A UV/Vis time course is a mixture: at every time, the measured absorbance is the sum of
each species' concentration times its pure spectrum. Curve resolution splits the data
back into those two parts.

## Why you get a band, not a single answer

The data fix the **product** of concentrations and spectra, but usually not the split.
Mixing a little of one spectrum into another, and adjusting the concentrations to
match, reproduces the data exactly, and both spectra stay smooth and positive. No
amount of repeating the same measurement removes this; it is a property of the
experiment, not of the noise.

SpectraHandler therefore reports the **band**: for every concentration and every
spectrum value, the full range over all splits that fit the data, widened by the
measurement noise. The band contains the truth whatever the truth is, so its width is
an honest statement of what your data can and cannot tell you.

Methods that return a single split (classic MCR-ALS, NMF), or a narrow Bayesian
posterior under a convenient prior, pick one point inside that band without telling
you that they did.

## What narrows the band

Only information the time course does not already contain:

- **Closure.** The species of one pool sum to a known total, e.g. 12.5 µM cobalamin.
  Taken from `initial_state`: every run sums to `initial_state[run].sum()`.
- **Reference spectra.** Every species you can measure pure pins its own spectrum.
- **Runs that change which species are present.** All runs share one set of pure
  spectra. A run in which a species is absent cuts the band markedly; a run that only
  changes the rates barely helps.

## A runnable example

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from spectrahandler.curve_resolution import (
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

# Synthetic a -> b -> c data at 12.5 uM total, with known true spectra.
data, true_spectra, _ = make_realistic_dataset(jax.random.key(0))

# Suppose species "a" was measured pure: a 50 uM scan with 0.002 AU noise, divided
# by 50, gives its spectrum per uM with noise 0.002 / 50. Other species: NaN.
reference = jnp.full((3, data.n_wavelength), jnp.nan).at[0].set(true_spectra[0])
data = SpectralDataset.create(
    absorbance=data.absorbance,
    time=data.time,
    wavelength=data.wavelength,
    species=data.species,
    initial_state=data.initial_state,
    run_ids=data.run_ids,
    reference_spectra=reference,
    reference_sigma=jnp.array([0.002 / 50, jnp.nan, jnp.nan]),
)

band = resolve_band(data, jax.random.key(1))

print(f"{band.n_free} free re-mixing numbers, noise {band.sigma:.4f} AU")

# Only "a" has a reference, so only its name is pinned: the columns for "b" and "c"
# may come back swapped (see below).
lo, hi = band.concentration_lower[0, 10, 0], band.concentration_upper[0, 10, 0]
print(f"a at t = {float(data.time[0, 10]):.1f} h: {float(lo):.2f} - {float(hi):.2f} uM")
```

On this data species `a` at 3.4 h comes out between about 0.06 and 2.19 µM; the true
value is 1.58 µM.

## Reading the result

| Field | What it is |
| --- | --- |
| `concentration_lower`, `concentration_upper` | The band for every amount, in your concentration unit |
| `spectra_lower`, `spectra_upper` | The band for every spectrum value, in absorbance per concentration unit |
| `concentration_ambiguity`, `spectra_ambiguity` | The same extremes without the noise margin: the part more of the same data would never remove |
| `concentration_draws`, `spectra_draws` | Draws spread evenly over all splits that fit, with noise. A *typical-solution* summary under a stated flat prior, not a calibrated interval |
| `n_free` | How many numbers the data leave undetermined |

Species **without** a reference are resolved only up to relabelling among themselves:
the band for "b" may be the spectrum you know as "c". Give a reference, or a run where
one of them is absent, to pin the names.

## Is the number of species right?

Every split in the band reproduces the data through the same best fit, so the fit
cannot tell splits apart. It can tell you whether the number of species is right:
after removing that many components, what is left should be plain noise.

```python
from spectrahandler.curve_resolution import noise_diagnostics
from spectrahandler.plot import plot_noise  # needs the plot extra: uv add "spectrahandler[plot]"

diagnostics = noise_diagnostics(data)
print(f"{diagnostics.n_above_noise} components above the noise edge")
plot_noise(diagnostics, data).savefig("noise.png")
```

On the example data this prints `3 components above the noise edge`, and the figure
looks like this:

![Noise diagnostics for three species: three singular values above the noise edge, a residual of even static, and autocorrelation dropping to zero after lag 0](../../../assets/noise-white.png)

- **Left, what the data hold.** Singular values above the dashed white-noise edge
  carry signal. Their count should equal your number of species.
- **Middle, what is left.** The residual after that many components, over time and
  wavelength. Even static is noise; stripes, steps or blobs are something the model
  is missing.
- **Right, is it noise?** The residual's autocorrelation. White noise is 1 at lag 0
  and about 0 after.

The same data with a slowly drifting offset added, which is one more component:

![Noise diagnostics with an unmodelled offset: four singular values above the edge and correlated residuals](../../../assets/noise-offset.png)

What to do with what you see:

- **More components above the edge than species:** an unmodelled species, a baseline,
  or a per-run offset. Each is one more component; the band assumes there are none.
- **Autocorrelation near 1 along wavelength:** the export was smoothed or interpolated
  along wavelength, as JASCO interval scans are. The noise level estimated from the
  data is then far too low; pass `sigma` measured along time on a stretch where
  nothing reacts.

## Tuning

`resolve_band` exposes its knobs with defaults that worked on synthetic data:

- `sigma`: the noise level. By default it is estimated from the data; pass a value
  measured on a blank if you have one.
- `z_slack`: how many noise standard deviations a value may dip below zero and still
  count as non-negative, and the width of the noise margin. Default 3.5.
- `n_iter`, `n_burn`, `thin`: length of the walk over the feasible region.
- `n_als`, `n_restart`, `n_search`: the search for a first split that fits.

Current limits: every run must be fully measured, the number of species is given, and
the noise is assumed to be one level everywhere.
````

- [ ] **Step 2: Render the two figures the guide shows**

From the repo root, run this with `uv run python -` (stdin):

```python
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

from spectrahandler.curve_resolution import (
    SpectralDataset,
    make_realistic_dataset,
    noise_diagnostics,
)
from spectrahandler.plot import plot_noise

data, _, _ = make_realistic_dataset(jax.random.key(0))
plot_noise(noise_diagnostics(data), data).savefig("docs/src/assets/noise-white.png", dpi=130)
drift = 0.01 * jnp.sin(jnp.linspace(0.0, 3.0, data.n_time))[None, :, None]
offset = SpectralDataset.create(
    absorbance=data.absorbance + drift,
    time=data.time,
    wavelength=data.wavelength,
    species=data.species,
    initial_state=data.initial_state,
    run_ids=data.run_ids,
)
plot_noise(noise_diagnostics(offset), offset).savefig("docs/src/assets/noise-offset.png", dpi=130)
```

Create `docs/src/assets/` first if it does not exist. Synthetic data only: the real
fixtures are unpublished and the docs site is public.

- [ ] **Step 3: Run the guide's code blocks**

Concatenate the guide's `python` blocks into a scratch file outside the repo, prefixed
with `import matplotlib; matplotlib.use("Agg")`, and run it with
`uv run --project <repo> python <file>`. Expected output:

```
4 free re-mixing numbers, noise 0.0020 AU
a at t = 3.4 h: 0.06 - 2.19 uM
3 components above the noise edge
```

- [ ] **Step 4: Point the quickstart at it**

In `docs/src/content/docs/guides/quickstart.md`, replace the caution block and
everything after it with:

````markdown
The curve-resolution workflow, with a runnable example, is in
[Curve resolution with honest uncertainty](../curve-resolution/).

Every example needs float64, set once before anything else touches JAX:

```python
import jax

jax.config.update("jax_enable_x64", True)
```
````

- [ ] **Step 5: Build the site**

Run: `cd docs && npm run build`
Expected: `[build] Complete!`, one more page than before, and two `noise-*.webp`
files under `docs/dist/_astro/`.

- [ ] **Step 6: Commit**

```bash
git add docs/src/content/docs/guides docs/src/assets
git commit -m "docs: guide to curve resolution with the feasible band"
```

---

### Task 8: Close plan 001

- [ ] **Step 1: Delete the superseded plan**

Its report and bench are in git history from Task 0's commit; ADR 0003 cites them by path.

```bash
git rm -r plans/001-bayesian-curve-resolution
git commit -m "chore: remove plan 001; superseded by plan 002 and ADR 0003"
```

---

## Self-review notes

- **Spec coverage:** §2 method → Task 2. §3 data contract → Task 1. §4 readers → Task 4.
  §1 gate and §5 evidence → Task 3. Rank and noise diagnostics (§1, §5) → Task 5. v0
  retirement → Task 6. Docs (CLAUDE.md definition of done) → Task 7. §8 gate record →
  Task 3 step 4.
- **Not covered, by design (spec §1, §7):** real Probe a/c/d resolution, QC, pattern
  smoothing, sampled `φ`, kinetics, masking, automatic rank choice, shape constraints,
  optimisation-based band edges. The next plan is written from
  the evidence of this one.
- **Verified before writing:** every code block in Tasks 1–7 was run on this branch at
  `472c946`, all tasks applied together: 77 tests passing after Task 6, ruff and ty
  clean, docs build with both figures, guide output as in Task 7 step 3. Changes made
  while transcribing: the double backticks in `read_spectrum`'s docstring (stripped by
  a shell heredoc), and the two Task 1 fragments shown as `diff`/`text` blocks —
  `ruff format .` formats Python blocks inside Markdown and mangles partial snippets,
  so no `python` block in this plan is a fragment that is not valid on its own.
- **Type consistency:** `FeasibleBand` fields and `resolve_band` keywords match between
  Task 2's Interfaces block, its code, the tests in Tasks 2–3 and the guide in Task 7;
  `NoiseDiagnostics` fields match between Task 5's Interfaces, code, tests and guide.
