# Bayesian Curve Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover pure component spectra `S` and concentration profiles `C` from a
UV/Vis reaction time course, with posteriors, and prove it by round-tripping synthetic
data where the truth is known.

**Architecture:** A frozen `SpectralDataset` carrying dense `(run, time, wavelength)`
arrays is the only input to inference. A NumPyro model puts a second-order random walk
directly over the wavelength channels of each spectrum, softplus-transforms for
non-negativity, and normalises each spectrum to fix the scale ambiguity. NUTS samples
it. No basis expansion — see `spec.md` §4.

**Tech Stack:** JAX, NumPyro, NumPy (boundaries and tests only), pytest.

**Spec:** [`spec.md`](spec.md) — read it first; this plan argues from it.

## Global Constraints

Copied from [`CLAUDE.md`](../../CLAUDE.md) and [`spec.md`](spec.md) §0. Every task's
requirements implicitly include these.

- Python ≥ 3.13. `uv run ...` for everything; never bare `python`, never `pip`.
- **Do not import SciPy.** It is present transitively via JAX.
- NumPy is allowed only at boundaries and in tests. All model maths is `jax.numpy`.
- Work on a **binned** wavelength grid (~64 channels). The random walk runs per channel,
  so the raw 1 nm grid would be needlessly expensive.
- float64: already enabled in `tests/conftest.py`. Never force it at library import.
- Full annotations on every parameter and return, including `-> None`.
- Google-style docstring on every public module, class and function.
- Line length 100. Annotate arrays as `jax.Array`; state shape and units in the docstring.
- Explicit PRNG keys threaded through. No global seed, no `numpy.random`.
- No Python loops over array axes. No in-place mutation; use `.at[].set()`.
- Numerical assertions use `pytest.approx` or `numpy.testing`, never `==`.
- The pre-commit and Claude hooks run ruff and ty on every edit. A step is not done
  until `uv run ruff check . && uv run ty check && uv run pytest` is clean.

## Preconditions

[ADR 0001](../../docs/decisions/0001-array-layout-and-canonical-ordering.md) — the array
layout and alphabetical species order every task assumes — is **accepted**. Nothing
blocks Task 0.

[ADR 0002](../../docs/decisions/0002-scope-boundary-against-mcrals.md), where data
loading lives, is **unresolved and does not block this plan**: Tasks 0–2 use synthetic
data only. Do not try to read `tests/data/` — those fixtures are deliberately
unreachable until that decision is made.

## Scope

Steps 0–3 of [`spec.md`](spec.md) §7, as two tasks — the spec's step 2 (prior
predictive) folds into the model task, since without a basis there is nothing to build
before the model itself. Step 3 is the gate: the synthetic round-trip. Nothing past it
is planned, because what step 4 should be depends on how step 3 behaves.

---

### Task 0: `SpectralDataset` and its validation

**Files:**
- Create: `src/spectrahandler/curve_resolution/__init__.py`
- Create: `src/spectrahandler/curve_resolution/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SpectralDataset` (frozen dataclass) with fields `absorbance`, `time`,
  `wavelength`, `mask`, `species`, `initial_state`, `reference_spectra`,
  `reference_sigma`, `run_ids`, `time_unit`, `wavelength_unit`, `concentration_unit`;
  properties `n_run`, `n_time`, `n_wavelength`, `n_species`; classmethod
  `SpectralDataset.create(...) -> SpectralDataset` which canonicalises species order
  then validates.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
"""Construction invariants for SpectralDataset."""

import jax.numpy as jnp
import pytest
from jax import Array

from spectrahandler.curve_resolution import SpectralDataset


def _minimal(**overrides: object) -> SpectralDataset:
    """A valid 1-run, 3-time, 4-wavelength, 2-species dataset."""
    kwargs: dict[str, object] = {
        "absorbance": jnp.zeros((1, 3, 4)),
        "time": jnp.arange(3.0).reshape(1, 3),
        "wavelength": jnp.arange(4.0),
        "species": ("b", "a"),
        "initial_state": jnp.array([[10.0, 20.0]]),
        "run_ids": ("r0",),
    }
    kwargs.update(overrides)
    return SpectralDataset.create(**kwargs)  # type: ignore[arg-type]


def test_species_are_sorted_and_columns_follow() -> None:
    ds = _minimal()
    assert ds.species == ("a", "b")
    # initial_state was given in ("b", "a") order as [10, 20]; it must be permuted.
    assert ds.initial_state.tolist() == [[20.0, 10.0]]


def test_shape_properties() -> None:
    ds = _minimal()
    assert (ds.n_run, ds.n_time, ds.n_wavelength, ds.n_species) == (1, 3, 4, 2)


def test_mask_defaults_to_all_measured() -> None:
    assert bool(_minimal().mask.all())


def test_references_default_to_nan() -> None:
    ds = _minimal()
    assert bool(jnp.isnan(ds.reference_spectra).all())
    assert bool(jnp.isnan(ds.reference_sigma).all())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"wavelength": jnp.array([0.0, 2.0, 1.0, 3.0])}, "strictly increasing"),
        ({"time": jnp.array([[0.0, 2.0, 1.0]])}, "non-decreasing"),
        ({"absorbance": jnp.full((1, 3, 4), jnp.nan)}, "finite"),
        ({"initial_state": jnp.array([[-1.0, 1.0]])}, "non-negative"),
        ({"species": ("a", "a")}, "unique"),
        ({"species": ()}, "non-empty"),
        ({"run_ids": ("r0", "r1")}, "run_ids"),
        ({"absorbance": jnp.zeros((1, 3, 5))}, "shape"),
        ({"mask": jnp.zeros((1, 3), dtype=bool)}, "at least one"),
    ],
)
def test_validation_rejects(overrides: dict[str, Array], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _minimal(**overrides)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: collection error — `No module named 'spectrahandler.curve_resolution'`.

- [ ] **Step 3: Write the implementation**

```python
# src/spectrahandler/curve_resolution/__init__.py
"""Bayesian curve resolution: spectra and concentration profiles, with posteriors."""

from spectrahandler.curve_resolution.dataset import SpectralDataset

__all__ = ["SpectralDataset"]
```

```python
# src/spectrahandler/curve_resolution/dataset.py
"""The one data structure inference operates on.

Dense arrays with leading batch axes, validated once at construction. See
``docs/decisions/0001-array-layout-and-canonical-ordering.md`` for why the axes are
ordered ``(run, time, wavelength)`` and why species are sorted alphabetically.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

__all__ = ["SpectralDataset"]


@dataclass(frozen=True)
class SpectralDataset:
    """Absorbance over reaction time for one or more runs.

    A single-run experiment is ``n_run == 1``, never a 2-D special case.

    Attributes:
        absorbance: Absorbance, shape ``(n_run, n_time, n_wavelength)``. ``NaN`` where
            ``mask`` is False; never zero, which is a valid measurement.
        time: Reaction time, shape ``(n_run, n_time)``, in ``time_unit``.
        wavelength: Shared grid, shape ``(n_wavelength,)``, strictly increasing, in
            ``wavelength_unit``.
        mask: True where measured, shape ``(n_run, n_time)``.
        species: Species names, sorted alphabetically and unique. Every array with a
            species axis follows this order.
        initial_state: Initial concentrations, shape ``(n_run, n_species)``, in
            ``concentration_unit``. Carried but unused until a kinetic model exists.
        reference_spectra: Known pure spectra, shape ``(n_species, n_wavelength)``,
            ``NaN`` for species without one. Carried but unused in v0.
        reference_sigma: Uncertainty on each reference, shape ``(n_species,)``, ``NaN``
            where absent.
        run_ids: One identifier per run.
        time_unit: Unit string for ``time``, e.g. ``"h"``.
        wavelength_unit: Unit string for ``wavelength``, e.g. ``"nm"``.
        concentration_unit: Unit string for concentrations, e.g. ``"uM"``.
    """

    absorbance: Array
    time: Array
    wavelength: Array
    mask: Array
    species: tuple[str, ...]
    initial_state: Array
    reference_spectra: Array
    reference_sigma: Array
    run_ids: tuple[str, ...]
    time_unit: str
    wavelength_unit: str
    concentration_unit: str

    @property
    def n_run(self) -> int:
        """Number of runs."""
        return self.absorbance.shape[0]

    @property
    def n_time(self) -> int:
        """Number of timepoints, including padded ones."""
        return self.absorbance.shape[1]

    @property
    def n_wavelength(self) -> int:
        """Number of wavelength channels."""
        return self.absorbance.shape[2]

    @property
    def n_species(self) -> int:
        """Number of species."""
        return len(self.species)

    def __post_init__(self) -> None:
        """Validate the contract. Raises rather than letting a sampler fail later."""
        _validate(self)

    @classmethod
    def create(
        cls,
        *,
        absorbance: Array,
        time: Array,
        wavelength: Array,
        species: tuple[str, ...],
        initial_state: Array,
        run_ids: tuple[str, ...],
        mask: Array | None = None,
        reference_spectra: Array | None = None,
        reference_sigma: Array | None = None,
        time_unit: str = "h",
        wavelength_unit: str = "nm",
        concentration_unit: str = "uM",
    ) -> "SpectralDataset":
        """Build a dataset, sorting species into canonical order first.

        Arrays with a species axis are permuted to match the sorted names, so the
        caller's ordering never leaks into the model.

        Args:
            absorbance: Shape ``(n_run, n_time, n_wavelength)``.
            time: Shape ``(n_run, n_time)``.
            wavelength: Shape ``(n_wavelength,)``, strictly increasing.
            species: Names in any order; unique.
            initial_state: Shape ``(n_run, n_species)``, in ``species`` order as given.
            run_ids: One per run.
            mask: Shape ``(n_run, n_time)``. Defaults to all measured.
            reference_spectra: Shape ``(n_species, n_wavelength)`` in ``species`` order
                as given. Defaults to all ``NaN``.
            reference_sigma: Shape ``(n_species,)``. Defaults to all ``NaN``.
            time_unit: Unit of ``time``.
            wavelength_unit: Unit of ``wavelength``.
            concentration_unit: Unit of concentrations.

        Returns:
            A validated, immutable dataset.

        Raises:
            ValueError: If any contract in the class docstring is violated.
        """
        if len(set(species)) != len(species):
            raise ValueError(f"species must be unique, got {species!r}")
        order = sorted(range(len(species)), key=lambda i: species[i])
        index = jnp.asarray(order, dtype=int)
        n_species, n_wavelength = len(species), wavelength.shape[0]

        refs = (
            jnp.full((n_species, n_wavelength), jnp.nan)
            if reference_spectra is None
            else jnp.asarray(reference_spectra)[index]
        )
        ref_sigma = (
            jnp.full((n_species,), jnp.nan)
            if reference_sigma is None
            else jnp.asarray(reference_sigma)[index]
        )
        return cls(
            absorbance=jnp.asarray(absorbance),
            time=jnp.asarray(time),
            wavelength=jnp.asarray(wavelength),
            mask=jnp.ones(time.shape, dtype=bool) if mask is None else jnp.asarray(mask),
            species=tuple(species[i] for i in order),
            initial_state=jnp.asarray(initial_state)[:, index]
            if len(species)
            else jnp.asarray(initial_state),
            reference_spectra=refs,
            reference_sigma=ref_sigma,
            run_ids=tuple(run_ids),
            time_unit=time_unit,
            wavelength_unit=wavelength_unit,
            concentration_unit=concentration_unit,
        )


def _validate(ds: SpectralDataset) -> None:
    """Check every invariant in the class docstring, with a message naming the field."""
    if not ds.species:
        raise ValueError("species must be non-empty")
    if len(set(ds.species)) != len(ds.species):
        raise ValueError(f"species must be unique, got {ds.species!r}")
    if tuple(sorted(ds.species)) != ds.species:
        raise ValueError(f"species must be sorted, got {ds.species!r}")

    n_run, n_time, n_wavelength = ds.absorbance.shape
    expected = {
        "time": (ds.time.shape, (n_run, n_time)),
        "wavelength": (ds.wavelength.shape, (n_wavelength,)),
        "mask": (ds.mask.shape, (n_run, n_time)),
        "initial_state": (ds.initial_state.shape, (n_run, ds.n_species)),
        "reference_spectra": (ds.reference_spectra.shape, (ds.n_species, n_wavelength)),
        "reference_sigma": (ds.reference_sigma.shape, (ds.n_species,)),
    }
    for field, (got, want) in expected.items():
        if got != want:
            raise ValueError(f"{field} shape {got} does not match expected {want}")
    if len(ds.run_ids) != n_run:
        raise ValueError(f"run_ids has {len(ds.run_ids)} entries for {n_run} runs")

    if not bool(jnp.all(jnp.diff(ds.wavelength) > 0)):
        raise ValueError("wavelength must be strictly increasing")
    if not bool(jnp.isfinite(ds.wavelength).all()):
        raise ValueError("wavelength must be finite")
    if not bool(jnp.all(ds.mask.any(axis=1))):
        raise ValueError("every run needs at least one measured timepoint")

    measured_time = jnp.where(ds.mask, ds.time, jnp.inf)
    ordered = jnp.sort(measured_time, axis=1)
    if not bool(jnp.allclose(measured_time, ordered, equal_nan=True)):
        raise ValueError("time must be non-decreasing within each run where measured")

    if not bool(jnp.isfinite(jnp.where(ds.mask[:, :, None], ds.absorbance, 0.0)).all()):
        raise ValueError("absorbance must be finite wherever mask is True")
    if not bool(jnp.isfinite(ds.initial_state).all()):
        raise ValueError("initial_state must be finite")
    if not bool((ds.initial_state >= 0).all()):
        raise ValueError("initial_state must be non-negative")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: PASS, 13 tests.

- [ ] **Step 5: Run the full gate**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/spectrahandler/curve_resolution tests/test_dataset.py
git commit -m "feat: SpectralDataset with construction-time validation"
```

---

### Task 1: Synthetic data with known truth

**Files:**
- Create: `src/spectrahandler/curve_resolution/synthetic.py`
- Modify: `src/spectrahandler/curve_resolution/__init__.py`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `SpectralDataset.create` from Task 0.
- Produces: `make_easy_dataset(key: Array, *, n_time: int = 30, n_wavelength: int = 64,
  noise: float = 0.002) -> tuple[SpectralDataset, Array, Array]`, returning the dataset
  plus ground-truth `spectra` of shape `(n_species, n_wavelength)` and `concentrations`
  of shape `(n_run, n_time, n_species)`. Three species, `("a", "b", "c")`.

The "easy" regime from [`spec.md`](spec.md) §1: three well-separated Gaussian bands,
smooth monotone-then-peak kinetics, low uniform noise. Nothing here should be hard.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthetic.py
"""The synthetic generator produces what the recovery test assumes."""

import jax.numpy as jnp
from jax import Array

from spectrahandler.curve_resolution import make_easy_dataset


def test_shapes_and_truth_line_up(key: Array) -> None:
    ds, spectra, concentrations = make_easy_dataset(key)
    assert spectra.shape == (ds.n_species, ds.n_wavelength)
    assert concentrations.shape == (ds.n_run, ds.n_time, ds.n_species)
    assert ds.species == ("a", "b", "c")


def test_truth_reconstructs_the_data(key: Array) -> None:
    ds, spectra, concentrations = make_easy_dataset(key, noise=0.0)
    expected = jnp.einsum("rtk,kw->rtw", concentrations, spectra)
    assert jnp.allclose(ds.absorbance, expected, atol=1e-10)


def test_spectra_are_well_separated(key: Array) -> None:
    """'Easy' means the bands barely overlap; the recovery test relies on it."""
    _, spectra, _ = make_easy_dataset(key)
    normed = spectra / jnp.linalg.norm(spectra, axis=1, keepdims=True)
    gram = normed @ normed.T
    off_diagonal = gram - jnp.diag(jnp.diag(gram))
    assert float(off_diagonal.max()) < 0.3


def test_is_deterministic_given_a_key(key: Array) -> None:
    first, _, _ = make_easy_dataset(key)
    second, _, _ = make_easy_dataset(key)
    assert jnp.array_equal(first.absorbance, second.absorbance)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_synthetic.py -q`
Expected: FAIL — `ImportError: cannot import name 'make_easy_dataset'`.

- [ ] **Step 3: Write the implementation**

```python
# src/spectrahandler/curve_resolution/synthetic.py
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
```

Add to `__init__.py`:

```python
from spectrahandler.curve_resolution.synthetic import make_easy_dataset

__all__ = ["SpectralDataset", "make_easy_dataset"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_synthetic.py -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Run the full gate, then commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
git add src/spectrahandler/curve_resolution tests/test_synthetic.py
git commit -m "feat: synthetic datasets with known spectra and profiles"
```

---

### Task 2: The v0 model, NUTS, and the synthetic round-trip

This is the gate. Nothing in [`spec.md`](spec.md) §7 past step 3 gets written until the
recovery test passes.

**Files:**
- Create: `src/spectrahandler/curve_resolution/model.py`
- Create: `src/spectrahandler/curve_resolution/inference.py`
- Modify: `src/spectrahandler/curve_resolution/__init__.py`
- Test: `tests/test_model.py`, `tests/test_recovery.py`

**Interfaces:**
- Consumes: `SpectralDataset` (Task 0), `make_easy_dataset` (Task 1).
- Produces:
  - `curve_resolution_model(absorbance: Array | None, mask: Array, n_wavelength: int,
    n_species: int, *, tau: float, sigma_scale: float) -> None` — a NumPyro model.
    Sites: `theta_init`, `curvature`, `c_raw`, `sigma`; deterministics `spectra`
    `(n_species, n_wavelength)` and `concentrations` `(n_run, n_time, n_species)`.
  - `fit(dataset: SpectralDataset, n_species: int, key: Array, *, tau: float = 0.01,
    num_warmup: int = 200, num_samples: int = 200, num_chains: int = 2) -> MCMC`.

**One deviation from the spec, decided here.** §4 says each spectrum is normalised to
unit sum. Taken literally, with ~500 channels each `S` entry is ~0.002 and `C` must be
~500 to reproduce `A ~ 1` — the two factors sit at wildly different scales and NUTS
struggles. Normalising each spectrum to **mean one** (i.e. unit sum times
`n_wavelength`) is the identical constraint rescaled, fixes the same scale ambiguity,
and leaves both factors at O(1). Record it in the ADR if it survives.

- [ ] **Step 1: Write the failing shape and prior-predictive tests**

```python
# tests/test_model.py
"""Shapes and prior predictive sanity, before any data is fitted."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpyro.infer import Predictive

from spectrahandler.curve_resolution import curve_resolution_model, make_easy_dataset


@pytest.fixture
def prior_draws(key: Array) -> dict[str, Array]:
    dataset, _, _ = make_easy_dataset(key)
    predictive = Predictive(curve_resolution_model, num_samples=64)
    return predictive(
        jax.random.key(1),
        absorbance=None,
        mask=dataset.mask,
        n_wavelength=dataset.n_wavelength,
        n_species=3,
        tau=0.01,
        sigma_scale=0.01,
    )


def test_deterministic_shapes(prior_draws: dict[str, Array]) -> None:
    assert prior_draws["spectra"].shape == (64, 3, 64)
    assert prior_draws["concentrations"].shape == (64, 1, 30, 3)


def test_spectra_are_non_negative(prior_draws: dict[str, Array]) -> None:
    assert bool((prior_draws["spectra"] >= 0).all())


def test_spectra_are_scale_fixed(prior_draws: dict[str, Array]) -> None:
    """Mean one per spectrum. Without this the posterior has a free scale direction."""
    assert jnp.allclose(prior_draws["spectra"].mean(axis=-1), 1.0, atol=1e-6)


def test_spectra_are_smooth(prior_draws: dict[str, Array]) -> None:
    """Prior draws should look like UV/Vis bands, not noise. With the walk running per
    channel this is the test that pins tau -- see step 4."""
    curvature = jnp.diff(prior_draws["spectra"], n=2, axis=-1)
    assert float(jnp.abs(curvature).mean()) < 0.05


def test_concentrations_are_non_negative(prior_draws: dict[str, Array]) -> None:
    assert bool((prior_draws["concentrations"] >= 0).all())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL — `cannot import name 'curve_resolution_model'`.

- [ ] **Step 3: Write the model**

```python
# src/spectrahandler/curve_resolution/model.py
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
        tau: Fixed scale of the random walk curvature, **per channel**. It therefore
            scales with the square of the bin width: rebinning changes it.
        sigma_scale: Scale of the ``HalfNormal`` prior on the noise standard deviation.
    """
    n_run, n_time = mask.shape

    # Second order random walk, non-centred: two free values plus per-channel curvature.
    theta_init = numpyro.sample(
        "theta_init", dist.Normal(0.0, 1.0).expand([n_species, 2]).to_event(2)
    )
    curvature = numpyro.sample(
        "curvature",
        dist.Normal(0.0, tau).expand([n_species, n_wavelength - 2]).to_event(2),
    )
    slope = jnp.cumsum(
        jnp.concatenate([theta_init[:, 1:2] - theta_init[:, 0:1], curvature], axis=1),
        axis=1,
    )
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
```

- [ ] **Step 4: Run the model tests**

Run: `uv run pytest tests/test_model.py -q`
Expected: PASS, 5 tests. If `test_spectra_are_smooth` fails, `tau` is too large — tune
it from the prior draws, which is what [`spec.md`](spec.md) §9 says to do.

- [ ] **Step 5: Write the failing recovery test**

Component order is not identified: the model may return the same three spectra in any
order. The test matches fitted to true components by spectral correlation before
comparing. Skipping that step makes the test flake roughly two times in three.

```python
# tests/test_recovery.py
"""The gate: does the model recover known spectra and profiles from easy data?"""

import jax
import jax.numpy as jnp
from jax import Array
from numpyro.diagnostics import summary

from spectrahandler.curve_resolution import fit, make_easy_dataset


def _normed(a: Array) -> Array:
    """Rows scaled to unit norm, so a dot product is a correlation."""
    return a / jnp.linalg.norm(a, axis=-1, keepdims=True)


def _match(fitted: Array, truth: Array) -> Array:
    """For each TRUE component, the index of the fitted component matching it best.

    Component order is not identified by the likelihood, so comparing element-wise
    without matching compares arbitrary pairings. The direction matters: the result
    indexes the FITTED arrays, so ``fitted[_match(fitted, truth)]`` lines up with
    ``truth`` row for row.
    """
    similarity = _normed(truth) @ _normed(fitted).T
    taken: list[int] = []
    for row in similarity:
        order = [int(i) for i in jnp.argsort(-row)]
        taken.append(next(i for i in order if i not in taken))
    return jnp.asarray(taken)


def test_recovers_easy_synthetic_data(key: Array) -> None:
    dataset, true_spectra, true_concentrations = make_easy_dataset(key)
    mcmc = fit(dataset, n_species=3, key=jax.random.key(0))
    samples = mcmc.get_samples()

    spectra = samples["spectra"].mean(axis=0)
    permutation = _match(spectra, true_spectra / true_spectra.mean(axis=-1, keepdims=True))

    # Compare shapes, not magnitudes: after mean-one normalisation the scale has moved
    # into the concentrations.
    scaled_truth = true_spectra / true_spectra.mean(axis=-1, keepdims=True)
    for fitted, true in zip(spectra[permutation], scaled_truth, strict=True):
        correlation = jnp.corrcoef(fitted, true)[0, 1]
        assert float(correlation) > 0.98, "resolved spectrum does not match its species"

    lower, upper = jnp.percentile(samples["concentrations"], jnp.array([2.5, 97.5]), axis=0)
    inside = (true_concentrations >= lower[:, :, permutation]) & (
        true_concentrations <= upper[:, :, permutation]
    )
    assert float(inside.mean()) > 0.9, "truth outside the 95% interval too often"


def test_chains_converged(key: Array) -> None:
    """R-hat is checked, not merely printed -- spec.md section 8 and principle 6.

    A fit that has not converged cannot support any statement about posterior width,
    which is the whole deliverable.
    """
    dataset, _, _ = make_easy_dataset(key)
    mcmc = fit(dataset, n_species=3, key=jax.random.key(0))
    stats = summary(mcmc.get_samples(group_by_chain=True), prob=0.95)
    worst = max(float(jnp.nanmax(site["r_hat"])) for site in stats.values())
    assert worst < 1.05, f"worst R-hat {worst:.3f}; chains have not mixed"
```

- [ ] **Step 6: Write the inference driver**

```python
# src/spectrahandler/curve_resolution/inference.py
"""NUTS driver.

Non-dimensionalises absorbance before sampling, because NUTS geometry is much better
when the observations are O(1) -- see ``spec.md`` section 8.
"""

import jax.numpy as jnp
import numpyro
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

    Args:
        dataset: Validated observations. ``v0`` requires ``mask.all()``.
        n_species: Number of components to resolve. Given, not inferred.
        key: PRNG key for the sampler.
        tau: Fixed random walk curvature scale, per wavelength channel.
        num_warmup: Warmup iterations. 200 is the development setting.
        num_samples: Post-warmup draws per chain.
        num_chains: Chains; two is the minimum that lets R-hat mean anything.

    Returns:
        The completed ``MCMC`` object. Call ``print_summary()`` for R-hat and ESS.

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
    )
    return mcmc
```

Set `numpyro.set_host_device_count(2)` in `tests/conftest.py` so `num_chains=2` actually
runs in parallel rather than sequentially.

- [ ] **Step 7: Run the recovery test**

Run: `uv run pytest tests/test_recovery.py -q`
Expected: PASS. This is slow — a minute or two. If it fails, do **not** add model terms;
that is the trap [`spec.md`](spec.md) warns about. Work the list in order: check
divergences, raise `num_warmup`, tune `tau` from prior draws, then reconsider the
parameterisation.

- [ ] **Step 8: Run the full gate and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
git add src/spectrahandler/curve_resolution tests/test_model.py tests/test_recovery.py tests/conftest.py
git commit -m "feat: v0 curve resolution model with NUTS and synthetic round-trip"
```

- [ ] **Step 9: Record what the gate actually showed**

Append to [`spec.md`](spec.md) §7 what step 3 cost and how it behaved: divergences,
R-hat, wall clock, whether the concentration coverage was near 95% or nearer 60%. Step 4
is chosen from that evidence, not from this plan.

---

## Self-review notes

- **Spec coverage:** §3 → Task 0. §1 "easy synthetic" and §7 step 1 → Task 1. §4 model,
  §7 steps 2 and 3 → Task 2. §6 test priorities 1–3 → Tasks 0 and 2. §6 priority 4,
  masking equivalence, is **not** covered — masking is off in v0 and that test lands
  with step 4+, as §1 says.
- **Not covered by design:** steps 4–8, reference spectra, baselines, per-wavelength
  noise, multi-run pooling, rank diagnostics, real fixtures.
- **Open in the spec, not resolved here:** `tau` is given a default of 0.01 to make the
  plan runnable; §9 says to pick it from prior predictive draws, which is Task 2 step 4.
- **Deliberately not built:** no `basis.py`. An earlier draft had a B-spline basis as
  Task 2; it was removed because it buys sampling speed, not correctness, and binning
  (already required by §8) makes the direct random walk affordable. Reintroduce it only
  if profiling says the per-channel walk is the bottleneck.
