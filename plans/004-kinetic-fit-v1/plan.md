# Kinetic fit v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> [ADR 0005](../../docs/decisions/0005-kinetic-fit-per-replicate-uncertainty-from-replicates.md) accepted 2026-10-06; no blocking preconditions.

**Goal:** Fit any closed first-order reaction scheme to each run of a `SpectralDataset` by maximum likelihood, and pool replicate runs per condition into rate constants and spectra with replicate-based intervals.

**Architecture:** A new `spectrahandler.kinetics` package. `Scheme` turns a list of steps into concentration profiles via `expm`; `fit_kinetics` maximises a profile likelihood per run (spectra by least squares and noise level per wavelength solved exactly; log rates by damped Newton with the exact JAX Hessian); `summarize` pools runs by condition label with a Student-t interval on log rates. `SpectralDataset` gains per-run condition labels and a `from_runs` constructor; a Kinetic Studio CSV reader and a three-panel figure complete it.

**Tech Stack:** Python 3.13, JAX (float64, `jax.scipy.linalg.expm`, `jax.scipy.special.betainc`), NumPy at IO boundaries, matplotlib (optional `plot` extra), pytest, ruff, ty, uv.

**Spec:** [`plans/004-kinetic-fit-v1/spec.md`](spec.md) — read it first; §2.3 explains what the fit deliberately does not fit.

## Global Constraints

- `uv` only (`uv run …`); never `pip`, bare `python`, or activating `.venv`.
- JAX maths, NumPy only at IO boundaries; **no SciPy imports**; float64 (`jax_enable_x64`); the library never enables it itself.
- Explicit PRNG keys; no Python loops over array axes (a loop over runs, over steps' permutations, or over file rows at the IO boundary is allowed).
- Full type annotations incl. `-> None`; Google-style docstrings on every public module, class, function; ruff line length 100; `ty check` clean (`unsound-return-statement` is an error: annotate a local `x: Array = …` before returning JAX results).
- Tuning knobs are arguments with defaults; never silently drop or interpolate data.
- A public API change updates `docs/src/content/docs/` in the same plan (Task 7).
- **PsVAO data never enters git.** The only test that reads it skips when `local/psvao_stopped_flow` is absent.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- After every task: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest` must pass. The Claude hook formats each edited file; still run the full gate.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/spectrahandler/curve_resolution/dataset.py` | + `conditions` field, `from_runs` | 1 |
| `src/spectrahandler/kinetic_studio.py` | Kinetic Studio CSV reader | 2 |
| `src/spectrahandler/kinetics/scheme.py` | `Scheme`, `rate_matrix`, `concentrations` | 3 |
| `src/spectrahandler/kinetics/synthetic.py` | `make_kinetic_replicates` (tests, docs) | 4 |
| `src/spectrahandler/kinetics/fit.py` | likelihood, damped Newton, `fit_kinetics`, `KineticFit` | 4 (+5) |
| `src/spectrahandler/kinetics/summary.py` | t quantile, `RateEstimate`, `ConditionSummary`, `summarize` | 5 |
| `src/spectrahandler/kinetics/__init__.py` | public exports, grows per task | 3, 4, 5 |
| `src/spectrahandler/plot.py` | + `plot_kinetic_fit` | 6 |
| `docs/src/content/docs/guides/kinetic-fit.md` | guide | 7 |
| `tests/test_psvao_local.py` | local EXP0001 gate | 7 |

Every code block below was run in a scratch copy of the repository before this plan was written: the full suite passed (144 tests), `ruff check` and `ty check` were clean, and the EXP0001 gate reproduced the prototype.

---

### Task 1: Condition labels and `SpectralDataset.from_runs`

**Files:**
- Modify: `src/spectrahandler/curve_resolution/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SpectralDataset.conditions: tuple[str, ...]` (one per run; default `("all",) * n_run`); `SpectralDataset.create(..., conditions: Sequence[str] | None = None, ...)`; `SpectralDataset.from_runs(runs: Sequence[tuple[ArrayLike, ArrayLike]], *, wavelength, species, initial_state: Mapping[str, float] | Sequence[Mapping[str, float]], run_ids, conditions=None, time_unit="h", wavelength_unit="nm", concentration_unit="uM") -> SpectralDataset`.

`conditions` is a required dataclass field placed after `run_ids`; `create` defaults it. The one test that calls the raw constructor gains `conditions=("all",)`. `band.py` uses `dataclasses.replace`, which carries the field over unchanged.

- [ ] **Step 1: Write the failing tests** — apply this diff to `tests/test_dataset.py`:

```diff
--- a/tests/test_dataset.py
+++ b/tests/test_dataset.py
@@ -64,6 +64,8 @@
         ({"species": ("a", "a")}, "unique"),
         ({"species": ()}, "non-empty"),
         ({"run_ids": ("r0", "r1")}, "run_ids"),
+        ({"conditions": ("a", "b")}, "conditions"),
+        ({"conditions": ("",)}, "non-empty strings"),
         ({"absorbance": jnp.zeros((1, 3, 5))}, "shape"),
         ({"mask": jnp.zeros((1, 3), dtype=bool)}, "at least one"),
         # Strictly increasing but not finite: must hit the finiteness check, not the
@@ -120,7 +122,67 @@
             reference_spectra=jnp.full((2, 4), jnp.nan),
             reference_sigma=jnp.full((2,), jnp.nan),
             run_ids=("r0",),
+            conditions=("all",),
             time_unit="h",
             wavelength_unit="nm",
             concentration_unit="uM",
         )
+
+
+def test_conditions_default_to_one_condition() -> None:
+    assert _minimal().conditions == ("all",)
+
+
+def test_from_runs_pads_shorter_runs_with_a_mask() -> None:
+    ds = SpectralDataset.from_runs(
+        [
+            (np.array([0.0, 1.0, 2.0]), np.ones((3, 4))),
+            (np.array([0.0, 0.5]), 2 * np.ones((2, 4))),
+        ],
+        wavelength=np.arange(4.0),
+        species=("b", "a"),
+        initial_state={"a": 5.0},
+        run_ids=["r0", "r1"],
+        conditions=["x", "x"],
+        time_unit="s",
+    )
+    assert (ds.n_run, ds.n_time, ds.n_wavelength) == (2, 3, 4)
+    np.testing.assert_array_equal(ds.mask, [[True, True, True], [True, True, False]])
+    assert bool(jnp.isnan(ds.absorbance[1, 2]).all())
+    np.testing.assert_array_equal(ds.time[1], [0.0, 0.5, 0.5])
+    np.testing.assert_array_equal(ds.initial_state, [[5.0, 0.0], [5.0, 0.0]])
+    assert ds.conditions == ("x", "x")
+    assert ds.time_unit == "s"
+
+
+def test_from_runs_accepts_one_initial_state_per_run() -> None:
+    ds = SpectralDataset.from_runs(
+        [(np.arange(2.0), np.zeros((2, 3))), (np.arange(2.0), np.zeros((2, 3)))],
+        wavelength=np.arange(3.0),
+        species=("a",),
+        initial_state=[{"a": 1.0}, {"a": 2.0}],
+        run_ids=["r0", "r1"],
+    )
+    np.testing.assert_array_equal(ds.initial_state, [[1.0], [2.0]])
+
+
+@pytest.mark.parametrize(
+    ("kwargs", "message"),
+    [
+        ({"runs": [(np.arange(2.0), np.zeros((3, 3)))]}, "does not match"),
+        ({"initial_state": {"z": 1.0}}, "unknown species"),
+        ({"initial_state": [{"a": 1.0}, {"a": 1.0}]}, "mappings for 1 runs"),
+        ({"runs": []}, "non-empty"),
+    ],
+)
+def test_from_runs_rejects(kwargs: dict[str, object], message: str) -> None:
+    arguments: dict[str, object] = {
+        "runs": [(np.arange(2.0), np.zeros((2, 3)))],
+        "wavelength": np.arange(3.0),
+        "species": ("a",),
+        "initial_state": {"a": 1.0},
+        "run_ids": ["r0"],
+    }
+    arguments.update(kwargs)
+    with pytest.raises(ValueError, match=message):
+        SpectralDataset.from_runs(**arguments)  # ty: ignore[invalid-argument-type]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: FAIL — `TypeError: SpectralDataset.__init__() got an unexpected keyword argument 'conditions'`, and `AttributeError: ... from_runs`.

- [ ] **Step 3: Implement** — apply this diff to `src/spectrahandler/curve_resolution/dataset.py`:

```diff
--- a/src/spectrahandler/curve_resolution/dataset.py
+++ b/src/spectrahandler/curve_resolution/dataset.py
@@ -5,9 +5,12 @@
 ordered ``(run, time, wavelength)`` and why species are sorted alphabetically.
 """

+from collections.abc import Mapping, Sequence
 from dataclasses import dataclass

 import jax.numpy as jnp
+import numpy as np
+import numpy.typing as npt
 from jax import Array

 __all__ = ["SpectralDataset"]
@@ -37,6 +40,8 @@
             shape ``(n_species,)``, same units. Finite and positive wherever the row is
             finite; ignored where it is ``NaN``.
         run_ids: One identifier per run.
+        conditions: One condition label per run. Runs that share a label are replicates
+            of one condition and are pooled when results are summarised.
         time_unit: Unit string for ``time``, e.g. ``"h"``.
         wavelength_unit: Unit string for ``wavelength``, e.g. ``"nm"``.
         concentration_unit: Unit string for concentrations, e.g. ``"uM"``.
@@ -51,6 +56,7 @@
     reference_spectra: Array
     reference_sigma: Array
     run_ids: tuple[str, ...]
+    conditions: tuple[str, ...]
     time_unit: str
     wavelength_unit: str
     concentration_unit: str
@@ -92,6 +98,7 @@
         mask: Array | None = None,
         reference_spectra: Array | None = None,
         reference_sigma: Array | None = None,
+        conditions: Sequence[str] | None = None,
         time_unit: str = "h",
         wavelength_unit: str = "nm",
         concentration_unit: str = "uM",
@@ -114,6 +121,8 @@
                 all ``NaN`` rule: see the class docstring.
             reference_sigma: Shape ``(n_species,)``. Defaults to all ``NaN``. Units and
                 when it must be finite and positive: see the class docstring.
+            conditions: One label per run. Defaults to ``"all"`` for every run, so all runs
+                are replicates of one condition.
             time_unit: Unit of ``time``.
             wavelength_unit: Unit of ``wavelength``.
             concentration_unit: Unit of concentrations.
@@ -149,7 +158,96 @@
             initial_state=jnp.asarray(initial_state)[:, index],
             reference_spectra=refs,
             reference_sigma=ref_sigma,
+            run_ids=tuple(run_ids),
+            conditions=("all",) * len(run_ids) if conditions is None else tuple(conditions),
+            time_unit=time_unit,
+            wavelength_unit=wavelength_unit,
+            concentration_unit=concentration_unit,
+        )
+
+    @classmethod
+    def from_runs(
+        cls,
+        runs: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
+        *,
+        wavelength: npt.ArrayLike,
+        species: tuple[str, ...],
+        initial_state: Mapping[str, float] | Sequence[Mapping[str, float]],
+        run_ids: Sequence[str],
+        conditions: Sequence[str] | None = None,
+        time_unit: str = "h",
+        wavelength_unit: str = "nm",
+        concentration_unit: str = "uM",
+    ) -> "SpectralDataset":
+        """Stack runs of different lengths into one dataset, padding with a mask.
+
+        Shorter runs are padded at the end: absorbance ``NaN``, time repeated, mask
+        ``False``. Nothing is truncated or interpolated.
+
+        Args:
+            runs: One ``(time, absorbance)`` pair per run; ``time`` has shape ``(n_t,)``
+                and ``absorbance`` shape ``(n_t, n_wavelength)``. ``n_t`` may differ.
+            wavelength: Shared grid, shape ``(n_wavelength,)``, strictly increasing.
+            species: Species names, any order.
+            initial_state: Initial concentration per species name, one mapping for all
+                runs or one per run. Species left out start at 0.
+            run_ids: One identifier per run.
+            conditions: One label per run; defaults to ``"all"`` for every run.
+            time_unit: Unit of ``time``.
+            wavelength_unit: Unit of ``wavelength``.
+            concentration_unit: Unit of concentrations.
+
+        Returns:
+            A validated dataset.
+
+        Raises:
+            ValueError: If a run's shapes disagree, a mapping names an unknown species,
+                the number of mappings is not one per run, or any dataset contract fails.
+        """
+        if not runs:
+            raise ValueError("runs must be non-empty")
+        grid = np.asarray(wavelength, dtype=float)
+        times = [np.asarray(t, dtype=float) for t, _ in runs]
+        values = [np.asarray(a, dtype=float) for _, a in runs]
+        for r, (t, a) in enumerate(zip(times, values, strict=True)):
+            if t.ndim != 1 or a.shape != (t.shape[0], grid.shape[0]):
+                raise ValueError(
+                    f"run {r}: absorbance shape {a.shape} does not match "
+                    f"(len(time), len(wavelength)) = ({t.shape[0]}, {grid.shape[0]})"
+                )
+        n_time = max(t.shape[0] for t in times)
+        absorbance = np.full((len(runs), n_time, grid.shape[0]), np.nan)
+        time = np.zeros((len(runs), n_time))
+        mask = np.zeros((len(runs), n_time), dtype=bool)
+        for r, (t, a) in enumerate(zip(times, values, strict=True)):  # IO boundary
+            absorbance[r, : t.shape[0]] = a
+            time[r, : t.shape[0]] = t
+            time[r, t.shape[0] :] = t[-1]
+            mask[r, : t.shape[0]] = True
+
+        states = (
+            [initial_state] * len(runs)
+            if isinstance(initial_state, Mapping)
+            else list(initial_state)
+        )
+        if len(states) != len(runs):
+            raise ValueError(f"initial_state has {len(states)} mappings for {len(runs)} runs")
+        unknown = sorted({name for state in states for name in state} - set(species))
+        if unknown:
+            raise ValueError(
+                f"initial_state names unknown species {unknown}; species are {species}"
+            )
+        initial = np.array([[float(state.get(name, 0.0)) for name in species] for state in states])
+
+        return cls.create(
+            absorbance=jnp.asarray(absorbance),
+            time=jnp.asarray(time),
+            wavelength=jnp.asarray(grid),
+            species=tuple(species),
+            initial_state=jnp.asarray(initial),
             run_ids=tuple(run_ids),
+            mask=jnp.asarray(mask),
+            conditions=conditions,
             time_unit=time_unit,
             wavelength_unit=wavelength_unit,
             concentration_unit=concentration_unit,
@@ -179,6 +277,10 @@
             raise ValueError(f"{field} shape {got} does not match expected {want}")
     if len(ds.run_ids) != n_run:
         raise ValueError(f"run_ids has {len(ds.run_ids)} entries for {n_run} runs")
+    if len(ds.conditions) != n_run:
+        raise ValueError(f"conditions has {len(ds.conditions)} entries for {n_run} runs")
+    if not all(isinstance(c, str) and c for c in ds.conditions):
+        raise ValueError(f"conditions must be non-empty strings, got {ds.conditions!r}")

     if not bool(jnp.all(jnp.diff(ds.wavelength) > 0)):
         raise ValueError("wavelength must be strictly increasing")
```

- [ ] **Step 4: Run the tests and the full gate**

Run: `uv run pytest tests/test_dataset.py -q` → PASS, then:
```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/spectrahandler/curve_resolution/dataset.py tests/test_dataset.py
git commit -m "feat: condition labels per run and SpectralDataset.from_runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Kinetic Studio CSV reader

Moved here from plan 003 §4 (that section stays the layout reference).

**Files:**
- Create: `src/spectrahandler/kinetic_studio.py`
- Test: `tests/test_kinetic_studio.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `read_kinetic_studio(path: str | Path) -> KineticStudioScan`; `KineticStudioScan(wavelength_nm (n_wl,), time_s (n_t,), absorbance (n_t, n_wl), header: dict[str, str])`, a frozen dataclass with `eq=False`, like `jasco.Scan`.

- [ ] **Step 1: Write the failing test** — create `tests/test_kinetic_studio.py`:

```python
"""The Kinetic Studio reader against a small file in the instrument's layout."""

from pathlib import Path

import numpy as np
import pytest

from spectrahandler.kinetic_studio import read_kinetic_studio

_FILE = (
    'File Info:"," DX2 Kinetics File:data11.ksd | 21/10/2025 14:55:43" Temp: -36.7,\n'
    ",0,0.01,0.02,\n"
    "301.0,0.30,0.31,0.32,\n"
    "300.5,0.20,,0.22,\n"
    "\n"
    "end,\n"
)


@pytest.fixture
def export(tmp_path: Path) -> Path:
    path = tmp_path / "measurement01_001s.csv"
    path.write_text(_FILE, encoding="utf-8")
    return path


def test_shape_order_and_values(export: Path) -> None:
    scan = read_kinetic_studio(export)
    np.testing.assert_array_equal(scan.time_s, [0.0, 0.01, 0.02])
    np.testing.assert_array_equal(scan.wavelength_nm, [300.5, 301.0])
    assert scan.absorbance.shape == (3, 2)
    assert scan.absorbance[2, 1] == pytest.approx(0.32)
    assert scan.absorbance[0, 0] == pytest.approx(0.20)


def test_empty_cell_is_nan_not_dropped(export: Path) -> None:
    scan = read_kinetic_studio(export)
    assert np.isnan(scan.absorbance[1, 0])


def test_header_fields(export: Path) -> None:
    header = read_kinetic_studio(export).header
    assert header["file"] == "data11.ksd"
    assert header["timestamp"] == "21/10/2025 14:55:43"
    assert header["temperature"] == "-36.7"
    assert header["raw"].startswith("File Info")


def test_rejects_repeated_wavelength(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("x\n,0,1\n300,1,2\n300,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        read_kinetic_studio(path)


def test_rejects_file_without_data(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("x\n,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a header line"):
        read_kinetic_studio(path)
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_kinetic_studio.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectrahandler.kinetic_studio'`.

- [ ] **Step 3: Implement** — create `src/spectrahandler/kinetic_studio.py`:

```python
"""Reader for TgK Scientific Kinetic Studio CSV exports (stopped-flow, diode array).

Layout, from a KinetAsyst SF-61DX2 with a DET2B diode array::

    File Info:"," DX2 Kinetics File:data11.ksd | 21/10/2025 14:55:43" Temp: -36.7,
    ,0,0.01,0.02, ... ,0.99          <- leading empty field, then times in seconds
    300.21,0.21666,0.18396, ...      <- one row per wavelength

The reader returns the file's numbers unchanged apart from sorting wavelength ascending.
Traps it does not fix:

- The ``_NNNs`` in a filename is the acquisition window, not a time point.
- Times are returned as recorded. In the EXP0001 set, shots with windows longer than
  1 s label each frame one sampling step late; the label is not the reaction time.
- The temperature in the header is kept as text: the instrument's reading is unreliable.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["KineticStudioScan", "read_kinetic_studio"]

type _Floats = npt.NDArray[np.float64]

_HEADER = re.compile(
    r'File:\s*(?P<file>[^|"]+?)\s*\|\s*(?P<timestamp>[^"]+?)\s*"\s*Temp:\s*(?P<temperature>[^,]*)'
)


@dataclass(frozen=True, eq=False)
class KineticStudioScan:
    """One stopped-flow shot as recorded.

    Attributes:
        wavelength_nm: Strictly increasing grid, shape ``(n_wavelength,)``, in nm.
        time_s: Frame times as recorded, shape ``(n_time,)``, in seconds.
        absorbance: Shape ``(n_time, n_wavelength)``, in absorbance units. An empty cell
            in the file is ``NaN``, never dropped.
        header: ``raw`` (the first line), and ``file``, ``timestamp`` and
            ``temperature`` (text) when the first line has the usual form.
    """

    wavelength_nm: _Floats
    time_s: _Floats
    absorbance: _Floats
    header: dict[str, str]


def _number(field: str) -> float:
    """A cell as a float; an empty cell is ``NaN``."""
    field = field.strip()
    return float(field) if field else float("nan")


def read_kinetic_studio(path: str | Path) -> KineticStudioScan:
    """Read one Kinetic Studio CSV export.

    Rows with fewer than three fields (blank lines, footers) are skipped: they are not
    spectra. Single-wavelength trace files are not supported.

    Args:
        path: The ``.csv`` export.

    Returns:
        The shot, with wavelength sorted ascending.

    Raises:
        ValueError: If the file has no time row or no data rows, or repeats a wavelength.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: expected a header line, a time row and data rows")
    header = {"raw": lines[0]}
    match = _HEADER.search(lines[0])
    if match:
        header.update({k: v.strip() for k, v in match.groupdict().items()})

    time_fields = lines[1].split(",")[1:]
    while time_fields and not time_fields[-1].strip():  # trailing comma, not a timepoint
        time_fields.pop()
    time_s = np.array([float(f) for f in time_fields])

    rows = [line.split(",") for line in lines[2:]]
    rows = [row for row in rows if len(row) >= 3]
    if not rows or time_s.size == 0:
        raise ValueError(f"{path}: no time row or no data rows")
    wavelength = np.array([float(row[0]) for row in rows])
    absorbance = np.full((len(rows), time_s.size), np.nan)
    for i, row in enumerate(rows):  # IO boundary: one row per wavelength
        cells = [_number(f) for f in row[1 : 1 + time_s.size]]
        absorbance[i, : len(cells)] = cells

    order = np.argsort(wavelength, kind="stable")
    if np.any(np.diff(wavelength[order]) <= 0):
        raise ValueError(f"{path}: a wavelength appears more than once")
    return KineticStudioScan(
        wavelength_nm=wavelength[order],
        time_s=time_s,
        absorbance=np.ascontiguousarray(absorbance[order].T),
        header=header,
    )
```

- [ ] **Step 4: Run the test and the full gate**

Run: `uv run pytest tests/test_kinetic_studio.py -q` → PASS, then:
```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/spectrahandler/kinetic_studio.py tests/test_kinetic_studio.py
git commit -m "feat: read Kinetic Studio stopped-flow CSV exports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `Scheme` and closed-form concentration profiles

**Files:**
- Create: `src/spectrahandler/kinetics/__init__.py`, `src/spectrahandler/kinetics/scheme.py`
- Test: `tests/test_scheme.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Scheme(steps: Sequence[tuple[str, str]])` with `.steps`, `.species` (sorted), `.n_steps`, `.index_steps(species) -> tuple[tuple[int, int], ...]`; `type Step = tuple[str, str]`; `type IndexStep = tuple[int, int]`; `rate_matrix(rates (n_steps,), steps, n_species) -> (n_species, n_species)`; `concentrations(time (n_t,), rates (n_steps,), initial (n_species,), steps) -> (n_t, n_species)`.

`expm` (not an eigendecomposition) is deliberate: equal rates make `K` defective and reversible pairs can make it awkward; `test_equal_rates_are_exact` pins this.

- [ ] **Step 1: Write the failing test** — create `tests/test_scheme.py`:

```python
"""Scheme validation and its closed-form concentration profiles."""

import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.kinetics import Scheme
from spectrahandler.kinetics.scheme import concentrations, rate_matrix

TIME = jnp.linspace(0.0, 10.0, 50)


def test_species_are_sorted_and_steps_kept_in_order() -> None:
    scheme = Scheme(steps=[("B", "C"), ("A", "B")])
    assert scheme.species == ("A", "B", "C")
    assert scheme.steps == (("B", "C"), ("A", "B"))
    assert scheme.n_steps == 2


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([], "at least one step"),
        ([("A", "A")], "change the species"),
        ([("A", "B"), ("A", "B")], "unique"),
        ([("A", "B", "C")], "pair"),
        ([("A", "")], "pair"),
    ],
)
def test_rejects(steps: list[tuple[str, ...]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Scheme(steps=steps)  # ty: ignore[invalid-argument-type]


def test_index_steps_follow_the_given_species_order() -> None:
    scheme = Scheme(steps=[("A", "B")])
    assert scheme.index_steps(("B", "A")) == ((1, 0),)
    with pytest.raises(ValueError, match="differ"):
        scheme.index_steps(("A", "C"))


def test_rate_matrix_columns_sum_to_zero() -> None:
    k = rate_matrix(jnp.array([0.6, 0.4, 0.1]), ((0, 1), (1, 2), (1, 0)), 3)
    np.testing.assert_allclose(k.sum(axis=0), 0.0, atol=1e-15)


def test_consecutive_matches_the_analytic_solution() -> None:
    c = concentrations(TIME, jnp.array([0.6, 0.4]), jnp.array([1.0, 0.0, 0.0]), ((0, 1), (1, 2)))
    a = jnp.exp(-0.6 * TIME)
    b = 0.6 / (0.4 - 0.6) * (jnp.exp(-0.6 * TIME) - jnp.exp(-0.4 * TIME))
    np.testing.assert_allclose(c[:, 0], a, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(c[:, 1], b, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(c.sum(axis=1), 1.0, rtol=1e-12)


def test_equal_rates_are_exact() -> None:
    """Equal rates make K defective; expm still gives b = k t exp(-k t)."""
    c = concentrations(TIME, jnp.array([0.5, 0.5]), jnp.array([1.0, 0.0, 0.0]), ((0, 1), (1, 2)))
    np.testing.assert_allclose(c[:, 1], 0.5 * TIME * jnp.exp(-0.5 * TIME), rtol=1e-10, atol=1e-14)


def test_reversible_pair_reaches_equilibrium() -> None:
    c = concentrations(
        jnp.array([200.0]), jnp.array([0.3, 0.1]), jnp.array([1.0, 0.0]), ((0, 1), (1, 0))
    )
    np.testing.assert_allclose(c[0], [0.25, 0.75], rtol=1e-10)
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_scheme.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectrahandler.kinetics'`.

- [ ] **Step 3: Implement** — create `src/spectrahandler/kinetics/scheme.py`:

```python
"""First-order reaction schemes and their concentration profiles in closed form.

A scheme is a list of steps ``reactant -> product``, each with its own rate constant.
Every step conserves the total concentration, so a scheme is closed:
``dc/dt = K c`` with the columns of ``K`` summing to zero, and
``c(t) = expm(K t) c(0)``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm

__all__ = ["Scheme", "Step", "concentrations", "rate_matrix"]

#: One first-order step, ``(reactant, product)``.
type Step = tuple[str, str]
#: A step as indices into a species tuple.
type IndexStep = tuple[int, int]


@dataclass(frozen=True, init=False)
class Scheme:
    """A closed scheme of first-order steps.

    ``Scheme(steps=[("A", "B"), ("B", "C")])`` is A -> B -> C. ``("A", "B")`` together
    with ``("B", "A")`` is a reversible pair. Every step has its own rate constant;
    rates are always given and returned in the order of ``steps``.

    Attributes:
        steps: The steps, in the order given.
    """

    steps: tuple[Step, ...]

    def __init__(self, steps: Sequence[Step]) -> None:
        """Validate and store the steps.

        Args:
            steps: ``(reactant, product)`` pairs of species names.

        Raises:
            ValueError: If there are no steps, a step is not a pair of two different
                non-empty names, or a step appears twice.
        """
        checked: list[Step] = []
        for step in steps:
            if len(step) != 2 or not all(isinstance(n, str) and n for n in step):
                raise ValueError(f"a step is a (reactant, product) pair of names, got {step!r}")
            if step[0] == step[1]:
                raise ValueError(f"a step must change the species, got {step!r}")
            checked.append((step[0], step[1]))
        if not checked:
            raise ValueError("a scheme needs at least one step")
        if len(set(checked)) != len(checked):
            raise ValueError(f"steps must be unique, got {checked!r}")
        object.__setattr__(self, "steps", tuple(checked))

    @property
    def species(self) -> tuple[str, ...]:
        """Every species named in a step, sorted alphabetically."""
        return tuple(sorted({name for step in self.steps for name in step}))

    @property
    def n_steps(self) -> int:
        """Number of steps, which is the number of rate constants."""
        return len(self.steps)

    def index_steps(self, species: tuple[str, ...]) -> tuple[IndexStep, ...]:
        """The steps as ``(reactant, product)`` indices into ``species``.

        Args:
            species: The dataset's species, in its order.

        Returns:
            One index pair per step, in step order.

        Raises:
            ValueError: If the scheme's species and ``species`` are not the same set.
        """
        if set(species) != set(self.species):
            raise ValueError(
                f"scheme species {self.species} differ from dataset species {tuple(species)}"
            )
        position = {name: i for i, name in enumerate(species)}
        return tuple((position[a], position[b]) for a, b in self.steps)


def rate_matrix(rates: Array, steps: tuple[IndexStep, ...], n_species: int) -> Array:
    """The matrix ``K`` of ``dc/dt = K c``.

    Step ``i`` moves ``rates[i] * c[reactant]`` from reactant to product, so every column
    sums to zero.

    Args:
        rates: Shape ``(n_steps,)``, in 1 / time unit.
        steps: Index pairs from :meth:`Scheme.index_steps`.
        n_species: Number of species.

    Returns:
        Shape ``(n_species, n_species)``.
    """
    reactant = jnp.asarray([a for a, _ in steps])
    product = jnp.asarray([b for _, b in steps])
    k = jnp.zeros((n_species, n_species), dtype=rates.dtype)
    return k.at[reactant, reactant].add(-rates).at[product, reactant].add(rates)


def concentrations(
    time: Array, rates: Array, initial: Array, steps: tuple[IndexStep, ...]
) -> Array:
    """Concentration profiles ``c(t) = expm(K t) c(0)``.

    ``expm`` stays exact for equal and for reversible rates, where an eigendecomposition
    of ``K`` is defective or complex.

    Args:
        time: Shape ``(n_time,)``, in the time unit of ``rates``.
        rates: Shape ``(n_steps,)``.
        initial: Initial concentrations, shape ``(n_species,)``.
        steps: Index pairs from :meth:`Scheme.index_steps`.

    Returns:
        Shape ``(n_time, n_species)``, in the unit of ``initial``.
    """
    k = rate_matrix(rates, steps, initial.shape[0])
    profiles: Array = jax.vmap(lambda t: expm(k * t) @ initial)(time)
    return profiles
```

and `src/spectrahandler/kinetics/__init__.py`:

```python
"""Kinetic fits of first-order schemes, per run, with replicates pooled by condition."""

from spectrahandler.kinetics.scheme import Scheme, Step

__all__ = ["Scheme", "Step"]
```

- [ ] **Step 4: Run the test and the full gate**

Run: `uv run pytest tests/test_scheme.py -q` → PASS, then:
```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/spectrahandler/kinetics/__init__.py src/spectrahandler/kinetics/scheme.py tests/test_scheme.py
git commit -m "feat: first-order reaction schemes with closed-form profiles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `fit_kinetics` — maximum likelihood per run

**Files:**
- Create: `src/spectrahandler/kinetics/synthetic.py`, `src/spectrahandler/kinetics/fit.py`
- Modify: `src/spectrahandler/kinetics/__init__.py` (full replacement below)
- Test: `tests/test_fit.py`

**Interfaces:**
- Consumes: `Scheme`, `Step`, `IndexStep`, `concentrations` (Task 3); `SpectralDataset` with `conditions` (Task 1).
- Produces: `make_kinetic_replicates(key, scheme, rates: Mapping[Step, float], *, initial: Mapping[str, float], n_replicates=4, between_sd_log=0.0, noise=0.002, n_time=40, n_wavelength=30, condition="synthetic") -> (SpectralDataset, log_rates (n_rep, n_steps), spectra (n_species, n_wl))`; `fit_kinetics(dataset, scheme, *, initial_rates=None, t_offset=0.0, max_iter=200, ambiguity_tolerance=1e-3, rank_tolerance=1e-10) -> KineticFit`; `KineticFit` fields `scheme, species, run_ids, conditions, rates (n_run, n_steps), log_rate_sd, spectra (n_run, n_species, n_wl), concentrations (n_run, n_t, n_species), residuals (n_run, n_t, n_wl), sigma (n_run, n_wl), log_likelihood (n_run,), converged: tuple[bool, ...], ambiguities: tuple[tuple[str, tuple[float, ...]], ...], time_unit`.

Why damped Newton and not `jax.scipy.optimize.minimize(method="BFGS")`: in the plan-004 prototype, BFGS's line search failed from reasonable starts on A → B → C even at noise 1e-3, while damped Newton with the exact Hessian converged from every start at noise 1e-6 to 1e-2. Do not swap it back.

`KineticFit.summary()` arrives in Task 5; this task's `fit.py` does not have it yet.

- [ ] **Step 1: Write the failing test** — create `tests/test_fit.py`:

```python
"""Per-run maximum likelihood: recovery, what is absorbed, ambiguity, rank, masks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates

ONE_STEP = Scheme(steps=[("A", "B")])
TWO_STEP = Scheme(steps=[("A", "B"), ("B", "C")])


def test_recovers_rates_and_spectra_from_nearly_clean_data(key: jax.Array) -> None:
    data, log_rates, spectra = make_kinetic_replicates(
        key,
        TWO_STEP,
        {("A", "B"): 0.6, ("B", "C"): 0.25},
        initial={"A": 10.0},
        n_replicates=1,
        noise=1e-7,
    )
    fit = fit_kinetics(data, TWO_STEP, initial_rates={("A", "B"): 0.5, ("B", "C"): 0.2})
    assert fit.converged == (True,)
    np.testing.assert_allclose(fit.rates[0], jnp.exp(log_rates[0]), rtol=1e-5)
    np.testing.assert_allclose(fit.spectra[0], spectra, rtol=1e-4, atol=1e-7)


def test_shapes_and_units(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0})
    fit = fit_kinetics(data, ONE_STEP)
    assert fit.rates.shape == (4, 1)
    assert fit.spectra.shape == (4, 2, data.n_wavelength)
    assert fit.concentrations.shape == (4, data.n_time, 2)
    assert fit.residuals.shape == (4, data.n_time, data.n_wavelength)
    assert fit.sigma.shape == (4, data.n_wavelength)
    assert fit.log_likelihood.shape == (4,)
    assert fit.time_unit == "s"
    assert all(fit.converged)
    assert bool(jnp.isfinite(fit.log_rate_sd).all())


def _shifted(
    data: SpectralDataset, absorbance: jax.Array | None = None, time: jax.Array | None = None
) -> SpectralDataset:
    return SpectralDataset.create(
        absorbance=data.absorbance if absorbance is None else absorbance,
        time=data.time if time is None else time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        conditions=data.conditions,
        time_unit=data.time_unit,
    )


def test_static_offset_and_time_shift_do_not_move_the_rates(key: jax.Array) -> None:
    """Spec §2.3: both are absorbed by the species spectra."""
    data, _, _ = make_kinetic_replicates(
        key, TWO_STEP, {("A", "B"): 0.6, ("B", "C"): 0.25}, initial={"A": 10.0}
    )
    rates = {("A", "B"): 0.5, ("B", "C"): 0.2}
    base = fit_kinetics(data, TWO_STEP, initial_rates=rates).rates
    tilt = 0.01 + 0.02 * jnp.linspace(-1.0, 1.0, data.n_wavelength)
    offset = fit_kinetics(
        _shifted(data, absorbance=data.absorbance + tilt), TWO_STEP, initial_rates=rates
    ).rates
    duration = float(data.time[0, -1] - data.time[0, 0])
    shifted = fit_kinetics(
        _shifted(data, time=data.time + 0.02 * duration), TWO_STEP, initial_rates=rates
    ).rates
    np.testing.assert_allclose(offset, base, rtol=1e-6)
    np.testing.assert_allclose(shifted, base, rtol=1e-6)


def test_rate_swap_is_reported_for_a_consecutive_scheme(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, TWO_STEP, {("A", "B"): 0.6, ("B", "C"): 0.25}, initial={"A": 10.0}, n_replicates=1
    )
    fit = fit_kinetics(data, TWO_STEP, initial_rates={("A", "B"): 0.5, ("B", "C"): 0.2})
    assert len(fit.ambiguities) == 1
    run_id, alternative = fit.ambiguities[0]
    assert run_id == "replicate1"
    np.testing.assert_allclose(alternative, np.asarray(fit.rates[0])[::-1], rtol=1e-12)


def test_one_step_has_no_ambiguity(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0})
    assert fit_kinetics(data, ONE_STEP).ambiguities == ()


def test_padded_run_gives_the_same_rates(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    padded = SpectralDataset.create(
        absorbance=jnp.concatenate(
            [data.absorbance, jnp.full((1, 5, data.n_wavelength), jnp.nan)], axis=1
        ),
        time=jnp.concatenate([data.time, jnp.full((1, 5), data.time[0, -1])], axis=1),
        mask=jnp.concatenate([jnp.ones((1, data.n_time), bool), jnp.zeros((1, 5), bool)], axis=1),
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        time_unit=data.time_unit,
    )
    fit, fit_padded = fit_kinetics(data, ONE_STEP), fit_kinetics(padded, ONE_STEP)
    np.testing.assert_allclose(fit_padded.rates, fit.rates, rtol=1e-10)
    assert bool(jnp.isnan(fit_padded.residuals[0, -5:]).all())


def test_species_never_populated_is_rank_deficient(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    extra = Scheme(steps=[("A", "B"), ("C", "B")])
    with_c = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=("A", "B", "C"),
        initial_state=jnp.array([[10.0, 0.0, 0.0]]),
        run_ids=data.run_ids,
        time_unit=data.time_unit,
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_kinetics(with_c, extra, initial_rates={("A", "B"): 0.8, ("C", "B"): 0.3})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scheme": Scheme(steps=[("A", "C")])}, "differ"),
        ({"initial_rates": {("A", "C"): 1.0}}, "exactly the steps"),
        ({"initial_rates": {("A", "B"): -1.0}}, "positive"),
    ],
)
def test_rejects(key: jax.Array, kwargs: dict[str, object], message: str) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    arguments: dict[str, object] = {"scheme": ONE_STEP}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        fit_kinetics(data, **arguments)  # ty: ignore[invalid-argument-type]


def test_requires_float64(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            fit_kinetics(data, ONE_STEP)
    finally:
        jax.config.update("jax_enable_x64", True)
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_fit.py -q`
Expected: FAIL — `ImportError: cannot import name 'fit_kinetics' from 'spectrahandler.kinetics'`.

- [ ] **Step 3: Implement** — create `src/spectrahandler/kinetics/synthetic.py`:

```python
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
```

create `src/spectrahandler/kinetics/fit.py`:

```python
"""Maximum likelihood per run for a first-order scheme (ADR 0005).

Per run ``r``: ``D_r(t, λ) = Σ_s c_rs(t) S_rs(λ) + ε``, with ``c_r(t)`` from the scheme,
spectra free, and independent Gaussian noise with its own level per wavelength. Given
the rates, the spectra are the least-squares solution and each wavelength's noise
variance is ``RSS_λ / n``; both are solved exactly, leaving the profile log likelihood
``-n/2 Σ_λ [log(2π RSS_λ / n) + 1]`` over the log rates.

Runs are fitted independently. What runs share is decided afterwards, by condition, in
:func:`spectrahandler.kinetics.summary.summarize`.

Two things the data cannot tell apart are absorbed rather than fitted (spec §2.3): a
static offset and a shift of the time axis both end up in the species spectra, and the
rates do not move. The first species' spectrum therefore means "the state at the first
time label".
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.kinetics.scheme import IndexStep, Scheme, Step, concentrations

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
            is typically too small (ADR 0005).
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
            as well as the fitted rates (within ``ambiguity_tolerance``), as in the rate
            swap of A -> B -> C. Not resolved: the data cannot choose.
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

    A handful of log rates, exact JAX gradient and Hessian: Newton converges in tens of
    steps where BFGS line searches stall (prototype, plan 004). Each step is capped at a
    factor e² in any rate. Converged means the Hessian is positive definite and the
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
        t_offset: Added to every time before fitting. Rates do not depend on it; it only
            changes which state the first species' spectrum describes. Use it when the
            instrument's dead time is known from a calibration.
        max_iter: Newton iterations per run.
        ambiguity_tolerance: Log-likelihood difference below which a permutation of the
            fitted rates counts as fitting equally well.
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
        singular = jnp.linalg.svd(c, compute_uv=False)
        if float(singular.min()) <= rank_tolerance * float(singular.max()):
            raise ValueError(
                f"run {run_id!r}: the concentration profiles are rank-deficient at the fitted "
                f"rates, so the spectra of {dataset.species} are not all identified "
                "(a species is never populated, or two rates coincide)"
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
                if abs(float(alt_lik - log_lik)) < ambiguity_tolerance:
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
```

and replace `src/spectrahandler/kinetics/__init__.py` with:

```python
"""Kinetic fits of first-order schemes, per run, with replicates pooled by condition."""

from spectrahandler.kinetics.fit import KineticFit, fit_kinetics
from spectrahandler.kinetics.scheme import Scheme, Step
from spectrahandler.kinetics.synthetic import make_kinetic_replicates

__all__ = ["KineticFit", "Scheme", "Step", "fit_kinetics", "make_kinetic_replicates"]
```

- [ ] **Step 4: Run the test and the full gate**

Run: `uv run pytest tests/test_fit.py -q` → PASS (about 20 s: JIT compiles once per run shape), then:
```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/spectrahandler/kinetics/__init__.py src/spectrahandler/kinetics/synthetic.py src/spectrahandler/kinetics/fit.py tests/test_fit.py
git commit -m "feat: per-run maximum likelihood fit of first-order schemes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pool replicates per condition

**Files:**
- Create: `src/spectrahandler/kinetics/summary.py`
- Modify: `src/spectrahandler/kinetics/fit.py` (add `KineticFit.summary`), `src/spectrahandler/kinetics/__init__.py` (full replacement)
- Test: `tests/test_summary.py`

**Interfaces:**
- Consumes: `KineticFit`, `fit_kinetics`, `make_kinetic_replicates`, `Scheme` (Tasks 3–4).
- Produces: `student_t_quantile(p, df) -> float`; `RateEstimate(value, lower, upper, between_sd_log, within_sd_log, n)`; `ConditionSummary(condition, run_ids, rates: dict[Step, RateEstimate], spectrum_mean, spectrum_lower, spectrum_upper, warnings)`; `summarize(fit, *, level=0.95, min_ratio=3.0) -> dict[str, ConditionSummary]`; `KineticFit.summary(level=0.95, min_ratio=3.0)`.

The coverage test fits 1200 runs (300 conditions × 4) and takes about 30 s. Its band 0.92–0.98 is ±2.4 standard errors around 0.95 for 300 conditions; it was 0.98 with BFGS on 200 and passes with this code. If it fails, the interval is wrong — do not widen the band.

- [ ] **Step 1: Write the failing test** — create `tests/test_summary.py`:

```python
"""Pooling replicates: the t quantile, coverage of the interval, and the warnings."""

import jax
import numpy as np
import pytest

from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates
from spectrahandler.kinetics.summary import student_t_quantile

ONE_STEP = Scheme(steps=[("A", "B")])
STEP = ("A", "B")


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [
        (0.975, 1.0, 12.706204736),
        (0.975, 3.0, 3.182446305),
        (0.975, 30.0, 2.042272456),
        (0.95, 10.0, 1.812461123),
    ],
)
def test_student_t_quantile(p: float, df: float, expected: float) -> None:
    assert student_t_quantile(p, df) == pytest.approx(expected, rel=1e-8)


def test_interval_covers_the_true_rate_at_the_stated_level() -> None:
    """300 conditions x 4 replicates; replicate log rates scatter by 2 %."""
    hits = []
    for i in range(300):
        data, _, _ = make_kinetic_replicates(
            jax.random.key(i), ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
        )
        estimate = fit_kinetics(data, ONE_STEP).summary()["synthetic"].rates[STEP]
        assert estimate.lower is not None
        assert estimate.upper is not None
        hits.append(estimate.lower <= 0.8 <= estimate.upper)
    assert 0.92 <= np.mean(hits) <= 0.98


def test_conditions_are_pooled_separately(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
    )
    from spectrahandler.curve_resolution import SpectralDataset

    relabelled = SpectralDataset.create(
        absorbance=data.absorbance,
        time=data.time,
        wavelength=data.wavelength,
        species=data.species,
        initial_state=data.initial_state,
        run_ids=data.run_ids,
        conditions=("low", "low", "high", "high"),
        time_unit=data.time_unit,
    )
    summary = fit_kinetics(relabelled, ONE_STEP).summary()
    assert list(summary) == ["low", "high"]
    assert summary["high"].run_ids == ("replicate3", "replicate4")
    assert summary["low"].rates[STEP].n == 2


def test_one_run_has_no_interval(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, n_replicates=1
    )
    summary = fit_kinetics(data, ONE_STEP).summary()["synthetic"]
    assert summary.rates[STEP].lower is None
    assert summary.spectrum_lower is None
    assert any("no replicate-based uncertainty" in w for w in summary.warnings)


def test_warns_when_replicates_do_not_spread_beyond_fit_precision(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.0
    )
    summary = fit_kinetics(data, ONE_STEP).summary()["synthetic"]
    assert any("not clearly larger" in w for w in summary.warnings)


def test_no_warning_when_replicates_spread(key: jax.Array) -> None:
    data, _, _ = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.05
    )
    assert fit_kinetics(data, ONE_STEP).summary()["synthetic"].warnings == ()


def test_spectrum_band_brackets_the_mean(key: jax.Array) -> None:
    data, _, spectra = make_kinetic_replicates(
        key, ONE_STEP, {STEP: 0.8}, initial={"A": 10.0}, between_sd_log=0.02
    )
    summary = fit_kinetics(data, ONE_STEP).summary()["synthetic"]
    assert summary.spectrum_lower is not None
    assert summary.spectrum_upper is not None
    assert bool((summary.spectrum_lower <= summary.spectrum_mean).all())
    assert bool((summary.spectrum_mean <= summary.spectrum_upper).all())
    np.testing.assert_allclose(summary.spectrum_mean, spectra, atol=5e-4)


def test_swap_ambiguity_becomes_a_warning(key: jax.Array) -> None:
    two = Scheme(steps=[("A", "B"), ("B", "C")])
    data, _, _ = make_kinetic_replicates(
        key, two, {("A", "B"): 0.6, ("B", "C"): 0.25}, initial={"A": 10.0}
    )
    summary = fit_kinetics(data, two, initial_rates={("A", "B"): 0.5, ("B", "C"): 0.2}).summary()[
        "synthetic"
    ]
    assert any("not separately identified" in w for w in summary.warnings)
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_summary.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectrahandler.kinetics.summary'`.

- [ ] **Step 3: Implement** — create `src/spectrahandler/kinetics/summary.py`:

```python
"""Pool replicate runs per condition (ADR 0005).

Rates are combined on the log scale: the geometric mean, with a Student-t interval on
``n - 1`` degrees of freedom from the spread between replicates. For normally
distributed replicate log rates with flat priors on their mean and on the log of their
spread, and negligible within-run error, this interval is also the posterior interval of
the mean log rate. Spectra are combined per wavelength on the linear scale.

The interval covers only what varies between the declared replicates: consecutive shots
from one loading do not cover preparation-to-preparation or day-to-day variation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import betainc

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
        level: Coverage of the intervals, in (0, 1).
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
```

apply this diff to `src/spectrahandler/kinetics/fit.py`:

```diff
--- a/src/spectrahandler/kinetics/fit.py
+++ b/src/spectrahandler/kinetics/fit.py
@@ -19,6 +19,7 @@
 from collections.abc import Mapping
 from dataclasses import dataclass
 from functools import partial
+from typing import TYPE_CHECKING

 import jax
 import jax.numpy as jnp
@@ -28,6 +29,9 @@
 from spectrahandler.curve_resolution.dataset import SpectralDataset
 from spectrahandler.kinetics.scheme import IndexStep, Scheme, Step, concentrations

+if TYPE_CHECKING:
+    from spectrahandler.kinetics.summary import ConditionSummary
+
 __all__ = ["KineticFit", "fit_kinetics"]

 #: Newton state: log rates, objective, damping, iteration, done.
@@ -82,7 +86,22 @@
     ambiguities: tuple[tuple[str, tuple[float, ...]], ...]
     time_unit: str

+    def summary(self, level: float = 0.95, min_ratio: float = 3.0) -> dict[str, "ConditionSummary"]:
+        """Replicates pooled per condition; see :func:`~spectrahandler.kinetics.summarize`.

+        Args:
+            level: Coverage of the reported intervals.
+            min_ratio: Warn when the between-replicate spread of a log rate is less than
+                this multiple of its within-run standard deviation.
+
+        Returns:
+            One summary per condition, in order of first appearance.
+        """
+        from spectrahandler.kinetics.summary import summarize
+
+        return summarize(self, level=level, min_ratio=min_ratio)
+
+
 def _profile(
     log_rates: Array, time: Array, absorbance: Array, initial: Array, steps: tuple[IndexStep, ...]
 ) -> tuple[Array, Array, Array, Array]:
```

and replace `src/spectrahandler/kinetics/__init__.py` with:

```python
"""Kinetic fits of first-order schemes, per run, with replicates pooled by condition."""

from spectrahandler.kinetics.fit import KineticFit, fit_kinetics
from spectrahandler.kinetics.scheme import Scheme, Step
from spectrahandler.kinetics.summary import ConditionSummary, RateEstimate, summarize
from spectrahandler.kinetics.synthetic import make_kinetic_replicates

__all__ = [
    "ConditionSummary",
    "KineticFit",
    "RateEstimate",
    "Scheme",
    "Step",
    "fit_kinetics",
    "make_kinetic_replicates",
    "summarize",
]
```

- [ ] **Step 4: Run the test and the full gate**

Run: `uv run pytest tests/test_summary.py -q` → PASS, then:
```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add src/spectrahandler/kinetics/summary.py src/spectrahandler/kinetics/fit.py src/spectrahandler/kinetics/__init__.py tests/test_summary.py
git commit -m "feat: pool replicates per condition with a replicate-based interval

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `plot_kinetic_fit`

**Files:**
- Modify: `src/spectrahandler/plot.py`, `tests/test_plot.py`

**Interfaces:**
- Consumes: `KineticFit`, `fit_kinetics`, `make_kinetic_replicates`, `Scheme`.
- Produces: `plot_kinetic_fit(fit: KineticFit, dataset: SpectralDataset, *, wavelengths: Sequence[float]) -> Figure` with three left-titled axes (+ colorbar): "data and model…", "species spectra…", "what is left, per noise level…".

Colours: wavelengths take the existing `_CATEGORICAL` slots in order; species take the new fixed `_SPECIES` tuple (five slots, never cycled — a sixth species raises). Residuals use the existing `_DIVERGING` map at ±3 noise levels, like `plot_noise`.

- [ ] **Step 1: Write the failing tests** — apply this diff to `tests/test_plot.py`:

```diff
--- a/tests/test_plot.py
+++ b/tests/test_plot.py
@@ -5,8 +5,11 @@

 matplotlib.use("Agg")

+import pytest
+
 from spectrahandler.curve_resolution import make_realistic_dataset, noise_diagnostics
-from spectrahandler.plot import plot_noise
+from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates
+from spectrahandler.plot import plot_kinetic_fit, plot_noise


 def test_plot_noise_has_three_panels() -> None:
@@ -15,3 +18,23 @@
     titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
     assert len(titles) == 3
     assert titles[0].startswith("what the data hold")
+
+
+def test_plot_kinetic_fit_has_three_panels() -> None:
+    scheme = Scheme(steps=[("A", "B")])
+    data, _, _ = make_kinetic_replicates(
+        jax.random.key(0), scheme, {("A", "B"): 0.8}, initial={"A": 10.0}
+    )
+    fig = plot_kinetic_fit(fit_kinetics(data, scheme), data, wavelengths=[400, 520])
+    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
+    assert len(titles) == 3
+    assert titles[0].startswith("data and model")
+
+
+def test_plot_kinetic_fit_rejects_four_wavelengths() -> None:
+    scheme = Scheme(steps=[("A", "B")])
+    data, _, _ = make_kinetic_replicates(
+        jax.random.key(0), scheme, {("A", "B"): 0.8}, initial={"A": 10.0}, n_replicates=1
+    )
+    with pytest.raises(ValueError, match="1 to 3"):
+        plot_kinetic_fit(fit_kinetics(data, scheme), data, wavelengths=[400, 450, 500, 550])
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_plot.py -q`
Expected: FAIL — `ImportError: cannot import name 'plot_kinetic_fit'`.

- [ ] **Step 3: Implement** — apply this diff to `src/spectrahandler/plot.py`:

```diff
--- a/src/spectrahandler/plot.py
+++ b/src/spectrahandler/plot.py
@@ -1,22 +1,26 @@
 """Figures for checking a resolution by eye. Needs the ``plot`` extra (matplotlib)."""

+from collections.abc import Sequence
 from typing import TYPE_CHECKING

 import numpy as np

 from spectrahandler.curve_resolution.dataset import SpectralDataset
 from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics
+from spectrahandler.kinetics.fit import KineticFit

 if TYPE_CHECKING:
     from matplotlib.figure import Figure

-__all__ = ["plot_noise"]
+__all__ = ["plot_kinetic_fit", "plot_noise"]

 #: Diverging pair with a neutral midpoint: residuals below zero blue, above red.
 _DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
 #: Categorical slots for series that are kinds, not signs: wavelength first, time second.
 _CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
 _INK, _MUTED, _SIGNAL, _NOISE = "#0b0b0b", "#52514e", "#2a78d6", "#a3a29b"
+#: Species slots, fixed order, never cycled: a sixth species is an error, not a new hue.
+_SPECIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8c5bd6", "#c4950a")


 def plot_noise(diagnostics: NoiseDiagnostics, dataset: SpectralDataset) -> "Figure":
@@ -133,3 +137,114 @@
         for side in ("top", "right"):
             ax.spines[side].set_visible(False)
     return fig
+
+
+def plot_kinetic_fit(
+    fit: KineticFit, dataset: SpectralDataset, *, wavelengths: Sequence[float]
+) -> "Figure":
+    """Three panels: data against model, the species spectra, and what is left.
+
+    Left, absorbance over time at up to three wavelengths, every run overlaid: dots are
+    data, black lines the fitted model. Middle, each species' spectrum per run (thin)
+    and their mean (thick). Right, the residual of every run divided by that run's
+    fitted noise level per wavelength, runs stacked: even static means the model and
+    the noise level explain the data; stripes or blocks mean something is missing.
+
+    Args:
+        fit: From :func:`spectrahandler.kinetics.fit_kinetics` on ``dataset``.
+        dataset: The data that was fitted.
+        wavelengths: One to three wavelengths for the left panel, in
+            ``dataset.wavelength_unit``; the nearest channel is shown.
+
+    Returns:
+        A matplotlib figure with three axes.
+
+    Raises:
+        ValueError: If not one to three wavelengths are given, or there are more than
+            five species.
+        ImportError: If matplotlib is not installed.
+    """
+    if not 1 <= len(wavelengths) <= len(_CATEGORICAL):
+        raise ValueError(f"give 1 to {len(_CATEGORICAL)} wavelengths, got {len(wavelengths)}")
+    if len(fit.species) > len(_SPECIES):
+        raise ValueError(f"plot_kinetic_fit draws at most {len(_SPECIES)} species")
+    try:
+        import matplotlib.pyplot as plt
+        from matplotlib.colors import LinearSegmentedColormap
+    except ImportError as err:
+        raise ImportError(
+            "plot_kinetic_fit needs matplotlib: install the 'plot' extra, "
+            "e.g. uv add 'spectrahandler[plot]'"
+        ) from err
+
+    fig, (ax_tr, ax_sp, ax_res) = plt.subplots(
+        1, 3, figsize=(15, 4), width_ratios=(1.2, 1, 1.4), constrained_layout=True
+    )
+    wl = np.asarray(dataset.wavelength)
+    mask = np.asarray(dataset.mask)
+    time = np.asarray(dataset.time)
+    absorbance = np.asarray(dataset.absorbance)
+    residuals = np.asarray(fit.residuals)
+
+    for j, target in enumerate(wavelengths):
+        i = int(np.argmin(np.abs(wl - target)))
+        for r in range(dataset.n_run):
+            m = mask[r]
+            ax_tr.plot(
+                time[r, m],
+                absorbance[r, m, i],
+                "o",
+                ms=2.5,
+                mec="none",
+                alpha=0.45,
+                color=_CATEGORICAL[j],
+                label=f"{wl[i]:.0f} {dataset.wavelength_unit}" if r == 0 else None,
+            )
+            ax_tr.plot(time[r, m], absorbance[r, m, i] - residuals[r, m, i], color=_INK, lw=1)
+    ax_tr.set(
+        xlabel=f"time ({dataset.time_unit})",
+        ylabel="absorbance",
+        title="data and model\ndots: data, lines: fitted model, runs overlaid",
+    )
+    ax_tr.legend(frameon=False, fontsize=8)
+
+    spectra = np.asarray(fit.spectra)
+    for s, name in enumerate(fit.species):
+        for r in range(dataset.n_run):
+            ax_sp.plot(wl, spectra[r, s], color=_SPECIES[s], lw=0.8, alpha=0.5)
+        ax_sp.plot(wl, spectra[:, s].mean(axis=0), color=_SPECIES[s], lw=2, label=name)
+    ax_sp.set(
+        xlabel=f"wavelength ({dataset.wavelength_unit})",
+        ylabel=f"absorbance per {dataset.concentration_unit}",
+        title="species spectra\nthin: each run, thick: mean over runs",
+    )
+    ax_sp.legend(frameon=False, fontsize=8)
+
+    scaled = residuals / np.asarray(fit.sigma)[:, None, :]
+    stacked = np.concatenate([scaled[r, mask[r]] for r in range(dataset.n_run)])
+    boundaries = np.cumsum(mask.sum(axis=1))[:-1]
+    image = ax_res.imshow(
+        stacked,
+        aspect="auto",
+        cmap=LinearSegmentedColormap.from_list("residual", _DIVERGING),
+        vmin=-3,
+        vmax=3,
+        extent=(wl[0], wl[-1], stacked.shape[0] - 0.5, -0.5),
+        interpolation="nearest",
+    )
+    for boundary in boundaries:
+        ax_res.axhline(boundary - 0.5, color=_INK, lw=0.8)
+    fig.colorbar(image, ax=ax_res, label="residual / noise level")
+    ax_res.set(
+        xlabel=f"wavelength ({dataset.wavelength_unit})",
+        ylabel="timepoint (runs stacked)",
+        title="what is left, per noise level\neven static: noise; stripes or blocks: misfit",
+    )
+
+    for ax in (ax_tr, ax_sp, ax_res):
+        title = ax.get_title()
+        ax.set_title("")
+        ax.set_title(title, loc="left", fontsize=9.5)
+        for side in ("top", "right"):
+            ax.spines[side].set_visible(False)
+    return fig
```

- [ ] **Step 4: Run the tests and the full gate**

Run: `uv run pytest tests/test_plot.py -q` → PASS, then:
```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 5: Look at it.** Render once and inspect: `uv run python -c "import jax; jax.config.update('jax_enable_x64', True); import matplotlib; matplotlib.use('Agg'); from spectrahandler.kinetics import *; from spectrahandler.plot import plot_kinetic_fit; s=Scheme([('A','B')]); d,_,_=make_kinetic_replicates(jax.random.key(0), s, {('A','B'): 0.8}, initial={'A': 10.0}); plot_kinetic_fit(fit_kinetics(d, s), d, wavelengths=[400, 520]).savefig('/tmp/kinetic_fit.png', dpi=120)"` and open `/tmp/kinetic_fit.png`: no overlapping labels, legends readable, residual panel even static.

- [ ] **Step 6: Commit**

```bash
git add src/spectrahandler/plot.py tests/test_plot.py
git commit -m "feat: plot_kinetic_fit, data and model, spectra, residual

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Guide page and the local EXP0001 gate

**Files:**
- Create: `docs/src/content/docs/guides/kinetic-fit.md`, `tests/test_psvao_local.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the public guide; a local-only regression against the spec §5 numbers.

The guide uses only synthetic data. PsVAO data and numbers never appear in `docs/`.

- [ ] **Step 1: Write the local gate** — create `tests/test_psvao_local.py`:

```python
"""EXP0001 reproduces the plan-004 prototype. Local only: skips without the data."""

from pathlib import Path

import numpy as np
import pytest

from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetic_studio import read_kinetic_studio
from spectrahandler.kinetics import Scheme, fit_kinetics

ROOT = Path(__file__).resolve().parents[1] / "local" / "psvao_stopped_flow" / "Raw data" / "EXP0001"
FILES = ("measurement01_001s", "measurement02_001s", "measurement03_001s", "measurement01_005s")

pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="PsVAO data is local-only")


def test_exp0001_matches_the_prototype() -> None:
    scans = [read_kinetic_studio(ROOT / f"{name}.csv") for name in FILES]
    keep = (scans[0].wavelength_nm >= 340.0) & (scans[0].wavelength_nm <= 700.0)
    data = SpectralDataset.from_runs(
        [(s.time_s, s.absorbance[:, keep]) for s in scans],
        wavelength=scans[0].wavelength_nm[keep],
        species=("E_red_QM", "E_start"),
        initial_state={"E_start": 7.0},
        run_ids=["shot1", "shot2", "shot3", "shot5s"],
        conditions=["250uM"] * 4,
        time_unit="s",
    )
    scheme = Scheme(steps=[("E_start", "E_red_QM")])
    fit = fit_kinetics(data, scheme)
    np.testing.assert_allclose(fit.rates[:, 0], [3.1665, 3.1308, 3.1966, 3.1273], atol=0.002)
    rate = fit.summary()["250uM"].rates[("E_start", "E_red_QM")]
    assert rate.value == pytest.approx(3.155, abs=0.002)
    assert rate.lower == pytest.approx(3.104, abs=0.002)
    assert rate.upper == pytest.approx(3.208, abs=0.002)
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_psvao_local.py -q -rs`
Expected: PASS where `local/psvao_stopped_flow` exists (it does on the author's machine); `SKIPPED (PsVAO data is local-only)` elsewhere, including CI. If it fails with the data present, stop and report the numbers — do not edit the expected values.

- [ ] **Step 3: Write the guide** — create `docs/src/content/docs/guides/kinetic-fit.md`:

````markdown
---
title: Rate constants from replicate time courses
description: Fit a first-order reaction scheme to each run, pool replicates by condition, and report rate constants with an interval that comes from the replicates.
sidebar:
  order: 4
---

When you know the reaction scheme, the time course of the spectra pins down both the
rate constants and the spectrum of every species. `fit_kinetics` fits the scheme to each
run on its own; `summary()` pools the replicates of each condition.

## Run it

```python
import jax

jax.config.update("jax_enable_x64", True)            # spectrahandler needs float64

from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates

scheme = Scheme(steps=[("A", "B")])                  # A -> B, first order
data, true_log_rates, true_spectra = make_kinetic_replicates(
    jax.random.key(0), scheme, {("A", "B"): 0.8},    # 0.8 per second
    initial={"A": 10.0}, n_replicates=4, between_sd_log=0.02,
)

fit = fit_kinetics(data, scheme)
rate = fit.summary()["synthetic"].rates[("A", "B")]
print(f"k = {rate.value:.3f} 1/s, 95 % interval [{rate.lower:.3f}, {rate.upper:.3f}]")
print(fit.rates[:, 0])                               # one rate per replicate
```

Your own data go in through `SpectralDataset.from_runs`, one `(time, absorbance)` pair per
run. Runs with the same `conditions` label are replicates. Stopped-flow exports from
Kinetic Studio are read with `spectrahandler.kinetic_studio.read_kinetic_studio`.

```python
from spectrahandler.curve_resolution import SpectralDataset

data = SpectralDataset.from_runs(
    [(scan.time_s, scan.absorbance) for scan in scans],
    wavelength=scans[0].wavelength_nm,
    species=("A", "B"),
    initial_state={"A": 7.0},                       # per run: pass a list of dicts
    run_ids=["shot1", "shot2", "shot3"],
    conditions=["250 uM"] * 3,
    time_unit="s",
)
```

Schemes are lists of steps: `[("A", "B"), ("B", "C")]` is A → B → C, and
`[("A", "B"), ("B", "A")]` is a reversible pair. For more than one step, pass
`initial_rates` with a rough guess per step.

## Where the interval comes from

Each run is fitted by maximum likelihood, with the noise level of every wavelength
estimated in the same fit. The rate's interval does **not** come from that fit: within
one run the fit is far more confident than repeated runs agree, because real noise is
correlated and every run carries small differences of its own. The interval comes from
how much your replicates disagree: the geometric mean of the replicate rates, with a
Student-t interval on n − 1 degrees of freedom.

So:

- **One run gives a rate but no interval.** The summary says so.
- **The interval covers only what differs between your replicates.** Three shots from one
  syringe loading say nothing about the next enzyme preparation.
- **`within_sd_log`** is the single-run precision, as a diagnostic. When replicates do not
  spread clearly beyond it, the summary warns.

## What the fit cannot tell apart

- **A constant offset.** A baseline that does not change in time adds equally to every
  species' spectrum. Rates are unaffected; the spectra include the offset.
- **The true time zero.** Shifting the time axis changes only the spectrum of the first
  species, which then describes the mixture present at your first time label. Rates are
  unaffected. If you know your instrument's dead time, pass it as `t_offset`.
- **Swapped rates in a chain.** In A → B → C with an unknown spectrum for B, the two
  rates can be exchanged with an equally good fit. `fit.ambiguities` lists such cases and
  the summary warns; it does not choose for you.

## Check the fit

```python
import matplotlib.pyplot as plt
from spectrahandler.plot import plot_kinetic_fit

plot_kinetic_fit(fit, data, wavelengths=[400, 520])
plt.show()
```

The right panel shows what is left after the fit, in units of each wavelength's noise
level. Even static is good. Stripes along wavelength or blocks along time mean the
scheme, or the assumption of no drifting baseline, does not hold, and the rates are then
the best the chosen scheme can do, not the truth.
````

- [ ] **Step 4: Run the guide's code and build the site**

```bash
uv run python - <<'EOF'
import re, pathlib
text = pathlib.Path("docs/src/content/docs/guides/kinetic-fit.md").read_text()
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
exec("import matplotlib\nmatplotlib.use('Agg')\n" + blocks[0] + "\n" + blocks[2])
EOF
```

Expected: prints `k = 0.818 1/s, 95 % interval [0.786, 0.852]` and four replicate rates (block 2 needs real files and is not run). Then `cd docs && npm run build` must succeed.

- [ ] **Step 5: Full gate**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest
```

- [ ] **Step 6: Commit**

```bash
git add docs/src/content/docs/guides/kinetic-fit.md tests/test_psvao_local.py
git commit -m "docs: guide for kinetic fits; local EXP0001 regression

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review against the spec

| Spec item | Task |
|---|---|
| §1.1 `Scheme`, closed form | 3 |
| §1.2 `fit_kinetics`, `KineticFit` | 4 |
| §1.3 `summary()`, `ConditionSummary` | 5 |
| §1.4 `conditions`, `from_runs` | 1 |
| §1.5 `read_kinetic_studio` | 2 |
| §1.6 `plot_kinetic_fit` | 6 |
| §1.7 guide | 7 |
| §2.3 offset / time shift absorbed (tests) | 4 (`test_static_offset_and_time_shift_do_not_move_the_rates`) |
| §2.3 rate swap reported | 4, 5 |
| §2.3 rank check | 4 |
| §2.4 log-scale pooling, t interval, validity warning, n = 1 | 5 |
| §6 gate 1–7 | 3, 4, 5 |
| §6 local gate | 7 |
| float64 guard (as `resolve_band`) | 4 |
