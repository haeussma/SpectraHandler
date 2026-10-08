"""The one data structure curve resolution operates on.

Dense arrays with leading batch axes, validated once at construction. The axes are ordered
``(run, time, wavelength)``, and species are sorted alphabetically so that every array with
a species axis has one canonical order regardless of how the caller listed them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
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
            ``concentration_unit``.
        reference_spectra: Known pure spectra, shape ``(n_species, n_wavelength)``, in
            absorbance per ``concentration_unit`` (a scan divided by the concentration
            it was taken at). A row is all finite, or all ``NaN`` for no reference.
        reference_sigma: Per-channel noise standard deviation of each reference row,
            shape ``(n_species,)``, same units. Finite and positive wherever the row is
            finite; ignored where it is ``NaN``.
        run_ids: One identifier per run.
        conditions: One condition label per run. Runs that share a label are replicates
            of one condition and are pooled when results are summarised.
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
    conditions: tuple[str, ...]
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

    def run_index(self, run_id: str) -> int:
        """Position of a run on the run axis of every array.

        Args:
            run_id: One of ``run_ids``.

        Returns:
            The index of that run.

        Raises:
            ValueError: If the dataset has no run ``run_id``.
        """
        if run_id not in self.run_ids:
            raise ValueError(f"no run {run_id!r}; the runs are {list(self.run_ids)}")
        return self.run_ids.index(run_id)

    def project(self, spectra: Array) -> Array:
        """Concentrations that best reproduce every measured spectrum from given species spectra.

        Least squares per spectrum, ``absorbance @ pinv(spectra)``: no kinetic model and no
        constraint, so it shows what the spectra alone say about the amounts.

        Args:
            spectra: Shape ``(n_run, n_species, n_wavelength)``, or
                ``(1, n_species, n_wavelength)`` for one set shared by all runs; in
                absorbance per ``concentration_unit``.

        Returns:
            Shape ``(n_run, n_time, n_species)``, in ``concentration_unit``; ``NaN`` where
            not measured.

        Raises:
            ValueError: If ``spectra`` does not fit the dataset's runs, species and
                wavelengths.
        """
        expected = (self.n_species, self.n_wavelength)
        if (
            spectra.ndim != 3
            or spectra.shape[0] not in (1, self.n_run)
            or spectra.shape[1:] != expected
        ):
            raise ValueError(
                f"spectra must have shape (1 or {self.n_run}, {expected[0]}, {expected[1]}), "
                f"got {spectra.shape}"
            )
        projected: Array = self.absorbance @ jnp.linalg.pinv(spectra)
        return projected

    def __post_init__(self) -> None:
        """Validate the contract. Raises rather than letting a fit fail later."""
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
        conditions: Sequence[str] | None = None,
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
                as given. Defaults to all ``NaN``. Units and the per-row all finite or
                all ``NaN`` rule: see the class docstring.
            reference_sigma: Shape ``(n_species,)``. Defaults to all ``NaN``. Units and
                when it must be finite and positive: see the class docstring.
            conditions: One label per run. Defaults to ``"all"`` for every run, so all runs
                are replicates of one condition.
            time_unit: Unit of ``time``.
            wavelength_unit: Unit of ``wavelength``.
            concentration_unit: Unit of concentrations.

        Returns:
            A validated, immutable dataset.

        Raises:
            ValueError: If any contract in the class docstring is violated.
        """
        if isinstance(conditions, str):
            raise ValueError("conditions must be a sequence of labels, one per run, not a string")
        # Duplicates are not rejected here: sorting is well-defined even with ties,
        # and __post_init__'s _validate is the single source of truth for uniqueness.
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
            initial_state=jnp.asarray(initial_state)[:, index],
            reference_spectra=refs,
            reference_sigma=ref_sigma,
            run_ids=tuple(run_ids),
            conditions=("all",) * len(run_ids) if conditions is None else tuple(conditions),
            time_unit=time_unit,
            wavelength_unit=wavelength_unit,
            concentration_unit=concentration_unit,
        )

    @classmethod
    def from_runs(
        cls,
        runs: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
        *,
        wavelength: npt.ArrayLike,
        species: tuple[str, ...],
        initial_state: Mapping[str, float] | Sequence[Mapping[str, float]],
        run_ids: Sequence[str],
        conditions: Sequence[str] | None = None,
        time_unit: str = "h",
        wavelength_unit: str = "nm",
        concentration_unit: str = "uM",
    ) -> "SpectralDataset":
        """Stack runs of different lengths into one dataset, padding with a mask.

        Shorter runs are padded at the end: absorbance ``NaN``, time repeated, mask
        ``False``. Nothing is truncated or interpolated.

        Args:
            runs: One ``(time, absorbance)`` pair per run; ``time`` has shape ``(n_t,)``
                and ``absorbance`` shape ``(n_t, n_wavelength)``. ``n_t`` may differ.
            wavelength: Shared grid, shape ``(n_wavelength,)``, strictly increasing.
            species: Species names, any order.
            initial_state: Initial concentration per species name, one mapping for all
                runs or one per run. Species left out start at 0.
            run_ids: One identifier per run.
            conditions: One label per run; defaults to ``"all"`` for every run.
            time_unit: Unit of ``time``.
            wavelength_unit: Unit of ``wavelength``.
            concentration_unit: Unit of concentrations.

        Returns:
            A validated dataset.

        Raises:
            ValueError: If a run's shapes disagree, a mapping names an unknown species,
                the number of mappings is not one per run, or any dataset contract fails.
        """
        if not runs:
            raise ValueError("runs must be non-empty")
        grid = np.asarray(wavelength, dtype=float)
        times = [np.asarray(t, dtype=float) for t, _ in runs]
        values = [np.asarray(a, dtype=float) for _, a in runs]
        for r, (t, a) in enumerate(zip(times, values, strict=True)):
            if t.ndim != 1 or a.shape != (t.shape[0], grid.shape[0]):
                raise ValueError(
                    f"run {r}: absorbance shape {a.shape} does not match "
                    f"(len(time), len(wavelength)) = ({t.shape[0]}, {grid.shape[0]})"
                )
        n_time = max(t.shape[0] for t in times)
        absorbance = np.full((len(runs), n_time, grid.shape[0]), np.nan)
        time = np.zeros((len(runs), n_time))
        mask = np.zeros((len(runs), n_time), dtype=bool)
        for r, (t, a) in enumerate(zip(times, values, strict=True)):  # IO boundary
            absorbance[r, : t.shape[0]] = a
            time[r, : t.shape[0]] = t
            time[r, t.shape[0] :] = t[-1]
            mask[r, : t.shape[0]] = True

        states = (
            [initial_state] * len(runs)
            if isinstance(initial_state, Mapping)
            else list(initial_state)
        )
        if len(states) != len(runs):
            raise ValueError(f"initial_state has {len(states)} mappings for {len(runs)} runs")
        unknown = sorted({name for state in states for name in state} - set(species))
        if unknown:
            raise ValueError(
                f"initial_state names unknown species {unknown}; species are {species}"
            )
        initial = np.array([[float(state.get(name, 0.0)) for name in species] for state in states])

        return cls.create(
            absorbance=jnp.asarray(absorbance),
            time=jnp.asarray(time),
            wavelength=jnp.asarray(grid),
            species=tuple(species),
            initial_state=jnp.asarray(initial),
            run_ids=tuple(run_ids),
            mask=jnp.asarray(mask),
            conditions=conditions,
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
    if len(ds.conditions) != n_run:
        raise ValueError(f"conditions has {len(ds.conditions)} entries for {n_run} runs")
    if not all(isinstance(c, str) and c for c in ds.conditions):
        raise ValueError(f"conditions must be non-empty strings, got {ds.conditions!r}")

    if not bool(jnp.all(jnp.diff(ds.wavelength) > 0)):
        raise ValueError("wavelength must be strictly increasing")
    if not bool(jnp.isfinite(ds.wavelength).all()):
        raise ValueError("wavelength must be finite")
    if not bool(jnp.all(ds.mask.any(axis=1))):
        raise ValueError("every run needs at least one measured timepoint")

    # Only compare pairs of timepoints that are both measured: a masked-out slot's
    # value is unconstrained and must not affect the check, wherever it sits.
    earlier = jnp.arange(n_time)[:, None] < jnp.arange(n_time)[None, :]
    both_measured = ds.mask[:, :, None] & ds.mask[:, None, :]
    out_of_order = ds.time[:, :, None] > ds.time[:, None, :]
    if bool(jnp.any(earlier & both_measured & out_of_order)):
        raise ValueError("time must be non-decreasing within each run where mask is True")

    if not bool(jnp.isfinite(jnp.where(ds.mask[:, :, None], ds.absorbance, 0.0)).all()):
        raise ValueError("absorbance must be finite wherever mask is True")
    if not bool(jnp.isfinite(ds.initial_state).all()):
        raise ValueError("initial_state must be finite")
    if not bool((ds.initial_state >= 0).all()):
        raise ValueError("initial_state must be non-negative")

    ref_finite = jnp.isfinite(ds.reference_spectra)
    has_ref = ref_finite.all(axis=1)
    if not bool((has_ref | jnp.isnan(ds.reference_spectra).all(axis=1)).all()):
        raise ValueError("each reference_spectra row must be all finite or all NaN")
    sigma_ok = jnp.isfinite(ds.reference_sigma) & (ds.reference_sigma > 0)
    if not bool((sigma_ok | ~has_ref).all()):
        raise ValueError(
            "reference_sigma must be finite and positive for every species with a reference"
        )
