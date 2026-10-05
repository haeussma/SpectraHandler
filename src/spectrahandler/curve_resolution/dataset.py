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
        reference_spectra: Known pure spectra, shape ``(n_species, n_wavelength)``, in
            absorbance per ``concentration_unit`` (a scan divided by the concentration
            it was taken at). A row is all finite, or all ``NaN`` for no reference.
        reference_sigma: Per-channel noise standard deviation of each reference row,
            shape ``(n_species,)``, same units. Finite and positive wherever the row is
            finite; ignored where it is ``NaN``.
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
    if not bool((has_ref | ~ref_finite.any(axis=1)).all()):
        raise ValueError("each reference_spectra row must be all finite or all NaN")
    sigma_ok = jnp.isfinite(ds.reference_sigma) & (ds.reference_sigma > 0)
    if not bool((sigma_ok | ~has_ref).all()):
        raise ValueError(
            "reference_sigma must be finite and positive for every species with a reference"
        )
