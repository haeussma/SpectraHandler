"""Construction invariants for SpectralDataset."""

import jax.numpy as jnp
import numpy as np
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
    return SpectralDataset.create(**kwargs)  # ty: ignore[invalid-argument-type]


def test_species_are_sorted_and_columns_follow() -> None:
    ds = _minimal()
    assert ds.species == ("a", "b")
    # initial_state was given in ("b", "a") order as [10, 20]; it must be permuted.
    np.testing.assert_array_equal(ds.initial_state, [[20.0, 10.0]])


def test_shape_properties() -> None:
    ds = _minimal()
    assert (ds.n_run, ds.n_time, ds.n_wavelength, ds.n_species) == (1, 3, 4, 2)


def test_mask_defaults_to_all_measured() -> None:
    assert bool(_minimal().mask.all())


def test_references_default_to_nan() -> None:
    ds = _minimal()
    assert bool(jnp.isnan(ds.reference_spectra).all())
    assert bool(jnp.isnan(ds.reference_sigma).all())


def test_time_ignores_order_of_masked_out_points() -> None:
    """A masked-out timepoint's value may be out of order; only measured times must be."""
    ds = _minimal(
        absorbance=jnp.zeros((1, 4, 4)),
        time=jnp.array([[0.0, 5.0, 2.0, 3.0]]),
        mask=jnp.array([[True, False, True, True]]),
    )
    assert bool(ds.mask[0, 1]) is False


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
        ({"conditions": ("a", "b")}, "conditions"),
        ({"conditions": ("",)}, "non-empty strings"),
        ({"conditions": "a"}, "not a string"),
        ({"absorbance": jnp.zeros((1, 3, 5))}, "shape"),
        ({"mask": jnp.zeros((1, 3), dtype=bool)}, "at least one"),
        # Strictly increasing but not finite: must hit the finiteness check, not the
        # monotonicity one (jnp.inf > every finite value, so "increasing" alone
        # wouldn't catch this).
        ({"wavelength": jnp.array([0.0, 1.0, 2.0, jnp.inf])}, "finite"),
        ({"initial_state": jnp.array([[jnp.nan, 1.0]])}, "finite"),
        (
            {
                "reference_spectra": jnp.array([[0.0, jnp.nan, 0.0, 0.0], [jnp.nan] * 4]),
                "reference_sigma": jnp.array([1.0, jnp.nan]),
            },
            "all finite or all NaN",
        ),
        # Not finite anywhere, but not NaN either: inf is not "no reference".
        (
            {
                "reference_spectra": jnp.array([[jnp.inf] * 4, [jnp.nan] * 4]),
                "reference_sigma": jnp.array([1.0, jnp.nan]),
            },
            "all finite or all NaN",
        ),
        (
            {
                "reference_spectra": jnp.array([[0.0] * 4, [jnp.nan] * 4]),
                "reference_sigma": jnp.array([jnp.nan, jnp.nan]),
            },
            "reference_sigma",
        ),
        (
            {
                "reference_spectra": jnp.array([[0.0] * 4, [jnp.nan] * 4]),
                "reference_sigma": jnp.array([0.0, jnp.nan]),
            },
            "reference_sigma",
        ),
    ],
)
def test_validation_rejects(overrides: dict[str, Array], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _minimal(**overrides)


def test_direct_construction_rejects_unsorted_species() -> None:
    """create() always sorts species; the raw constructor must reject it itself."""
    with pytest.raises(ValueError, match="sorted"):
        SpectralDataset(
            absorbance=jnp.zeros((1, 3, 4)),
            time=jnp.arange(3.0).reshape(1, 3),
            wavelength=jnp.arange(4.0),
            mask=jnp.ones((1, 3), dtype=bool),
            species=("b", "a"),
            initial_state=jnp.array([[10.0, 20.0]]),
            reference_spectra=jnp.full((2, 4), jnp.nan),
            reference_sigma=jnp.full((2,), jnp.nan),
            run_ids=("r0",),
            conditions=("all",),
            time_unit="h",
            wavelength_unit="nm",
            concentration_unit="uM",
        )


def test_conditions_default_to_one_condition() -> None:
    assert _minimal().conditions == ("all",)


def test_from_runs_pads_shorter_runs_with_a_mask() -> None:
    ds = SpectralDataset.from_runs(
        [
            (np.array([0.0, 1.0, 2.0]), np.ones((3, 4))),
            (np.array([0.0, 0.5]), 2 * np.ones((2, 4))),
        ],
        wavelength=np.arange(4.0),
        species=("b", "a"),
        initial_state={"a": 5.0},
        run_ids=["r0", "r1"],
        conditions=["x", "x"],
        time_unit="s",
    )
    assert (ds.n_run, ds.n_time, ds.n_wavelength) == (2, 3, 4)
    np.testing.assert_array_equal(ds.mask, [[True, True, True], [True, True, False]])
    assert bool(jnp.isnan(ds.absorbance[1, 2]).all())
    np.testing.assert_array_equal(ds.time[1], [0.0, 0.5, 0.5])
    np.testing.assert_array_equal(ds.initial_state, [[5.0, 0.0], [5.0, 0.0]])
    assert ds.conditions == ("x", "x")
    assert ds.time_unit == "s"


def test_from_runs_accepts_one_initial_state_per_run() -> None:
    ds = SpectralDataset.from_runs(
        [(np.arange(2.0), np.zeros((2, 3))), (np.arange(2.0), np.zeros((2, 3)))],
        wavelength=np.arange(3.0),
        species=("a",),
        initial_state=[{"a": 1.0}, {"a": 2.0}],
        run_ids=["r0", "r1"],
    )
    np.testing.assert_array_equal(ds.initial_state, [[1.0], [2.0]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"runs": [(np.arange(2.0), np.zeros((3, 3)))]}, "does not match"),
        ({"initial_state": {"z": 1.0}}, "unknown species"),
        ({"initial_state": [{"a": 1.0}, {"a": 1.0}]}, "mappings for 1 runs"),
        ({"runs": []}, "non-empty"),
    ],
)
def test_from_runs_rejects(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "runs": [(np.arange(2.0), np.zeros((2, 3)))],
        "wavelength": np.arange(3.0),
        "species": ("a",),
        "initial_state": {"a": 1.0},
        "run_ids": ["r0"],
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SpectralDataset.from_runs(**arguments)  # ty: ignore[invalid-argument-type]
