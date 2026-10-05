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
            time_unit="h",
            wavelength_unit="nm",
            concentration_unit="uM",
        )
