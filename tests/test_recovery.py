"""The gate: does the model recover known spectra and profiles from easy data?

It does not, and both tests below are strict xfails recording that. The v0 model
resolves the spectra essentially perfectly (correlation 1.00000 to five decimals) and
recovers sigma, but NUTS does not mix on it and the concentration credible intervals are
too narrow by roughly a factor of two. See ``spec.md`` §7 "Gate result (step 3)" for the
full evidence and for what was tried. ``strict=True`` so that whichever change finally
fixes this turns these into failures and forces the marks off.
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpyro.diagnostics import summary

from spectrahandler.curve_resolution import fit, make_easy_dataset

# Species axis of each sampled site, in the (draw, ...) layout one chain has.
_SPECIES_AXIS = {
    "spectra": 1,
    "concentrations": 3,
    "theta_init": 1,
    "curvature": 1,
    "c_raw": 3,
}


@dataclass(frozen=True)
class Fitted:
    """One fit, with every species axis relabelled onto the truth.

    Attributes:
        spectra: Relabelled draws, shape ``(n_chain, n_draw, n_species, n_wavelength)``.
        concentrations: Relabelled draws, shape
            ``(n_chain, n_draw, n_run, n_time, n_species)``, in the model's scaled units.
        scaled_truth: True spectra normalised to mean one, matching the model.
        true_concentrations: True concentrations in the model's scaled units.
        worst_r_hat: Worst R-hat over the relabelled deterministics and ``sigma``.
    """

    spectra: Array
    concentrations: Array
    scaled_truth: Array
    true_concentrations: Array
    worst_r_hat: float


def _normed(a: Array) -> Array:
    """Rows scaled to unit norm, so a dot product is a correlation."""
    scaled: Array = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    return scaled


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


def _pool(draws: Array) -> Array:
    """Flatten the chain axis into the draw axis. Safe only after relabelling."""
    return draws.reshape(-1, *draws.shape[2:])


@pytest.fixture(scope="module")
def fitted() -> Fitted:
    """Run the fit once, then relabel each chain's components onto the truth.

    Labelling is per chain, not pooled. The likelihood is invariant to permuting the
    species, so two chains can and do settle on different orderings; pooling them or
    taking R-hat across them without aligning first compares species ``a`` in one chain
    against species ``c`` in the other and reports a spurious failure. Relabelling is
    post-processing of an unidentified label, not a change to the model.

    Module-scoped so both gate tests share one fit, which halves the wall clock. That
    rules out the function-scoped ``key`` fixture, so the key is built here.
    """
    dataset, true_spectra, true_concentrations = make_easy_dataset(jax.random.key(0))
    started = time.perf_counter()
    mcmc = fit(dataset, n_species=3, key=jax.random.key(0))
    by_chain = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    # JAX dispatches asynchronously, so the clock must stop after the arrays land.
    wall_clock = time.perf_counter() - started

    scaled_truth = true_spectra / true_spectra.mean(axis=-1, keepdims=True)
    n_chain = by_chain["spectra"].shape[0]
    permutations = [
        _match(by_chain["spectra"][c].mean(axis=0), scaled_truth) for c in range(n_chain)
    ]
    relabelled = {
        site: (
            jnp.stack(
                [jnp.take(d[c], permutations[c], axis=_SPECIES_AXIS[site]) for c in range(n_chain)]
            )
            if site in _SPECIES_AXIS
            else d
        )
        for site, d in by_chain.items()
    }

    worst_r_hat = _print_evidence(mcmc, relabelled, permutations, wall_clock)

    # fit() divides absorbance by max|A|, and the model's spectra have mean one, so the
    # fitted concentrations are in those non-dimensional units. Move the truth into them
    # rather than moving the model: comparing uM against a scaled fit would fail with
    # certainty and would say nothing about whether the model works.
    scale = float(jnp.nanmax(jnp.abs(dataset.absorbance)))
    return Fitted(
        spectra=relabelled["spectra"],
        concentrations=relabelled["concentrations"],
        scaled_truth=scaled_truth,
        true_concentrations=true_concentrations * true_spectra.mean(axis=-1) / scale,
        worst_r_hat=worst_r_hat,
    )


def _print_evidence(
    mcmc: object,
    relabelled: dict[str, Array],
    permutations: list[Array],
    wall_clock: float,
) -> float:
    """Print the gate diagnostics and return the worst deterministic R-hat.

    spec.md section 8 asks for R-hat and ESS on every fit. The raw latent sites are
    reported but not asserted on: they carry the same unidentified label plus the walk's
    own internal freedom, so they are diagnostics, not acceptance criteria.
    """
    deterministics = {k: relabelled[k] for k in ("spectra", "concentrations", "sigma")}
    latents = {k: relabelled[k] for k in ("theta_init", "curvature", "c_raw")}
    det = summary(deterministics, prob=0.95)
    lat = summary(latents, prob=0.95)
    worst_det = max(float(jnp.nanmax(s["r_hat"])) for s in det.values())
    extra = mcmc.get_extra_fields(group_by_chain=True)  # ty: ignore[unresolved-attribute]
    labels = [[int(i) for i in p] for p in permutations]

    print("\n--- gate evidence ------------------------------------------------")
    print(f"wall clock (fit only)    : {wall_clock:.1f} s")
    print(f"divergences per chain    : {[int(d) for d in extra['diverging'].sum(axis=1)]}")
    print(f"worst R-hat (determ.)    : {worst_det:.4f}")
    print(
        "worst R-hat (raw latents): "
        f"{max(float(jnp.nanmax(s['r_hat'])) for s in lat.values()):.4f}  (not asserted)"
    )
    print(
        f"min ESS (determ.)        : {min(float(jnp.nanmin(s['n_eff'])) for s in det.values()):.1f}"
    )
    print(f"mean tree depth          : {float(jnp.log2(extra['num_steps'].mean() + 1)):.2f}")
    print(f"per-chain labelling      : {labels}   disagree = {labels[0] != labels[-1]}")
    print(f"posterior sigma mean     : {float(relabelled['sigma'].mean()):.3e}")
    print("------------------------------------------------------------------")
    return worst_det


@pytest.mark.xfail(
    strict=True,
    reason="gate failure: spectra recover, but concentration coverage is ~0.43, not >0.9",
)
def test_recovers_easy_synthetic_data(fitted: Fitted) -> None:
    # Chains are aligned to the truth, so pooling them compares like with like.
    spectra = _pool(fitted.spectra).mean(axis=0)

    # Compare shapes, not magnitudes: after mean-one normalisation the scale has moved
    # into the concentrations.
    for resolved, true in zip(spectra, fitted.scaled_truth, strict=True):
        correlation = jnp.corrcoef(resolved, true)[0, 1]
        assert float(correlation) > 0.98, "resolved spectrum does not match its species"

    lower, upper = jnp.percentile(_pool(fitted.concentrations), jnp.array([2.5, 97.5]), axis=0)
    inside = (fitted.true_concentrations >= lower) & (fitted.true_concentrations <= upper)
    assert float(inside.mean()) > 0.9, "truth outside the 95% interval too often"


@pytest.mark.xfail(strict=True, reason="gate failure: NUTS does not mix on the v0 model")
def test_chains_converged(fitted: Fitted) -> None:
    """R-hat is checked, not merely printed -- spec.md section 8 and principle 6.

    A fit that has not converged cannot support any statement about posterior width,
    which is the whole deliverable.
    """
    assert fitted.worst_r_hat < 1.05, f"worst R-hat {fitted.worst_r_hat:.3f}; chains have not mixed"
