# Bayesian Curve Resolution — Spec

**Status:** draft · **Date:** 2026-09-20 · **Component:** SpectraHandler

Scope: the **inference module only**, and within it only the stage with **no kinetic
model** — spectra and concentration profiles, nothing about rate constants, stoichiometry
or ODEs.

**Guiding rule for v0: get one fit running at all.** Every modelling choice is taken at
its simplest until a synthetic round-trip works end to end. Realism is added one term at a
time afterwards, each with a before/after comparison. A model that is wrong but runs
teaches more than a correct model that won't sample.

The one thing that is *not* simplified is the data layout, because kinetics and multi-run
pooling arrive later and must slot in without a breaking change.

Background and prior-art rationale: **not yet written.** When it is, it goes in
`research.md` beside this file — not under `docs/`, which is the published site.

The executable task breakdown derived from this spec is [`plan.md`](plan.md).

---

## 0. Constraints inherited from the repo

These are not restated per task. From [`CLAUDE.md`](../../CLAUDE.md):

- Python ≥ 3.13, `uv` only. JAX for all array maths; NumPy at IO boundaries and in tests.
  **Do not import SciPy** — it is present transitively via JAX, which is not permission.
- `jax.config.update("jax_enable_x64", True)`. float32 destroys least-squares residuals.
  Enabled in `tests/conftest.py`; never forced at library import time.
- Full type annotations on every parameter and return, including `-> None` (ruff `ANN`).
- Google-style docstring on every public module, class and function (ruff `D`).
- Line length 100. Arrays annotated `jax.Array`, with shape and units in the docstring.
- Explicit PRNG keys threaded through; never a global seed, never `numpy.random`.
- `vmap` over batch axes; no Python loops over array axes. No in-place mutation.
- Numerical assertions use `pytest.approx` or `numpy.testing`, never `==`.

---

## 1. Complexity ladder

The whole spec in one table. Left column is v0; nothing in the right column is built until
the left column runs.

| Component | v0 — simplest thing | Later |
| --- | --- | --- |
| Noise | **one shared scalar σ** | smooth σ(λ); then absorbance-dependent photometric error |
| Baseline | **none** | smooth per-run baseline |
| Reference spectra | **none** — carried in the dataset, not used by the model | soft likelihood terms with per-species σ_ref |
| Concentration profiles | **independent per timepoint**, non-negative, no smoothness | smoothness prior; then extents; then ODE |
| Spectral smoothness | 2nd-order RW **directly on the binned wavelength grid**, τ fixed | Matérn GP with a learnable length-scale (affordable at 64 channels); spline basis if sampling is too slow |
| Runs | array carries the run axis; **fit with n_run = 1** | shared S across runs, pooled fit |
| Wavelength grids | **must already match** — raise otherwise | resample at construction |
| Masking | implemented, but **v0 requires all-True** | ragged runs, padded |
| Rank diagnostic | none | effective-rank report per run and pooled |
| Species count | **given by the caller** | shrinkage prior, posterior over count |
| Selectivity windows | **none** | pin a species where only it absorbs; the controls make the windows findable |
| Test data | **easy synthetic**: well-separated spectra, low noise | overlapping spectra, realistic noise, real fixtures |
| Wavelength resolution | **binned to ~64 channels** | full 1 nm grid, once it is affordable |

Deliberately kept even in v0, because without them the fit is not identified at all:
non-negativity, and unit-sum normalisation of each spectrum.

---

## 2. What this stage does

Given absorbance over reaction time, recover pure component spectra `S` and concentration
profiles `C`, with posteriors.

**Out of scope:** rate constants, reaction networks, stoichiometric matrices, equilibrium
constants, ODE solving, file readers.

A consequence worth stating plainly: with no file readers, **v0 cannot load the real
fixtures in `tests/data/`.** Those stay unreachable until the boundary question in
[ADR 0002](../../docs/decisions/0002-scope-boundary-against-mcrals.md) is settled. v0 is
validated on synthetic data only, by design, not by oversight.

---

## 3. Data representation

Everything is a dense array. No dataframes, no dicts of runs, no ragged nesting. This
section is the contract and is **not** simplified for v0.

Locked in [ADR 0001](../../docs/decisions/0001-array-layout-and-canonical-ordering.md).

### Canonical shape

```
absorbance : (n_run, n_time, n_wavelength)   float64
time       : (n_run, n_time)                 float64
wavelength : (n_wavelength,)                 float64   # shared across runs
mask       : (n_run, n_time)                 bool      # True = measured
```

**Dimension order is (run, time, wavelength).** Rationale:

- The run axis is a batch axis, and leading batch dimensions are the JAX and NumPyro
  convention — `vmap` over runs is then free.
- Per run the slice is `(n_time, n_wavelength)`, which is exactly `C @ S.T` with `C` of
  shape `(n_time, n_species)` and `S` of shape `(n_wavelength, n_species)`. No transposes
  inside the likelihood.
- Absorbance is the stored value, not an axis.

A single-run experiment is `n_run = 1`, never a 2-D special case. One code path.

### Shared wavelength grid

`wavelength` is shared across runs and strictly increasing. **v0 requires the grids to
match already and raises otherwise**; resampling at construction comes later.

Reason for sharing: `S` is shared across runs, which is the whole reason the run axis
exists. A per-run grid would mean interpolating `S` on every gradient evaluation.

### Ragged time

The layout supports padding to `n_time = max` with a boolean `mask`. Padded entries are
`NaN`, never zero — zeros are valid measurements and silently corrupt a fit.

**v0 asserts `mask.all()`.** The machinery is present so the contract is stable; the code
path is exercised later.

### Species and the initial state

```
species       : tuple[str, ...]              # sorted, unique
initial_state : (n_run, n_species)           float64
```

**Invariant: `species` is sorted alphabetically, and every array with a species axis is in
that order.** Enforced at construction, not by convention. This gives a canonical order
independent of input order, so `initial_state` columns, `S` columns and later the
stoichiometric matrix columns all line up.

Two consequences:

- The permutation from user order to canonical order is recorded, and outputs always carry
  `species` names. Nothing returns a bare array whose column meaning depends on memory of
  the input.
- Alphabetical order is not chemically meaningful. Plotting takes an optional
  `display_order` and reorders at the presentation layer only; internal arrays are never
  reordered.

`initial_state` is **carried but unused in v0** — there is no kinetic model to consume it.
It is in the contract so that adding one is not a breaking change.

### Reference spectra

```
reference_spectra : (n_species, n_wavelength)  float64   # NaN where unmeasured
reference_sigma   : (n_species,)               float64   # NaN where unmeasured
```

One row per species, `NaN` marking "no reference". Avoids a separate index list drifting
out of sync with `species`. Also **carried but unused in v0**.

### Units

Stored explicitly, not assumed: `time_unit`, `wavelength_unit`, `concentration_unit`.
Strings only in v0 — no conversion logic.

### The object

A frozen dataclass `SpectralDataset` holding exactly the arrays above plus
`run_ids: tuple[str, ...]`. Immutable, validated once at construction, serialisable.
Inference functions take one of these and nothing else.

### Validation at construction

Fail loudly at build time, never mid-sampling:

- shapes mutually consistent
- `wavelength` strictly increasing, finite
- `time` non-decreasing within each run where `mask` is True
- `absorbance` finite wherever `mask` is True
- `initial_state` finite and non-negative
- `species` sorted, unique, non-empty
- `n_run == len(run_ids)`
- at least one `True` per run in `mask`

---

## 4. Model, v0

```
θ_k  ~ 2nd-order random walk over the wavelength grid, τ FIXED
S_k  = softplus(θ_k) / Σ             # unit-sum normalisation fixes scale
C    ~ softplus(Normal), independent per (run, time, species), non-negative
σ    ~ HalfNormal(σ_scale)           # ONE scalar for the whole dataset
A    ~ Normal(C @ S.T, σ)
```

That is the entire v0 model. No baseline, no references, no per-wavelength noise, no
smoothness on `C`, no sampled `τ`, **and no basis expansion** — the random walk runs
directly over wavelength channels.

**Why these three survive the cull:**

| Kept | Why it cannot be dropped |
| --- | --- |
| Non-negativity (softplus) | Negative concentrations and absorptivities make the fit meaningless and the posterior multimodal |
| Unit-sum normalisation of each spectrum | Without it `C·a, S/a` is exactly unidentified and the sampler wanders the scale direction forever |
| Smoothness on `S` (2nd-order RW) | Without it each channel is free and the decomposition is far less identified; spectra come out spiky and the sampler wanders |

**Why there is no spline basis.** An earlier draft expanded each spectrum in ~10 spline
coefficients, justified as *reducing* complexity against ~500 free wavelengths. That
argument only holds at 500 channels. §8 already says to bin to 4–5 nm for development,
which leaves ~64 — and 64 free channels per species with a random walk on them is both
fewer moving parts and less code than a basis plus its knot placement and `n_basis`
choice. A B-spline basis is a real optimisation, but it is an optimisation: it buys
sampling speed, not correctness, and it is the first thing to reach for if NUTS is too
slow. Binning is what makes the simple version affordable, so binning is in v0 and the
basis is not.

A straight line in wavelength is *not* an option for `S` — a line cannot represent an
absorption band, so the model could not express the thing it is fitting. Linear models
belong to the baseline, which v0 does not have at all.

### What the prior on `S` actually is, and what it is for

Spline, random walk and Gaussian process are **not three competing choices** — they are
three parameterisations of one smoothness prior, differing in cost, not in what they
express:

| Form | Relationship | Cost at `n_wavelength` |
| --- | --- | --- |
| 2nd-order random walk on the grid | discrete integrated Wiener process | O(n), sparse, what v0 uses |
| P-spline (B-spline basis + difference penalty) | the same penalty on `k << n` coefficients | O(k) |
| GP with an integrated-Wiener kernel | the continuous object; its posterior mean **is** a cubic smoothing spline | O(n³) dense |

The equivalence is exact, not loose: a second-order random walk is the state-space form
of that GP, and the posterior mean of that GP is the cubic smoothing spline
(Kimeldorf & Wahba). A P-spline is the same penalty applied to basis coefficients. So
"are we fitting a spline?" and "should we use a GP?" have the same answer — we are
already doing both, in the cheapest available form.

**A Matérn GP is genuinely affordable here**, and worth considering at step 4+. At the
binned 64 channels a dense 64×64 Cholesky per species per gradient evaluation is
nothing; at the raw 551 it is not. What it buys over the fixed-`τ` walk is a
**length-scale that is interpretable and learnable** — it is a band width in nm, so it
takes a real prior ("UV/Vis bands are 15–30 nm wide") and can be read off the posterior.
What it costs is a hyperparameter with a known funnel pathology. v0 keeps `τ` fixed for
that reason; the GP is the principled upgrade, not a different idea.

### The prior on `S` is regularisation, not identification

This is the part that matters, and it is easy to get backwards.

Rotational ambiguity is `D = C·Sᵀ = (C·T)(S·T⁻ᵀ)ᵀ`. **A rotation of smooth spectra is
still smooth.** A smoothness prior therefore barely shrinks the feasible set — it makes
the posterior geometry tractable and stops the spectra coming out spiky, and that is all
it does. No amount of tuning `τ`, and no upgrade from walk to GP, addresses
identifiability.

What the literature says does address it, in rough order of strength:

- **Several runs sharing one `S`.** Called "the strongest practical constraint against
  rotational ambiguity" in `~/code/mcrals`, and it is why the run axis exists in §3.
- **A hard kinetic model on `C`.** Removes the free-profile problem outright, and is the
  stage after this one.
- **Anchored or known reference spectra.** Already carried in the dataset (§3), unused.
- **Selectivity / local rank** — a wavelength region where only one species absorbs.
- Ambiguity is driven by **spectral overlap**: with low overlap the profiles come out
  nearly unique, with high overlap substantial ambiguity survives every soft constraint
  (Olivieri 2025, doi:10.1016/j.aca.2025.343897; de Juan & Tauler 2020,
  doi:10.1016/j.aca.2020.02.048).

Every one of these sits in the **Later** column of §1, while the smoothness refinements
sit in v0. That ordering is right for "make it run", and wrong for "make it mean
something" — so once the gate in §7 passes, the next steps are 5 and 6, not 7.

Note also what canonical MCR constrains with (de Juan & Tauler 2020): non-negativity,
unimodality, closure, selectivity and equality to known spectra. Smoothness is an
available secondary constraint, not the headline one. v0 has non-negativity and
normalisation.

**Unimodality does not apply to the spectra in this system.** It means literally one
maximum, and cobalamins have the classic γ band near 350–390 nm plus α/β bands near
470–560 nm. The resolved spectra in `tests/data/probe_a/reference_mcrals_figure.png`
carry two annotated maxima each for cob(I) (385, 550 nm), cob(II) (381, 472 nm) and
Co(III) aquo/hydroxo (352, 523 nm). Constraining `S` to be unimodal would forbid the
correct answer. Never apply it here.

Its legitimate home is the **concentration profiles**: an intermediate in a consecutive
scheme rises and falls once, which is standard MCR practice and visibly true of cob(II)
and Co(III) in panel B of that figure. Even there it is an assumption about the
mechanism that a hard kinetic model supersedes — and it fails for any network where a
species is consumed and later regenerated.

The cheap strong constraint on `S` for this system is **selectivity / local rank**: a
wavelength window in which only one species absorbs pins that species outright. The
controls make those windows findable — Ti(III) citrate alone gives the 351 nm region,
and the α/β region above ~500 nm is cobalamin-only. Worth trying before anything exotic,
and unlike smoothness it genuinely narrows the feasible set.

### A branch not taken yet: a parametric peak model

Instead of a nonparametric prior, each spectrum could be a **sum of a few Gaussians in
wavenumber** — bands are approximately Gaussian in energy, not in wavelength. Roughly
three parameters per band, so comparable in size to the spline it would replace, but
every parameter is interpretable, and literature band positions become priors
(cob(I) near 385 and 550 nm, and so on — the annotations in
`tests/data/probe_a/reference_mcrals_figure.png` are exactly these).

Crucially this **does** attack rotational ambiguity, because a rotated mixture of
few-Gaussian spectra is generally not itself a few-Gaussian spectrum. The cost is
misspecification when band shapes are not Gaussian, and a much stronger commitment to
the chemistry being right.

Not in v0 — it is a different model, not a simplification of this one. Worth an explicit
comparison once the gate passes.

**Why one shared σ is defensible as a starting point:** it is wrong in a known direction.
Real DAD noise varies with wavelength and grows with absorbance, so a single σ over-weights
noisy regions and under-weights clean ones. On easy synthetic data with uniform noise it is
*correct*, which is exactly what v0 needs — any failure is then the sampler or the
parameterisation, not the noise model.

### Known limitation, documented not hidden

Without a kinetic model the free-profile decomposition is subject to rotational ambiguity,
and a single run cannot resolve it. v0 is expected to produce wide and partly arbitrary
posteriors on *hard* data. It is validated on **easy** data — well-separated spectra, low
noise — where it should succeed. Failure there means something is broken; success there
means nothing about hard cases.

---

## 5. Module layout

```
src/spectrahandler/
  curve_resolution/
    __init__.py
    dataset.py        # SpectralDataset, validation, canonical sort
    model.py          # NumPyro model
    inference.py      # NUTS driver
    synthetic.py      # generate synthetic datasets for tests and demos
```

`basis.py` arrives only if profiling says the random walk over channels is the
bottleneck. `priors.py` and `diagnostics.py` arrive when there is more than one prior
choice and more than R̂/ESS to report.

`synthetic.py` lives in the package, not in tests — the suite must run before any real
fixture is reachable, and synthetic data is also how the later coverage check is run. Note
the cost: it ships in the wheel and becomes supported public API.

---

## 6. Test layout

The repo already has `tests/data/` with real fixtures and a provenance README; synthetic
output joins it rather than starting a second convention.

```
tests/
  conftest.py                 # exists: data_dir, key fixtures, x64
  data/
    probe_a/                  # exists: real interval scans
    1a/                       # exists: real single spectra, negative fixture
    synthetic/                # new: generated, regenerable, git-ignored
  test_dataset.py             # shape/validation invariants, canonical sort
  test_model.py               # shapes, prior predictive sanity
  test_recovery.py            # synthetic round-trip
```

No `tests/__init__.py` — with the src layout and pytest's default import mode it is not
needed and confuses rootdir resolution.

Priorities in order:

1. **Dataset invariants.** Every validation rule in §3 gets a failing case. Canonical
   sorting gets a test with deliberately unsorted input.
2. **Prior predictive sanity.** Draws from the prior should look like plausible
   non-negative spectra and profiles. Catches most parameterisation bugs before any data
   is involved, and costs nothing.
3. **Synthetic recovery on easy data.** Known `S` and `C` in, truth inside the 95%
   interval out.
4. **Masking equivalence** — once masking is switched on: a padded run must give
   bit-identical results to that run supplied alone and unpadded. Otherwise this surfaces
   months later as "results depend on which runs were in the batch".

Synthetic fixtures are generated in-test from a fixed seed, not committed. Real spectra
stay in `tests/data/` with the provenance README already there.

---

## 7. Build order

| Step | Deliverable | Done when |
| --- | --- | --- |
| 0 | `SpectralDataset` + validation + tests | All §3 invariants enforced and tested |
| 1 | `synthetic.py`, easy regime | Produces a dataset with known `S`, `C`, uniform noise |
| 2 | Prior predictive | Prior draws look like plausible UV/Vis spectra; `τ` picked from them |
| 3 | **v0 model + NUTS, n_run = 1** | **Synthetic round-trip recovers `S` and `C` within CI. This is the milestone.** |
| 4 | Smoothness on `C`, sampled `τ` | Posterior narrows or stays equal; no divergences |
| 5 | Reference spectra terms | Ablation: posterior width with vs. without. Attacks rotational ambiguity directly |
| 6 | Multi-run, shared `S` | Posterior narrows measurably vs. single run. The strongest soft constraint there is — consider promoting it above 4 |
| 7 | Smooth σ(λ) | Changes conclusions or does not — either is a result. **Lowest priority**: it refines the noise model, not identifiability |
| 8 | Harder synthetic: overlapping spectra, realistic noise | Honest failure modes documented |

Step 3 is the gate. Nothing past it is written until it passes. [`plan.md`](plan.md)
covers steps 0–3 only, for that reason.

---

## 8. Practical notes for getting the first fit to run

- **Bin the wavelengths.** At 1 nm over 300–800 nm you have ~500 points and are badly
  oversampled relative to the width of a UV/Vis band. Binning to 4–5 nm for development
  costs nothing in information and cuts likelihood cost several fold. Propagate the noise
  correctly when binning.
- **Short chains during development.** 200 warmup / 200 samples to find crashes and
  divergences; full runs only once it works.
- **Non-dimensionalise absorbance and time** to roughly O(1). Sampler geometry improves
  markedly and it costs one line.
- **Fix the seed** in every test that runs a sampler.
- **Report R̂ and ESS on every fit** from the start. Most of this literature reports trace
  plots or nothing; the habit is free and puts the library ahead of the field.

---

## 9. Open questions

Resolved into ADRs, no longer open:

- ~~Dimension order~~ → [ADR 0001](../../docs/decisions/0001-array-layout-and-canonical-ordering.md),
  **proposed** — confirm before Task 0 is written; changing it later is a wide refactor.
- ~~Canonical species order~~ → same ADR.

Still open:

- **Where does data loading live?** v0 declares readers out of scope, but something must
  turn `tests/data/probe_a/*.csv` into a `SpectralDataset` eventually. Depend on `mcrals`,
  absorb it, or write a reader here?
  → [ADR 0002](../../docs/decisions/0002-scope-boundary-against-mcrals.md), unresolved.
  Does not block steps 0–3.
- The fixed `τ` for v0. Pick from prior predictive draws rather than by argument — this
  is what Task 2 is for. Note that `τ` is curvature *per channel*, so it scales roughly
  with the square of the bin width: rebinning changes it.
- At what channel count does the random walk over wavelengths become the bottleneck, and
  is a spline basis then worth reintroducing? Profile before assuming.
- Should step 6 (multi-run, shared `S`) be promoted ahead of step 4? It is the strongest
  constraint against rotational ambiguity and the data for it already exists — three
  simultaneous runs in `tests/data/probe_a/`. The counter-argument is that it needs the
  reader question in ADR 0002 settled first.
- Is a parametric peak model the better answer for this chemistry than any nonparametric
  prior? It would break rotational ambiguity rather than merely regularise. Decide by
  comparison after the gate, not by argument.
- Does `C` need a smoothness prior at all once kinetics arrive, or does the ODE replace it
  entirely? Probably the latter — so step 4 may be throwaway.
- Whether v0's single σ should be sampled or fixed to a measured blank estimate. Fixed is
  simpler and prevents σ absorbing model misfit. `tests/data/probe_a/` has a buffer blank
  that would give a real number for this.
