# Bayesian Curve Resolution — Spec

**Status:** superseded in part · **Date:** 2026-09-20 · **Component:** SpectraHandler

> **Superseded 2026-10-05.** §4 (the random-walk model), §7 steps 4–7 and the model in §10
> are replaced by [ADR 0003](../../docs/decisions/0003-feasible-band-not-posterior.md) and
> [`plans/002-feasible-band/spec.md`](../002-feasible-band/spec.md). §3 (data layout) and
> the §7 gate record stand. This directory is deleted at the end of plan 002; its report
> and bench stay reachable in git history from plan 002 Task 0's commit.

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

## 0.5 Principles

These govern every version. A change that violates one of them is wrong, however well it
fits.

**1. Beer–Lambert bilinearity is the model, and it is an assumption with a stated limit.**
`A(t, λ) = Σ_k c_k(t) · ε_k(λ) · l`. Linear in both factors, which is the entire
structure being exploited. It fails above roughly 1.5 AU — the `1a` fixture is kept
precisely because it violates this.

**2. Every parameter is a physical quantity with units, or it is a nuisance term that has
to justify itself.** `ε` in M⁻¹cm⁻¹, `c` in µM. A parameter that means nothing cannot be
given a prior, checked against literature, or reported.

**3. Fix the scale with chemistry, not convention.** `C·a, S/a` fits identically, so
something must pin it. A unit-sum normalisation does that but leaves `C` in arbitrary
units. **Closure** — `Σ_{k ∈ pool} c_k(t) = c_total` over a conserved pool — does the same
job and leaves `C` in µM. Same number of constraints, strictly more meaning. Prefer it
wherever a total is actually known.

**4. Non-negativity yields a band of solutions, not a point** (Lawton & Sylvestre 1971).
This is the field's founding result and it is not a numerical nuisance to be tuned away.
Plan to report the band.

**5. Smoothness regularises; only structure identifies.** A rotation of smooth spectra is
still smooth. Closure, selectivity, shared spectra across runs, known references and a
kinetic model narrow the feasible set. Priors on shape do not.

**6. A narrow posterior on an ambiguous problem is a lie.** If the rotational ambiguity is
real and the posterior is tight, the tightness came from the prior, not from the data.
The point of doing this in NumPyro rather than ALS is **an honest width**, not a
narrower answer. The check is concrete: the posterior should contain the feasible band an
MCR-BANDS-style calculation returns. Much narrower means the priors are doing the work,
and that has to be said out loud.

**7. Validate where the answer is known.** Synthetic data with known truth first; then the
controls, which are an experimental version of the same idea — Probe c contains no
cobalamin, so any cobalamin species resolved into it is an artefact.

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

**Steps 5 and 6 together are v1** (§10), and they are the ones that make the result mean
something. Step 4 (smoothness on `C`) and step 7 (smooth σ(λ)) refine a model that is
not yet identified, so they come after. Reorder on the evidence from the gate, not on
this table.

### Gate result (step 3) — 2026-09-20

**The gate does not pass.** The spectra come back essentially exactly right; the sampler
does not mix, and the concentration intervals are about half the width they need to be.
Both tests in `tests/test_recovery.py` are `xfail(strict=True)` recording this. Evidence
only — what step 4 should be is not decided here.

Machine: Apple Silicon, 14 cores, `numpyro.set_host_device_count(2)`, 2 chains,
float64. Easy synthetic data: 1 run × 30 timepoints × 64 channels, σ_true = 2e-3 AU,
which is 1.689e-4 after `fit` divides by max|A| = 11.84. Truth in the model's units:
max c = 0.1374, and two of the 90 concentrations are exactly zero.

#### What shipped

| | |
| --- | --- |
| Configuration | `fit` defaults: τ = 0.01, 200 warmup, 200 samples, 2 chains |
| Wall clock | 6.7 s (fit only, after `block_until_ready`) |
| Divergences | 0, 0 |
| Worst R̂, relabelled deterministics (`spectra`, `concentrations`, `sigma`) | **4.446** |
| Worst R̂, raw latents (`theta_init`, `curvature`, `c_raw`), relabelled | **37.40** — reported, not asserted |
| Min ESS (deterministics) | **1.0** |
| Mean tree depth | 10.00 — pinned at the maximum |
| Step size | ~1e-4 |
| Spectral correlation per species | 1.00000, 1.00000, 1.00000 (five decimals) |
| Concentration coverage of the 95% interval | **0.20** |
| Posterior σ | 1.825e-4 vs true 1.689e-4 |

#### Everything tried, in the order the plan prescribes

All rows: 2 chains, relabelled per chain before pooling. "cover" is the fraction of the
90 true concentrations inside the pooled 95% interval. Wall clock is measured after
`jax.block_until_ready`.

Three rows need naming because they are not interchangeable. **Row A is the plan's code
as written**, before any deviation — original initial slope, centred curvature. Rows B–E
are that same code with warmup and τ varied. **Row SHIP is the configuration that
shipped** and is the "What shipped" table above: both model deviations applied, at the
plan's default 200/200. Every row from G down carries both deviations.

| # | Configuration | Wall | Div | R̂ det | R̂ latent | ESS | Step | Depth | min corr | cover | med width | med \|bias\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | **plan as written (pre-deviation)**, τ=.01, warm 200 | 6.7 s | 0,0 | 6.97 | 132.2 | 1.0 | 1.2e-4 | 10 | 1.00000 | 0.200 | 7.8e-5 | 6.3e-5 |
| B | τ=.01, warm 1000 | 31 s | 0,0 | 6.59 | 82.3 | 1.1 | 2.2e-4 | 10 | 1.00000 | 0.300 | 6.4e-5 | 4.2e-5 |
| C | τ=.01, warm 4000 | 31 s | 0,0 | 3.47 | 51.2 | 1.1 | 1.8e-4 | 10 | 1.00000 | 0.222 | 4.3e-5 | 3.4e-5 |
| D | τ=.03, warm 1000 | 31 s | 0,0 | 49.4 | 57.9 | 1.0 | 1.8e-4 | 10 | 0.99750 | 0.189 | 3.5e-4 | 7.3e-4 |
| E | τ=.05, warm 1000 | 31 s | 0,0 | 8.41 | 97.7 | 1.0 | 1.6e-4 | 10 | 0.99560 | 0.178 | 5.3e-5 | 9.9e-4 |
| **SHIP** | **both deviations, τ=.01, warm 200 — `fit` defaults** | **6.7 s** | **0,0** | **4.446** | **37.4** | **1.0** | ~1e-4 | 10 | 1.00000 | **0.200** | — | — |
| G | both deviations, τ=.01, warm 1000 | 32 s | 0,0 | 2.51 | 36.5 | 1.3 | 2.3e-4 | 10 | 1.00000 | 0.411 | 4.2e-5 | 2.3e-5 |
| **I** | **non-centred, τ=.01, max_tree_depth=14** | **486 s** | **0,0** | **1.020** | **1.06** | **221.5** | 1.1e-4 | 14 | 1.00000 | **0.433** | 4.3e-5 | 2.3e-5 |
| J | both deviations, τ=.01, dense_mass | 79 s | 40,11 | 1.048 | 1.22 | 18.0 | 1.5e-2 | 10 | 1.00000 | 0.422 | 4.3e-5 | 2.3e-5 |
| K | dense_mass, accept .95, warm 2000 | 121 s | 5,0 | 1.083 | 1.12 | 8.5 | 1.0e-2 | 10 | 1.00000 | 0.433 | 4.3e-5 | 2.3e-5 |
| L | dense_mass, τ=.03, warm 2000 | 122 s | 28,93 | 1.493 | 2.64 | 1.7 | 5.7e-3 | 10 | 1.00000 | 0.544 | 5.9e-5 | 2.6e-5 |
| M | dense_mass, τ=.05, warm 2000 | 122 s | 0,0 | 4.347 | 5.58 | 1.1 | 3.4e-3 | 10 | 0.99999 | 0.533 | 1.2e-4 | 3.9e-5 |
| N | dense_mass, τ=.10, warm 2000 | 122 s | 0,0 | 4.458 | 22.2 | 1.1 | 2.4e-3 | 10 | 1.00000 | 0.689 | 4.5e-5 | 1.3e-5 |
| P | both deviations, τ=.05, max_tree_depth=14 | 486 s | 0,0 | 2.113 | — | 1.3 | 2.2e-5 | 14 | 1.00000 | 0.589 | 5.6e-5 | 2.1e-5 |
| R | **centred in θ** (MVN on θ), τ=.01 | 44 s | 0,0 | 22.7 | — | 1.0 | 7.1e-4 | 10 | 1.00000 | 0.444 | 6.1e-5 | 2.6e-5 |
| S | centred in θ, τ=.05 | 44 s | 0,0 | 25.9 | — | 1.0 | 8.2e-4 | 10 | 0.99756 | 0.222 | 1.2e-3 | 5.9e-4 |
| T | centred in θ, τ=.03 | 43 s | 0,0 | 11.0 | — | 1.0 | 1.1e-3 | 10 | 0.99781 | 0.156 | 7.7e-4 | 9.7e-4 |

Step by step against the prescribed order:

1. **Divergences.** Zero in almost every configuration, including the converged one.
   Divergences are not the problem; the step size is. Tree depth is pinned at the
   maximum in every depth-10 run, which is the signature of a badly conditioned
   posterior rather than a badly behaved one.
2. **Raise `num_warmup`.** Monotone but weak: R̂ 6.97 → 6.59 → 3.47 for 200 → 1000 →
   4000 warmup, with ESS stuck at ~1. Warmup cannot adapt what the chain never explores.
3. **Tune τ from prior draws.** The plan's step 4 says a failing smoothness test means
   τ is too large. That is wrong here, and the measurement says so plainly: prior
   mean |∂²S| is **0.158 at every τ from 1e-4 to 1e-2**. The roughness came from the
   initial slope of the walk being left at unit scale (sd ≈ 1.4 per channel, ramping θ
   across ±90 over 64 channels, so softplus turns every draw into a hinge). After
   scaling the initial slope to `τ·√n_wavelength`, τ controls roughness as intended:

   | τ | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 |
   | --- | --- | --- | --- | --- | --- | --- |
   | prior mean \|∂²S\| | 0.0054 | 0.0117 | 0.0196 | 0.0288 | 0.0390 | 0.0498 |
   | prior mean peak height | 2.68 | 4.12 | 5.43 | 6.63 | 7.73 | 8.75 |

   The true mean-one spectra have peak heights 8.17, 6.62, 7.28, so prior-predictive
   matching picks **τ ≈ 0.04–0.05**, not the plan's 0.01. Raising τ does improve
   coverage monotonically (0.43 → 0.54 → 0.59 → 0.69), but every high-τ run failed to
   converge, so those widths are not evidence of anything. `test_spectra_are_smooth`'s
   threshold of 0.05 caps τ at about 0.06, so the two tests pull in opposite directions
   but do not yet conflict outright.
4. **Re-parameterisation and one prior change.** Two distinct things, and the
   distinction matters:

   - *A genuine re-expression:* making the curvature truly non-centred — the plan's
     comment claimed this, its code did not — improved worst R̂ from 6.59 to 2.51 for
     free. Same distribution, different coordinates. Likewise sampling θ directly under
     the equivalent multivariate normal (rows R/S/T; prior verified identical,
     mean |∂²S| 0.0056 vs 0.0057 at τ=.01), which was **worse**, R̂ 11–26. That
     hypothesis — that the likelihood dominates and so the centred form should win — is
     disconfirmed.
   - *A prior change, not a re-expression:* the initial slope. The plan had
     `slope0 = theta_init[:,1] − theta_init[:,0]`, sd **≈ 1.414 per channel** and
     correlated with the level; what ships is
     `slope0 = tau·sqrt(n_wavelength)·theta_init[:,1]`, sd **0.08 at τ=0.01** and
     independent of the level. That is a **~17× tighter prior on the tilt of θ**, not a
     relabelling of the same distribution. No term was added, so the no-new-terms rule
     holds, but the prior on `S` is tighter than the plan's — and the gate's failure is
     a too-narrow, biased spectral estimate, which is the same direction. It was forced:
     without it `test_spectra_are_smooth` cannot pass at any τ. See "What this run does
     NOT tell us" below.

#### The one thing that did converge, and what it proves

Row **I** is the only configuration in the whole search that converged: R̂ 1.020 on the
relabelled deterministics, 1.06 on the raw latents, ESS 221, zero divergences. It took
486 s. Its coverage is **0.433**. So the coverage shortfall is **not** an artefact of
non-convergence; a properly converged posterior is genuinely about twice too narrow.

An oracle run settles where the remaining error lives. Pinning the spectra at the truth
and sampling only the concentrations and σ (diagnostic only, not a candidate model):

| | Oracle (S pinned to truth) | Full fit, converged (row I) |
| --- | --- | --- |
| R̂ | 1.0017 | 1.020 |
| Coverage | **0.922** (0.943 excluding the two exact zeros) | 0.433 |
| Median interval width | 3.6e-5 | 4.3e-5 |
| Median \|bias\| | 4.8e-6 | 2.3e-5 |
| Posterior σ | 1.691e-4 (true 1.689e-4) | 1.782e-4 |

So the 0.9 coverage bar **is** reachable, the softplus concentration parameterisation is
calibrated, and the interval width is right. What breaks it is the spectral estimate:
bias is 5× larger in the full fit, and at 2.3e-5 against a half-width of 2.15e-5 the
posterior mean sits about one half-width from the truth, which is exactly what coverage
0.43 looks like. Note the spectral correlation is 1.00000 to five decimals even so —
correlation is far too blunt a measure at this SNR, where the concentrations demand the
spectral shape to ~1e-4 relative.

Two of the 90 true concentrations are exactly zero at t = 0, and `softplus` cannot reach
zero, so those two can never be covered: a 2.2% ceiling loss, confirmed by the oracle
(0.922 → 0.943 when they are excluded). That is a real but minor effect and not the
cause of the failure.

σ is recovered well throughout (1.69e-4 to 1.83e-4 against a true 1.689e-4), and it is
biased slightly *high* in the unconverged runs, which widens intervals — so it flatters
the coverage number rather than depressing it.

#### Deviations from the plan

Recorded in full in the commit message for `model.py` / `inference.py`. In brief:

- **The initial-slope scaling — a prior change, not a re-expression.** Slope sd
  ≈ 1.414 per channel and correlated with the level, becoming 0.08 at τ=0.01 and
  independent of it: a ~17× tighter prior on the tilt of `S`. Forced by the prior
  predictive, but it is a different model in the prior, and it is not innocent with
  respect to the failure observed.
- The truly non-centred curvature — a faithful re-expression of the same distribution.
- `extra_fields=("diverging", "num_steps")` on `fit`, so divergences are recoverable.
- `sigma_scale` and `max_tree_depth` exposed as keyword arguments on `fit` with their
  previous values as defaults, per the repo rule on tuning knobs.

And in the tests, the units conversion of the truth into the model's scaled units,
per-chain relabelling before pooling and before R̂, one shared module-scoped fit, and
stopping the clock after `jax.block_until_ready` (timing `mcmc.run` alone reports ~1 s
for a 31 s fit, because JAX dispatches asynchronously).

Two further notes on the plan's own test code. The units bug in the concentration
coverage assertion made it fail with certainty as written. And the per-chain labelling
disagreement it did not anticipate is real: the two chains were observed settling on
`[2,1,0]` and `[1,0,2]` in one run and agreeing on `[2,0,1]` in another, so pooled R̂
without alignment is sometimes spuriously large and sometimes not — the worst kind of
flake.

#### What this run does NOT tell us

- **Whether the tighter initial-slope prior contributed to the failure.** It is an
  untested confound. The change was forced — the prior predictive cannot pass without
  it — so no run in the table uses the plan's original slope prior *and* a converged
  sampler, and the two cannot be separated from this evidence. What can be said is only
  that the change tightens the prior on `S` by ~17× in the tilt direction and that the
  failure is a too-narrow, biased spectral estimate: the same direction. No causal claim
  is made here, and none is available from this run.
- **Nothing about hard or overlapping data.** This is the easy regime only: three
  well-separated bands, uniform noise, one run. §4 already says success here would say
  nothing about hard cases; failure here says the model or the sampler is broken, which
  is what was found.
- **Nothing about whether the width is honest** (principle 6). The posterior is too
  narrow relative to the *truth*, which is a different statement from being narrower
  than the feasible band. No MCR-BANDS-style calculation was run, so the comparison the
  principle demands has not been done, in either direction.
- **Nothing about τ at convergence.** Every τ above 0.01 failed to mix, so the
  prior-predictive-preferred τ ≈ 0.05 has never been evaluated with a converged chain.
  The single most informative missing measurement.
- **Nothing about multi-run, closure, references or selectivity** — none are in v0.
- **Nothing about real fixtures**; there is still no reader (ADR 0002).
- **Nothing about the cost at 551 channels.** All of this is at the binned 64.

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
  → [ADR 0002](../../docs/decisions/0002-scope-boundary-against-mcrals.md), **accepted**:
  readers live here; `mcrals` is deprecated.
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

---

## 10. v1 — the simplest model that makes mathematical sense

v0 is the simplest thing that *runs*. v1 is the simplest thing that is *defensible*, and
the difference is two changes, both of which **remove** arbitrariness rather than adding
machinery.

```
S_k(λ) ≥ 0, smooth                      # softplus + RW2, unchanged from v0
C ≥ 0                                   # softplus, unchanged
Σ_{k ∈ pool} c_k(t) = c_total           # CLOSURE replaces the arbitrary normalisation
S shared across runs                    # n_run > 1, one spectra matrix
σ ~ HalfNormal(σ_scale)                 # still one scalar
A ~ Normal(C @ S.T, σ)
```

**Change 1: closure instead of normalisation** (principle 3). v0 normalises each spectrum
to mean one, which pins the scale but leaves `C` in arbitrary units. Closure pins it with
a number that was measured: total cobalamin is 12.5 µM, from 25 µM hydroxocobalamin
diluted 2:1. `C` then comes out in µM and can be compared to anything.

Closure is **per conserved pool, not global**. In the Probe a system there are two pools:

| Pool | Species | Total |
| --- | --- | --- |
| cobalamin | cob(I), cob(II), Co(III) aquo/hydroxo, methylcobalamin | 12.5 µM, known |
| titanium | Ti(III) citrate, Ti oxidised | not known |

So closure fixes the cobalamin block in real units, and the titanium block still needs a
normalisation to fix its own scale. That asymmetry is honest — it reflects what was
actually measured — and it is exactly what `~/code/mcrals` does with
`closure_columns=(0, 1, 2, 4)`.

**Change 2: shared spectra across runs** (principle 5). This is the strongest soft
constraint available, the reason the run axis exists in §3, and the data already exists:
Probes a, c and d ran simultaneously in the cell changer against one set of pure spectra.
A component is then credible only if it appears where the chemistry that makes it was
present — Probe c has no cobalamin, so it is a direct falsification test.

**What v1 still does not have,** and why that is the honest boundary of this stage: no
kinetic model. Rotational ambiguity is reduced by closure and sharing, not eliminated.
Principle 6 therefore applies in full — v1's deliverable is a width, and that width
should be compared against a feasible-band calculation before anyone quotes it.

### The v1 acceptance test

Not "does it converge". Three things, in order:

1. **Recovers synthetic truth**, as v0 does, with closure active and `C` in the right
   units rather than an arbitrary scale.
2. **Probe c stays empty.** Fit a, c, d jointly; the resolved cobalamin concentrations in
   the Ti-citrate-only run must be indistinguishable from zero. This is the experimental
   version of principle 7, it costs nothing, and it is the check that would have caught
   the two misreadings recorded in `tests/data/README.md`.
3. **The posterior width is defended, not just reported.** State whether it contains the
   feasible band, or state that the comparison has not been done.
