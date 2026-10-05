# Feasible-band curve resolution — Spec

**Status:** ready · **Date:** 2026-10-05 · **Component:** SpectraHandler

The decision this spec implements is [ADR 0003](../../docs/decisions/0003-feasible-band-not-posterior.md):
report the band of every spectra/profile split the data allow, not a posterior under a
prior that quietly picks one. Data loading lives here, per
[ADR 0002](../../docs/decisions/0002-scope-boundary-against-mcrals.md).

It **supersedes** [`plans/001-bayesian-curve-resolution/spec.md`](../001-bayesian-curve-resolution/spec.md)
§4 (the random-walk model), §7 steps 4–7 and the model in §10. It **keeps** §10's goals —
closure in real units, spectra shared across runs, Probe c as a falsification test — and
reaches them with constraints on the split instead of priors.

The executable task breakdown is [`plan.md`](plan.md). Evidence behind the choice:
[`report-spectrum-descriptions.md`](../001-bayesian-curve-resolution/report-spectrum-descriptions.md)
and its [`bench/results/`](../001-bayesian-curve-resolution/bench/results). The bench
scripts were never committed (`bench/.gitignore`).

---

## 0. Constraints inherited from the repo

From [`CLAUDE.md`](../../CLAUDE.md), restated once:

- Python ≥ 3.13, `uv` only. JAX for array maths; NumPy at IO boundaries and in tests.
  **No SciPy.**
- float64 (`tests/conftest.py` enables it; never at library import).
- Explicit PRNG keys; no Python loops over array axes; `.at[].set()`, no mutation.
- Full annotations, Google docstrings, line length 100, ruff + ty clean.
- Tuning knobs are arguments with defaults. Never silently drop data.
- A public API change updates `docs/src/content/docs/` in the same piece of work.

## 1. Goal and scope

Three deliverables:

1. **`resolve_band`** — given a `SpectralDataset` with closure (from `initial_state`) and
   any reference spectra, return the feasible band for every concentration and spectrum
   value, the ambiguity part without noise, and flat draws as a labelled
   typical-solution summary.
2. **JASCO readers** — the two export layouts in `tests/data/README.md`, as plain arrays.
3. **Rank and noise diagnostics** — `noise_diagnostics` and an optional `plot_noise`
   figure: singular values against the white-noise edge, the residual after K
   components, and its autocorrelation. Every split fits the data equally well, so the
   fit cannot choose a split, but it can say whether K is right and whether the noise is
   what the band assumes. matplotlib becomes an optional `plot` extra.

Plus retiring v0, and a docs guide.

**Gate:** the band contains the synthetic truth (≥ 0.95 of amounts and of spectrum
values; measured 1.000) on the bench's two datasets, its width matches the bench, the
flat summary covers an interior truth, and a run lacking a species narrows the band.
`tests/test_band_coverage.py` encodes all four.

**Non-goals for this plan** — each needs the gate's evidence first:

- Real Probe a/c/d resolution: six species, two closure pools, per-run offsets and
  scattering, deep-UV artefacts, time alignment, binning, and a noise estimate that
  survives interpolated exports. Next plan.
- Shape constraints and optimisation-based band edges — deferred, see §7.
- QC (saturation, noise along time, baseline-window checks). Next plan, with real data.
- Smoothing the spectral patterns (the one place smoothness can help, report §C.3).
- Sampling `φ` instead of the linearised noise margin.
- Kinetic constraints on `T`; unknown rank; masked (ragged) runs.

## 2. The method

All of it is in `src/spectrahandler/curve_resolution/band.py`. Notation: `n_row =
n_run·n_time` stacked time rows, `K = n_species`, `σ` the noise sd.

**Data matrix.** Time rows stacked over runs. Each species with a finite reference row
adds one more row, weighted so its noise is `σ` too:

```
w_k   = σ / reference_sigma[k]
row   = w_k · reference_spectra[k]          composition  w_k · e_k
```

**Patterns.** Rank-K SVD of the data matrix: `X = U_K·s_K`, `Y = V_Kᵀ` (orthonormal
rows). `σ` is estimated from the discarded singular values,
`σ² = Σ_{i>K} s_i² / ((n_row − K)(n_wavelength − K))`, unless the caller passes it.

**Equalities on `vec(T)`** (row-major):

- Closure: `X·T·1 = totals` for every row, solved in least squares as `T·1 = t*`,
  `t* = lstsq(X, row_totals)`. Row totals are `initial_state[run].sum()` for time rows and
  `w_k` for reference rows. Hence closure holds *in least squares* — single rows scatter
  at noise level (measured: 12.31–12.71 µM around 12.5).
- References: `x_refᵀ·T = w_k·e_k`.

Stack as `M·vec(T) = rhs`. Particular solution `T_p = lstsq(M, rhs)` (minimum norm, so
orthogonal to the null space). Null space `N` from the SVD of `M`, rank cut at
`1e-8·s_max`. `T(θ) = T_p + θ·N`. `n_free = K² − rank(M)`: 6 with closure only, 4 with
one reference, for K = 3.

**Feasibility.** `C = X_time·T`, `S = T⁻¹·Y`, with propagated noise sd

```
sd_C[k]  = σ·‖T[:, k]‖                       per species, all times
sd_S[k]  = σ·√Σⱼ (T⁻¹[k, j] / s_j)²          per species, all wavelengths
```

`violation(θ)` sums `max(0, −(value + z·sd)) / sd` over every entry of `C` and `S`, and
is infinite where `|det T| < 1e-12`. Feasible means violation exactly zero.

**Starting point.** Reference rows are pure by construction; each remaining species is
assigned, greedily, to the time row least explained by the rows picked so far, with
composition `total·e_k`. Solve for `T`, project onto the equalities, then run `n_als`
sweeps of MCR-ALS in the reduced space (clip `C`, refit `T`; clip `S`, refit `T`;
project). Then adaptive random search on `violation`, `n_restart` restarts in parallel
(`vmap`), up to `n_search` steps each. No feasible point → `ValueError`.

**Exploring `Θ`.** Hit-and-run in `θ`: random direction, distance to the boundary each
way by doubling then 25 bisections, then a uniform position on that bracket. If the
position is infeasible, shrink the bracket towards the current point and redraw (slice
sampling, Neal 2003). The extremes of every `C` and `S` entry at the bracket ends are
kept as the band. `lax.scan` over `n_iter` steps.

**Outputs** (`FeasibleBand`): band = extremes ∓ `z·median(sd)`, lower clipped at 0;
ambiguity = extremes alone; draws = walk positions after `n_burn`, every `thin`-th, plus
Gaussian noise at the propagated sd; `sigma`; `n_free`.

**Labels.** Columns of referenced species carry their names. Unreferenced species are
labelled by the greedy start and can come back swapped among themselves (measured: `b`
and `c` swapped on every synthetic run). Documented, not hidden.

## 3. Data contract change

`SpectralDataset` already carries `reference_spectra (n_species, n_wavelength)` and
`reference_sigma (n_species,)`; this plan gives them meaning and checks them:

- `reference_spectra`: absorbance **per `concentration_unit`** (per cm), i.e. a scan
  divided by the concentration it was taken at. A row is all finite or all `NaN`.
- `reference_sigma`: the per-channel noise sd of that row, same units. Finite and
  positive wherever the row is finite.

## 4. Readers

`src/spectrahandler/jasco.py`, NumPy only, no preprocessing:

- `read_interval_scan(path) -> Scan` — one run, times in **minutes** from the first data
  row, wavelength sorted ascending, absorbance `(n_time, n_wavelength)`.
- `read_spectrum(path) -> (wavelength_nm, absorbance, header)` — one spectrum, footer
  ignored. For blanks and reference scans.
- `read_spectrum_series(paths) -> Scan` — one file per time, time from the filename
  (`_15min`, `_2h`), ordered by time, never by name.

`Scan(wavelength_nm, time_min, absorbance, metadata)`. Building a multi-run
`SpectralDataset` from several scans — alignment, trimming, blanking — is the next plan.

## 5. Evidence gathered while writing the plan

The plan's code was prototyped in JAX and run before the plan was written; every number
quoted in a test docstring was measured on it.

| Check | Result |
| --- | --- |
| Band coverage, 8 seeds × {realistic, harder} | 1.000 / 1.000 on all 16 (bench: 1.00 / 1.00) |
| Median amount band | 1.97–2.12 µM realistic, 4.44–4.61 µM harder (bench: 2.0, 4.5) |
| Flat-draw coverage, amounts / spectra | realistic 0.93–0.98 / 0.82–0.89; harder 0.98–1.00 / 0.95–0.98 |
| Wall clock per fit | ~1.5 s (first call ~3 s, compilation) |
| `n_free` | 4 with one reference, 6 without |
| Full suite with every task applied, v0 retired | 77 passed, ruff and ty clean, docs build |

Figure: [`figures/band_port.png`](figures/band_port.png) — the band against the truth on
both sets (seed 0), and run 0 alone vs. jointly with a run lacking `a` (4.51 → 3.08 µM on
that exact data; the test asserts the narrowing on it).

**Rank and noise diagnostics** (figures `figures/noise_*.png`):

| Data | above the white-noise edge | residual lag-1 autocorrelation, wavelength / time |
| --- | --- | --- |
| synthetic, white noise | 3 | −0.07 / −0.09 |
| synthetic + drifting offset | 4 | 0.19 / 0.13 |
| real Probe a, ≥ 340 nm, rank 6 | 10 | **0.96** / 0.44 |

The real-data row is the interpolated-export trap from `tests/data/README.md`, seen
directly: the residual is smooth along wavelength, the rank-residual σ (2.0e-4 AU) is
far too low, and so is any band built on it. Its residual map also shows a step at
500 nm and structure in the early timepoints. None of this blocks the synthetic gate;
all of it is input to the real-data plan.

Three things the prototype found that the bench had not:

1. **A stuck-chain bug, inherited from the bench's `remix.py` (never committed).** `Θ` is
   not convex. A uniform step
   could land in a gap; from an infeasible point every bracket is empty, so the chain
   never moved again — 89% infeasible draws for one chain key, flat coverage 0.25. The
   band survived only because its extremes were collected before the chain stuck.
   Shrinkage fixes it: 1.000 for every chain key tried. Regression-tested.
2. **No reference → no start.** The bench's start knew which synthetic rows were pure.
   A generic greedy start plus random search fails with six free numbers (violation
   stuck at 27). Running MCR-ALS in the reduced space first fixes it in ~2 s.
3. **Shared spectra help only via absence.** A second run with only different rates:
   band 4.49 → 4.40 µM. A second run in which `a` never appears: 4.49 → 2.99 µM, coverage
   still 1.000.

## 6. Test layout

| File | What it pins |
| --- | --- |
| `tests/test_dataset.py` | reference rows all-finite-or-NaN; `reference_sigma` positive where used |
| `tests/test_band.py` | shapes, `n_free`, σ estimate, band ⊇ clipped ambiguity, closure, draws reproduce the data, reference inside its band, error paths |
| `tests/test_band_coverage.py` | **the gate**: coverage, width, interior-truth flat coverage, multi-run narrowing |
| `tests/test_jasco.py` | readers against the committed fixtures and their documented traps |
| `tests/test_diagnostics.py` | white noise reads as K; an offset reads as K + 1; the real export is not white |
| `tests/test_plot.py` | the noise figure renders with three panels |

Total suite runtime with these: ~30 s.

## 7. Open questions

**Deferred by decision (2026-10-05), not open:**

- **Shape constraints** — unimodal profiles, zero windows, monotonicity. They plug into
  the same feasibility check. Measured on the bench data, 3 seeds: every-profile
  unimodality narrowed the amount band 4.54 → 3.90 µM (harder) and 2.04 → 1.78 µM
  (realistic), truth still inside; "a only decreases" added nothing (closure and the
  reference already imply it). Each must be a true fact about the chemistry, so they
  arrive as opt-in flags. Decide the set with the real-data plan.
- **Optimisation-based band edges** (MCR-BANDS style, one small constrained optimisation
  per value, `vmap`ped). Hit-and-run gives an inner approximation and slows as free
  numbers grow; real Probe a has about twenty. The noise margin's median-sd shortcut
  also moved the spectral band 8% wider under a constraint, which an exact per-value
  margin would not. Decide with the real-data plan.

**Open:**

- **matplotlib as an optional extra.** This plan adds `spectrahandler[plot]`; the
  library imports without it. Alternative: no plotting in the library at all, figure
  code in the docs only.

- **NumPyro dependency.** After v0 is retired nothing in `src/` imports NumPyro. Keep it
  for the kinetic stage (this plan's default), or `uv remove numpyro` now and add it back
  then?
- **CLAUDE.md wording.** "NumPyro for inference — report posteriors wherever the science
  allows it" should say that the band is the output where the split is not identified.
  The user's call; not edited by this plan.
- **Naming unreferenced species.** `initial_state` at `time == 0` is a known composition
  and could act as one more equality, which would pin names in the synthetic data. Is
  the first scan ever truly at the initial state in real experiments? Probe a's first
  read is at 0 min but after mixing.
- **Defaults at higher `n_free`.** `n_iter = 8000` is tuned for four free numbers. The
  real-data plan must measure how the band's width converges with `n_iter` before
  trusting it at twenty.

## 8. Gate result

*To be filled in by plan Task 3, step 4.*
