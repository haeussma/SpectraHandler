# Kinetic fit v1: first-order schemes, per replicate, uncertainty from replicates — Spec

**Status:** draft, plan written · **Date:** 2026-10-06 · **Component:** SpectraHandler

Implements [ADR 0005](../../docs/decisions/0005-kinetic-fit-per-replicate-uncertainty-from-replicates.md)
(accepted 2026-10-06). [`plan.md`](plan.md) is written and unblocked.

This plan runs before [plan 003](../003-backgrounds-zero-window/spec.md) (backgrounds). v1 needs no
background handling: in a closed scheme a static offset is absorbed into the species spectra (§2.3).
Plan 003's reader deliverable (`read_kinetic_studio`, its §4) moves here, because v1's real-data
check needs it. Plan 003 keeps everything else.

The worked case is the PsVAO stopped-flow set (EXP0001). It is **local-only**: gitignored under
`local/`, never a fixture, never in the docs. The prototype is `local/psvao_analysis/v1_prototype.py`.

---

## 0. Constraints inherited from the repo

As in plan 002 spec §0 (removed; `git show 712b8dd:plans/002-feasible-band/spec.md`):
- `uv` only; JAX maths, NumPy at the boundaries, no SciPy; float64;
- explicit keys; no Python loops over array axes (a loop over the handful of runs is allowed: runs
  are fitted independently and differ in size);
- full annotations and Google docstrings;
- tuning knobs as arguments; never silently drop or interpolate data;
- a public API change updates `docs/src/content/docs/` in the same piece of work.

**The PsVAO data never enters git.** Tests that need it skip when `local/psvao_stopped_flow` is
absent.

## 1. Goal and scope

A researcher with replicate time courses of absorbance spectra, and a first-order reaction scheme,
gets:
- rate constants with an honest interval;
- the species spectra with a band;
- the concentration profiles;
- diagnostics that show where the model does not fit.

Deliverables:
1. **`Scheme`** (`kinetics/scheme.py`): a first-order reaction scheme from a list of steps, and
   its concentration profiles in closed form.
2. **`fit_kinetics`** (`kinetics/fit.py`): maximum likelihood per run (ADR 0005), returning a
   `KineticFit`.
3. **`KineticFit.summary()`** (`kinetics/summary.py`): replicates combined per condition into a
   `ConditionSummary`.
4. **Condition labels on `SpectralDataset`**, plus `SpectralDataset.from_runs` to stack recordings
   of different lengths without hand-padding.
5. **`read_kinetic_studio`** (moved from plan 003 §4, unchanged there).
6. **`plot_kinetic_fit`** (`plot.py`): traces with the model, deconvolution, and residual map.
7. **A guide page**, `docs/src/content/docs/guides/kinetic-fit.md`, on synthetic data.

**Gate:**
- **CI, synthetic:** recovery, invariance and coverage (§6).
- **Local, EXP0001:** reproduces the prototype numbers in §5.

**Non-goals (v1):**
- **No baselines or backgrounds.** Static ones are absorbed (§2.3). Drifting ones are plan 003.
- **No calibrated noise covariance; no binning inside the fit.** The user may bin beforehand.
- **No NUTS and no hierarchical model** (ADR 0005, options 2 and 3).
- **No controls in the fit** (enzyme blank as a reference spectrum, a fitted time zero). That is
  v1.1, prototyped in `local/psvao_analysis/t0_from_blank.py`.
- **No second-order or general ODE kinetics.**
- **No spectra shared across runs; no relative delays between runs.**
- **No non-negativity on spectra.**

## 2. The method

### 2.1 Model, per run r

```
D_r(t, λ) = Σ_s c_rs(t) · S_rs(λ) + ε,     ε ~ N(0, σ_rλ²), independent
c_r(t)    = expm(K(k_r) · (t + t_offset)) · c_r(0)
```

- `K` is the rate matrix of the scheme. A step `(a, b)` with rate `k` adds `−k` at `K[a, a]` and
  `+k` at `K[b, a]`. Columns sum to zero, so the scheme is closed: `Σ_s c_rs(t) = Σ_s c_rs(0)`.
- `c_r(0)` is the run's `initial_state`, which the dataset already carries.
- `expm` is `jax.scipy.linalg.expm`, vmapped over time. It stays robust for equal or reversible
  rates, where an eigendecomposition is defective.

### 2.2 Likelihood

Given `k_r`, the spectra solve a least-squares problem column by column: `S_r = C_r⁺ D_r`.
`σ_rλ` profiles out per wavelength, which leaves:

```
ℓ_r(log k) = −(n_r / 2) · Σ_λ log RSS_rλ(k)
```

`n_r` is the number of measured time points; masked points are excluded. The parameters are
`log k`, maximised by damped Newton (Levenberg–Marquardt) with the exact JAX gradient and Hessian,
starting from `initial_rates`. `jax.scipy.optimize.minimize` with BFGS was rejected: in the
prototype its line search failed from reasonable starts on A→B→C, while damped Newton converged
from every start. "Converged" means the Hessian is positive definite and the Newton decrement is
below 10⁻⁶ log-likelihood units. The within-run sd of `log k` comes from the inverse Hessian at the
optimum.

**No calibration step.** The noise scale per wavelength is estimated in the fit itself. The prototype
gives the same rates unbinned (3.155) and with 20-channel bins (3.151), so binning is the user's
choice for speed only.

### 2.3 What the data cannot tell apart (documented, tested)

- **Static offset.** Closure makes any static spectrum `B(λ)` equal to `B/Σc(0)` added to every
  species spectrum. Rates are unchanged; the spectra carry the run's baseline. EXP0001 shot 1:
  k = 3.16651 with and without an added offset + tilt.
- **Time shift.** A shift multiplies each decay-associated spectrum by a constant, which the free
  species spectra absorb. Rates and the spectra of end products are unchanged; the spectra of the
  starting species and of every intermediate describe the state at the first time label (A→B→C,
  shift 5 %: A off 90 %, B off 29 %, C 1e-7). EXP0001 shot 1: k = 3.16651 with time + 20 ms. `t_offset` exists for users who
  know their dead time from a calibration.
- **Rate swap in sequential schemes.** For A→B→C with a free B spectrum, `(k1, k2)` and `(k2, k1)`
  fit identically: synthetic log likelihood 214656.7966 both ways. After the fit, `fit_kinetics`
  evaluates every permutation of the fitted rates (up to 6 steps, so at most 720). If one fits as
  well, within `ambiguity_tolerance` = 10⁻³ log-likelihood units, `KineticFit.ambiguities`
  records it and the summary warns. v1 does not choose between them.
- **Rank.** If `C_r` loses column rank (two equal rates in a chain, or a species never populated),
  the spectra are not identified. The fit raises with the species named.

### 2.4 Combining replicates (ADR 0005)

Per condition, over its runs r = 1…n:
- **Rates, on log k.** Mean `m` and SE `s/√n`. Reported as the geometric mean `exp(m)`, with the
  95 % interval `exp(m ± t_{0.975, n−1}·SE)`. Student-t quantiles come from bisection on
  `jax.scipy.special.betainc`, so no SciPy is needed.
- **Spectra, per wavelength, linear.** The mean of `S_r` with the same t interval.
- **Validity check.** The ratio between-run sd / mean within-run sd of log k. Below 3 → warning:
  "between-replicate spread is not clearly larger than the fit precision; the interval ignores
  within-run uncertainty". EXP0001 gives 7.
- **n = 1.** The rates come with the within-run sd only, `interval=None`, and an explicit note:
  "no replicates — no replicate-based uncertainty".

## 3. API

```python
from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.kinetic_studio import read_kinetic_studio
from spectrahandler.kinetics import Scheme, fit_kinetics
from spectrahandler.plot import plot_kinetic_fit

scans = [read_kinetic_studio(p) for p in paths]  # KineticStudioScan per shot
ds = SpectralDataset.from_runs(
    [(s.time_s, s.absorbance) for s in scans],
    wavelength=scans[0].wavelength_nm,
    species=("E_red_QM", "E_start"),
    initial_state={"E_start": 7.0},  # same for all runs, or a list
    conditions=["250uM"] * 4,
    run_ids=["shot1", "shot2", "shot3", "shot5s"],
    time_unit="s",
    concentration_unit="uM",
)
scheme = Scheme(steps=[("E_start", "E_red_QM")])
fit = fit_kinetics(ds, scheme, initial_rates={("E_start", "E_red_QM"): 3.0})
summary = fit.summary()  # {condition: ConditionSummary}
summary["250uM"].rates  # {("E_start", "E_red_QM"): RateEstimate(value, lower, upper, ...)}
plot_kinetic_fit(fit, ds, wavelengths=[364, 441])
```

- **`Scheme(steps)`.** A frozen dataclass. Its species are the names in the steps. `fit_kinetics`
  raises if they differ from `ds.species`. A duplicate step raises; `(a, b)` plus `(b, a)` is a
  reversible pair.
- **`fit_kinetics(ds, scheme, *, initial_rates=None, t_offset=0.0, max_iter=200,
  ambiguity_tolerance=1e-3, rank_tolerance=1e-10)`.**
  - `initial_rates` defaults to `1 / (median run duration)` for the first step and a factor 3
    slower for each further step. Distinct starts keep A→B→C off its symmetric line.
  - It raises unless JAX runs in float64, like `resolve_band`.
  - Rates are in `1 / ds.time_unit`.
  - Returns `KineticFit`.
- **`KineticFit`** holds, per run:
  - `rates` (n_run, n_step) and `log_rate_sd` (within-run);
  - `spectra` (n_run, n_species, n_wavelength), in absorbance per concentration unit;
  - `concentrations` (n_run, n_time, n_species);
  - `residuals`, `sigma` (n_run, n_wavelength), `log_likelihood`;
  - `converged`, `ambiguities`.

  Methods: `summary()`.
- **`ConditionSummary`** holds `runs`, `rates` (a dict of `RateEstimate`: value, lower, upper,
  between_sd_log, within_sd_log, n), `spectrum_mean`, `spectrum_lower`, `spectrum_upper`, and
  `warnings: tuple[str, ...]`.
- **`SpectralDataset`** gains `conditions: tuple[str, ...]`, one per run. The default puts every
  run in one condition, `"all"`. `from_runs` pads unequal lengths with a mask; it never truncates.

## 4. Data contract change

`conditions` is added to `SpectralDataset`, validated as one label per run. It is backwards
compatible: `create` gains `conditions=None`. ADR 0001's layout is unchanged; this is one more
per-run attribute next to `run_ids`. `initial_state` is already per run, so different start
concentrations per condition need no new field.

## 5. Evidence (prototype, EXP0001, local)

Each shot alone, no baseline, unbinned 340–700 nm, noise sd per wavelength profiled:

| shot | k (s⁻¹) | within-run sd |
|---|---|---|
| shot 1 | 3.167 | 0.005 |
| shot 2 | 3.131 | 0.005 |
| shot 3 | 3.197 | 0.005 |
| 5 s shot | 3.127 | 0.004 |

**Geometric mean 3.155 s⁻¹, 95 % [3.104, 3.208].** The between-shot spread is 1.0 % against 0.15 %
within, a ratio of 7.

| variant | rates | geometric mean |
|---|---|---|
| 20-channel bins | 3.170 / 3.126 / 3.182 / 3.126 | 3.151 |
| one sd for all wavelengths | 3.211 / 3.177 / 3.243 / 3.109 | 3.185 (weighting matters) |
| earlier shared-spectra fits | 3.07, shot 3 an outlier at 2.98 | — |

The shared-spectra fits disagree because those cross-shot constraints pushed shot-specific
amplitude and timing into the rate. ADR 0005, finding 4.

## 6. Test layout and gate

`tests/kinetics/test_scheme.py`, `test_fit.py`, `test_summary.py`; `tests/test_dataset.py` extended;
`tests/test_kinetic_studio.py` with a small synthetic CSV written in the test. Never PsVAO data.

CI gate (synthetic, from `make_kinetic_replicates`, a new generator next to `synthetic.py`):
1. **Closed form:** `Scheme` concentrations for A→B and A→B→C equal the analytic expressions
   (rtol 1e-10). Reversible A⇌B reaches `k_b/(k_f+k_b)`.
2. **Noise-free recovery:** rates to rtol 1e-6, spectra to rtol 1e-6.
3. **Invariance:** an added static offset + tilt, and a time shift of 2 % of the duration, change
   the rates by less than 1e-6 relative.
4. **Coverage:** 200 conditions × 4 replicates, true log rates varying between replicates
   (sd 2 %), noise as in `make_realistic_dataset`. The 95 % interval contains the true geometric
   mean rate in 0.92–0.98 of conditions.
5. **Rate swap:** A→B→C with a free intermediate is flagged in `ambiguities`; A→B is not.
6. **n = 1:** `interval` is `None` and the note is present. The validity warning fires when the
   between-run spread is set to 0.
7. **Masks:** a padded run gives the same rates as the unpadded run.

Local gate (skips without data): EXP0001 reproduces §5 to ±0.002 per shot.

## 7. Open questions

1. **Shared spectra across replicates.** Is 3.155 (independent) or 3.07 (shared) the better
   pseudo-true rate? The difference is about 3 %, larger than the v1 interval. v1 takes the
   assumption-free variant. Controls in v1.1 (enzyme blank as reference, fitted time zero) are the
   way to put the shared structure back on evidence.
2. **Rate swap.** Report-only in v1. Breaking it needs spectral information: non-negativity, a
   reference spectrum, or an isosbestic point.
3. **Log vs linear combination of rates.** Log is chosen: rates are scale parameters, consistent
   with the log-uniform prior rule. For spreads of a few %, the two agree to 1e-4.
4. **Weighting.** The profiled σ per wavelength is principled under independent noise. Channel
   correlation (detector smoothing) leaves the rate unbiased but makes the within-run sd optimistic.
   That only matters for the diagnostic.
5. **Replicate meaning.** The docs must say that the interval covers only what varies between the
   declared replicates.
