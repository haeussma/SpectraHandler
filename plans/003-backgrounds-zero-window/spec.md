# Backgrounds and zero windows — Spec

**Status:** draft · **Date:** 2026-10-05 · **Component:** SpectraHandler

**Blocked on** execution of plan 002 (done, removed), because this extends
its `resolve_band`. [ADR 0004](../../docs/decisions/0004-backgrounds-need-a-zero-window.md)
was accepted on 2026-10-05. `plan.md` gets written once plan 002 lands.

The decision this spec implements is ADR 0004: backgrounds are known shapes with fitted
amplitudes, identified only through a declared wavelength range where no chemical species
absorbs. The baseline error left over is measured there as δ and carried into the band.
It removes the baseline assumption that [ADR 0003](../../docs/decisions/0003-feasible-band-not-posterior.md)
lists under Consequences ("a baseline or per-run offset shows up as an extra component").

The worked case is the PsVAO stopped-flow set (EXP0001). It is **local-only**: it is
gitignored under `local/`, never a fixture, and never in the docs. The scripts that
produced every number below are in `local/psvao_analysis/`.

---

## 0. Constraints inherited from the repo

As in plan 002 spec §0 (removed; `git show 712b8dd:plans/002-feasible-band/spec.md`):
- `uv` only, JAX maths, NumPy at the boundaries, no SciPy, float64;
- explicit keys, no Python loops over array axes;
- full annotations and Google docstrings;
- tuning knobs as arguments, never silently drop or interpolate data;
- a public API change updates `docs/src/content/docs/` in the same piece of work.

Additionally: **the PsVAO data never enters git.** Tests that need it skip when
`local/psvao_stopped_flow` is absent.

## 1. Goal and scope

Four deliverables:

1. **`Background` and `fit_backgrounds`** (`curve_resolution/background.py`). The known-shape
   background components, and the zero-window fit that measures their amplitudes and δ.
   `fit_backgrounds` is public, so a blank can be made baseline-free before it becomes a
   reference spectrum.
2. **`resolve_band` gains `backgrounds`, `zero_windows`, `baseline_tolerance`,
   `baseline_smoothing`**, and `sigma` may be a per-wavelength array. `FeasibleBand`
   records what was assumed.
3. **`noise_by_wavelength`** (`diagnostics.py`): σ(λ) from first differences along time, from
   a blank or a quiet stretch. EXP0001's noise spans 30× across wavelength, and one scalar σ
   misstates the margin everywhere.
4. **`read_kinetic_studio`**: the TgK Kinetic Studio CSV export (§4). **Moved to plan 004**
   (2026-10-06), which runs first and needs it; §4 stays as its reference. Data loading lives
   here per ADR 0002, and the gate needs it.

Plus a section in the curve-resolution guide.

**Gate:**
- **CI, synthetic:** with injected per-run offsets, tilts and a fixed instrument wiggle, the
  band still contains the truth (≥ 0.95 of amounts and spectrum values) when given the
  backgrounds and a window. δ bounds what was injected.
- **Local, EXP0001:** the band reproduces the prototype numbers in §5.

**Non-goals:**
- **Closure groups** (closure over a subset of species). Probe a needs them; they get their
  own plan, with the real-data six-species work.
- **Per-species zero regions.** These constrain rotation, not baseline (plan 002 spec §7).
- **Shot-noise scaling** σ(λ, A) = σ_blank(λ)·10^((A − A_blank)/2), measured within ~10% on
  EXP0001. It belongs with the kinetic model, where noise enters a likelihood.
- **Correlated noise along wavelength** (lag-1 0.91, decorrelated after ~10 channels). For
  now the user bins (§3); see open question 1.
- **Non-negative background amplitudes** (scattering). Amplitudes are sign-free in v1, and
  a negative scattering amplitude is reported, not prevented.

## 2. The method

Model, with `r` run, `t` frame, `λ` wavelength:

```
A_r(t, λ) = Σ_k c_rk(t)·S_k(λ)  +  Σ_j g_rj(t)·B_j(λ)  +  noise
            chemistry             backgrounds: B_j given, g_rj fitted
            S ≥ 0, C ≥ 0,         g_rj constant in t ("per_run") or free ("per_time")
            closure               sign-free, outside closure
```

`resolve_band(data, key, backgrounds=..., zero_windows=...)`:

1. **Validate.**
   - `backgrounds` non-empty needs `zero_windows` non-empty.
   - Every window lies inside the grid and holds at least `n_background + 2` channels.
   - The background shapes restricted to the window channels have full column rank, with a
     condition number of at most 1e6. Otherwise: "offset and scattering are not separable
     in 640–660 nm; widen the window".
   - A measured shape must be on the dataset grid exactly. **Never interpolated.**
2. **Weights.** `w(λ) = 1/σ(λ)` when `sigma` is an array, else uniform.
3. **Fit amplitudes in the window** by weighted least squares.
   - `per_run` fits each run's time-mean spectrum over the window channels.
   - `per_time` fits each frame.
   - Mixed sets fit jointly, with the per-run columns constant across that run's frames.
   - One `vmap` over runs; no loops.
4. **Leftover and δ.**
   - `L_r(t, λ∈W)` = data minus fitted backgrounds.
   - `δ = max over r, t, λ∈W of |running mean over k frames of L_r(·, λ)|`, with
     `k = baseline_smoothing` (default 20, clipped to `n_time`).
   - This bounds static wiggle and drift together; on EXP0001 they add. The smoothing
     leaves a noise floor of about `4σ/√k`, which is reported, so δ errs on the safe side.
5. **Check against a given tolerance.** If `baseline_tolerance` is given and δ exceeds it,
   raise `ValueError`, naming the measured δ, its location (run, λ) and the given value.
   If not given, use the measured δ.
6. **Subtract** `Σ_j g_rj(t)·B_j(λ)` everywhere, then resolve as in plan 002, with two
   changes:
   - The data and references are scaled by `w`, and spectra and margins are unscaled
     afterwards. Positive column scaling leaves non-negativity unchanged.
   - Spectrum non-negativity becomes `S_k(λ) ≥ −(z_slack·sd_S + δ/N)`. `N` is the
     smallest closure total over runs. Concentration constraints are unchanged.
7. **Record** `background_names`, `background_amplitude` (`(n_run, n_time, n_background)`,
   AU), `baseline_tolerance` (δ, AU) and `zero_source`. `zero_source` is
   `"zero windows 520–700 nm, δ measured"`, `"zero windows …, δ given"`, or
   `"instrument zero (assumed), δ given"`.

**Without windows:** no backgrounds are allowed, and δ = `baseline_tolerance` or 0.

**Uncertainty of the amplitudes** is not propagated:
- a `per_run` amplitude has sd ≈ σ/√(n_time·n_W) ≈ 1e-5 AU on EXP0001, 400× below δ;
- a `per_time` amplitude has sd ≈ σ/√n_W ≈ 1e-4 AU, still below δ.

The docstring says so.

**Why the prototype's δ trick is not the implementation.** The prototype added δ to the
data and the references, so that under closure every spectrum could dip to −δ/N. That is
exact only when every run has the same total. The library relaxes the constraint
directly (step 6) instead.

## 3. API

```python
@dataclass(frozen=True)
class Background:
    """A known spectral shape whose amplitude is fitted. Outside closure, sign-free."""

    name: str
    shape: Callable[[Array], Array]          # wavelength -> (n_wavelength,), unitless
    amplitude: Literal["per_run", "per_time"] = "per_run"

    @classmethod
    def offset(cls) -> "Background": ...     # 1
    @classmethod
    def tilt(cls) -> "Background": ...       # λ − mean(λ)
    @classmethod
    def scattering(cls, exponent: float = 4.0,
                   amplitude: Literal["per_run", "per_time"] = "per_time") -> "Background": ...
    @classmethod
    def measured(cls, name: str, spectrum: Array, wavelength: Array,
                 amplitude: Literal["per_run", "per_time"] = "per_run") -> "Background": ...
        # raises at resolve time if `wavelength` differs from the dataset grid


@dataclass(frozen=True)
class BackgroundFit:
    corrected: Array        # input minus fitted backgrounds, same shape, AU
    amplitude: Array        # (..., n_time, n_background) or (..., n_background), AU
    tolerance: float        # δ, AU
    leftover_max_at: tuple[int, float]   # (run, wavelength) where δ was attained


def fit_backgrounds(absorbance: Array, wavelength: Array, *,
                    backgrounds: tuple[Background, ...],
                    zero_windows: tuple[tuple[float, float], ...],
                    sigma: float | Array | None = None,
                    smoothing: int = 20) -> BackgroundFit: ...
    # absorbance (n_run, n_time, n_wavelength) or (n_time, n_wavelength)


def resolve_band(dataset, key, *, ...,                     # plan 002 arguments unchanged
                 sigma: float | Array | None = None,       # now also (n_wavelength,)
                 backgrounds: tuple[Background, ...] | None = None,
                 zero_windows: tuple[tuple[float, float], ...] = (),
                 baseline_tolerance: float | None = None,
                 baseline_smoothing: int = 20) -> FeasibleBand: ...
    # backgrounds=None means (Background.offset(),) if zero_windows else ()


def noise_by_wavelength(absorbance: Array) -> Array: ...
    # (n_time, n_wavelength) -> (n_wavelength,): std(diff along time) / sqrt(2)
```

EXP0001 as user code (the guide uses synthetic data; this is the shape):

```python
blank = fit_backgrounds(
    enzyme_blank,
    wl,
    backgrounds=(Background.offset(), Background.tilt()),
    zero_windows=((520.0, 700.0),),
)
data = SpectralDataset.create(..., reference_spectra=[blank.corrected.mean(0) / 7.0, nan_row])
band = resolve_band(
    data,
    key,
    sigma=noise_by_wavelength(buffer_blank),
    backgrounds=(Background.offset(), Background.tilt()),
    zero_windows=((520.0, 700.0),),
)
band.baseline_tolerance  # 0.0039
```

Binning stays with the user: 10-channel means, 3 lines of NumPy, shown in the guide. See open
question 1.

## 4. Reader: TgK Kinetic Studio CSV

`read_kinetic_studio(path) -> (wavelength_nm, time_s, absorbance, header)`. `absorbance` is
`(n_time, n_wavelength)`. The layout, from the EXP0001 export (KinetAsyst SF-61DX2, DET2B
diode array):

```
File Info:"," DX2 Kinetics File:data11.ksd | 21/10/2025 14:55:43" Temp: -36.7,
,0,0.01,0.02, ... ,0.99          <- leading empty field, then times in SECONDS
300.21,0.21666,0.18396, ...      <- one row per wavelength, ascending, ~0.47 nm, 1098 rows
```

- `header` holds `file` (the `.ksd` name), `timestamp` and the raw `temperature` string.
  The instrument README says the temperature reading is unreliable, so it is **not** parsed
  into a number.
- An empty cell becomes `NaN`, never dropped. The README warns of saturated points in trace
  files.
- Rows with fewer than 3 fields are skipped. Single-wavelength trace files are out of scope.

**Traps the reader does not fix, but the docstring names:**
- The `_NNNs` in a filename is the acquisition **window**, not a time point.
- In EXP0001, shots with windows longer than 1 s label each frame one sampling step late.
  Shifting the 5 s shot by −0.05 s cuts its mismatch with the 1 s shots from 0.0146 to
  0.0042 AU rms.
- The reader returns times as recorded.

## 5. Evidence

**Measured on EXP0001** (three 1 s shots; 342–695 nm in 10-channel bins; σ(λ) from the
buffer blank; closure 7 µM; oxidised enzyme pinned by the enzyme blank):

| | no baseline handling | hand-tuned: line at t≈0, δ = 0.003 guessed | **this spec**: offset + tilt per run, W = 520–700, δ measured | Fraaije 1997 |
|---|---|---|---|---|
| feasible | no (−24σ at 668 nm) | yes | yes | — |
| δ (AU) | — | 0.003 | **0.0039** | — |
| converted at 1 s (µM) | — | 5.04–6.99 | 5.16–7.02 | — |
| ε364 of the intermediate (mM⁻¹ cm⁻¹) | — | 44.1–55.8 | 43.7–54.6 | 46 |
| ε439 of the oxidised enzyme | — | 12.6 | 12.4–12.5 | 12.5 |
| rank-2 residual / blank noise | — | 1.6 | 1.5 | — |

Findings that shaped §2:

- **The window holds a weak chemical signal.** A rise of 0.0021–0.0031 AU appears in
  520–700 nm only when enzyme and substrate react. The controls show ≤ 0.0006 AU and the
  blank 0.0000. It correlates 0.90 with the 439 nm loss, with a half-time of 0.12 s against
  0.24 s.
- **The static wiggle and the bias from fitting a constant across this drift add.** δ as
  the larger of the two (0.0034) left the problem infeasible at −5.8σ. The combined
  smoothed leftover (0.0039) made it feasible. Hence δ's definition in step 4.
- **A rank test of the window reads rank 6 where the backgrounds explain 2.** Hence the
  magnitude test.
- **Offset + tilt per run explains only 4–13% of each shot's static leftover.** It is
  wiggle, not low-order. More polynomial terms would chase instrument structure, so δ
  carries it instead.
- **A `per_time` offset + tilt removed the 0.0026 AU chemical rise entirely.** Hence
  `per_run` as the default.
- **Noise:** σ(λ) runs 0.0004–0.0028 AU per 10-channel bin, lowest at 500–600 nm.

**Synthetic window:** in `make_realistic_dataset`, every species is below 1e-4 AU at
≥ 660 nm, under 5% of the noise. That gives `W` = (660, 700), with 11 channels at
`n_wavelength=96`.

## 6. Test layout and gate

| File | Tests |
|---|---|
| `tests/test_background.py` | shapes (values, grid check raises); per-run offset + tilt recovered within 3σ/√(n_time·n_W); per-time drift recovered; δ ≥ injected wiggle amplitude and ≤ wiggle + reported noise floor; backgrounds without a window raise; a 1-channel window raises (not separable); a measured shape on the wrong grid raises |
| `tests/test_band_background.py` | **gate**: `make_realistic_dataset(n_wavelength=96)` plus per-run offsets ±0.005 AU, tilts ±0.003 AU/100 nm and a fixed ±0.002 AU wiggle. With `backgrounds=(offset, tilt)` and `zero_windows=((660, 700),)`, coverage of amounts and spectra ≥ 0.95. Without them, record whether it is infeasible or under-covers (measured when the plan runs). A species tail of 0.003 AU injected into W makes δ ≥ 0.003, and `baseline_tolerance=0.001` then raises. A 30× heteroscedastic noise with `sigma` as an array keeps coverage ≥ 0.95. |
| `tests/test_diagnostics.py` | `noise_by_wavelength` recovers an injected σ(λ) within 10% |
| `tests/test_kinetic_studio.py` | reader on a small file written to `tmp_path` in the §4 layout: shape, ascending grid, seconds, NaN for an empty cell, header fields |
| `tests/test_local_psvao.py` | skips without `local/psvao_stopped_flow`. EXP0001 as in §5: feasible; δ in [0.003, 0.005]; ε364 band contains 46; ε439 within 12.5 ± 0.5; shots agree at 1 s within 0.15 µM |

## 7. Open questions

1. **Binning helper.** The detector's noise is correlated over ~10 channels, and the band's
   margin assumes independence. Should there be a `bin_wavelength(dataset, n)` that records
   the binning in the dataset, or should the user keep binning by hand?
2. **The time-label offset of the long-window shots.** Ask the instrument owner
   (Tobias Rapsch) whether Kinetic Studio stamps frame ends. If confirmed, the reader can
   offer an explicit `time_offset`.
3. **The weak absorber in 520–700 nm.** It is unassigned, and too weak for the rank test
   over 342–695 nm. Is it a third species to model in the 5 s and 100 s shots?
4. **A blank as a known-composition run** instead of a reference row. That is more general:
   it carries its own baseline through the same fit. It changes plan 002's reference
   handling, so it is out of scope here.
5. **A committed test that reads a local-only path** (`test_local_psvao.py`) contains no
   data, only expected numbers. Acceptable?
6. **`baseline_smoothing = 20`** is in frames, so it depends on the acquisition rate. Should
   it be in time units instead?

### Carried over from plan 002's final review (2026-10-05)

Do these before or inside this plan:
- **Memory:** `band.py`'s `lax.scan` stores `parts(moved)` (full C and S) for all `n_iter`
  steps. That costs about 600 MB at 600 rows × 1000 channels with K = 6. Scan only `theta`,
  then `vmap(parts)` over `thetas[n_burn::thin]`.
- **Silent band truncation:** `edge` stops doubling at `1e4·unit`. When the region is
  unbounded (wrong rank, or a species absent from every run), the band is reported finite
  without saying so. Record a cap hit, and warn or raise.
- **Reader edge cases:**
  - an interval scan with only the time row and no `NPOINTS` raises `IndexError`;
  - a non-integer `NPOINTS` gives an error without the path;
  - `n_iter <= 0` produces an error message that names `n_burn`.
- **`numpyro` is still a runtime dependency** with no importer in `src/`. Dropping it is
  the user's call (plan 002 spec §7).

## 8. Gate result

Not run. Fill in when `tests/test_band_background.py` and `tests/test_local_psvao.py` pass.
