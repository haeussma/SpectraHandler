# Backgrounds and zero windows — Plan

Executes [`spec.md`](spec.md) and
[ADR 0004](../../docs/decisions/0004-backgrounds-need-a-zero-window.md) on top of the
credible band of [ADR 0007](../../docs/decisions/0007-credible-band-posterior-for-the-identified-part.md).
Both ADRs are accepted; nothing blocks.

## What changed against the spec (2026-11-19)

The spec was written against the feasible band of plan 002. With ADR 0007 in place:

- **`sigma` per wavelength is already there** and is inferred by default, so
  `noise_by_wavelength` is dropped. The window fit is weighted by `sigma` only when the
  caller gives one.
- **`z_slack` is now `z_region`.** Spectrum non-negativity becomes
  `S ≥ −(z_region·sd_S + δ/N)`.
- **δ also widens the spectrum band by δ/N on both sides.** A baseline error `v(λ)` that
  is constant in time is absorbed into the species spectra as `v/N`, so the truth lies
  within δ/N of the resolved spectrum. The spec only relaxed non-negativity.
- **Amplitudes are fitted by least squares in the window, not sampled.** Their standard
  error is about 1e-5 AU on EXP0001, 400× below δ; δ is the uncertainty that matters and
  no sampler measures it. Sampling them belongs with the kinetic model.
- **Scope cut for this pass:** `per_run` amplitudes only; shapes `offset` and `tilt`, plus
  any callable through `Background(name, shape)`. `per_time`, `scattering`, `measured`
  and the reader (already shipped with plan 004) are not built here.
- **The items "carried over from plan 002's final review" are not done here.**

## Tasks

1. **`curve_resolution/background.py`**: `Background` (`offset`, `tilt`), `BackgroundFit`,
   `fit_backgrounds(absorbance, wavelength, *, backgrounds, zero_windows, sigma, smoothing)`.
   Validation per spec §2 step 1. Tests: `tests/test_background.py`.
2. **`resolve_band`** gains `backgrounds`, `zero_windows`, `baseline_tolerance`,
   `baseline_smoothing`; `CredibleBand` gains `background_names`, `background_amplitude`,
   `baseline_tolerance`, `zero_source`. Tests: `tests/test_band_background.py` (the
   synthetic gate of spec §6).
3. **Guide section** "Baselines and backgrounds" with a runnable synthetic example.
4. **Local check on EXP0001** (three 1 s shots, 10-channel bins, 2 species, closure 7 µM,
   oxidised enzyme pinned by the baseline-corrected enzyme blank), as a script under
   `local/`, never committed. Compare with spec §5.

Commands: `uv run pytest`, `uv run ruff check --fix .`, `uv run ty check`.

## Gate result

Filled in below when the tasks have run.

Run on 2026-11-19. Numbers and the three deviations they forced are in the amendment of
ADR 0004.

- **Synthetic, CI** (`tests/test_band_background.py`): two runs with offsets of ±0.005 AU
  and a shared 0.002 AU wiggle, `backgrounds=(offset,)`, window 660–700 nm. Over four
  noise seeds the band held 0.99–1.00 of the true amounts and 1.00 of the true spectrum
  values; offsets recovered to 0.0001 AU; inferred noise 0.0021 AU (true 0.002).
  The spec's gate with tilts and this 40 nm window is not feasible, for the reason given
  in the ADR.
- **Local, EXP0001:** feasible, δ = 0.0040 AU; both literature values inside the bands.
  Not a committed test.

Left open: `per_time` amplitudes, more shapes, a committed local test, and the items
carried over from plan 002's review.
