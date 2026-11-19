---
status: accepted
date: 2026-10-05
source_paper: doi:10.1074/jbc.272.29.18111
implements: [src/spectrahandler/curve_resolution/background.py, src/spectrahandler/curve_resolution/band.py]
---

# 0004 — Backgrounds are known shapes, identified only through a declared zero window

## Context

Real time courses are not strictly bilinear. Beyond the chemistry they hold:

- a per-run baseline that differs from shot to shot or cuvette to cuvette;
- an instrument pattern that drifts over the day;
- sometimes scattering that grows over time.

ADR 0003 lists this as a baked-in assumption: "a baseline or per-run offset shows up as an
extra component". The first real dataset made it concrete. The PsVAO stopped-flow set
(EXP0001: 7 µM vanillyl-alcohol oxidase + 4-(methoxymethyl)phenol, three 1 s shots) is
local-only, not a fixture, under the data owner's terms. Its runs disagree by up to
0.016 AU in offset. The blank and the shots carry a fixed wiggle of ±0.0026 AU at 668 nm.
Without baseline handling, `resolve_band` finds no feasible split: the oxidised-enzyme
spectrum reads −24σ at 668 nm.

**A free baseline is not identifiable.** This follows from the algebra; the vault has no
primary source on it. A closed reaction obeys conservation laws, so some combination of
concentrations is constant within each run: `Σₖ aₖ cᵣₖ(t) = Nᵣ`. Take a baseline that is
constant in time with a free spectrum `bᵣ(λ)`. For any spectrum `v`:

```
Sₖ → Sₖ + aₖ·v,   bᵣ → bᵣ − Nᵣ·v
```

leaves `C·S + b` unchanged. A background with a known shape `B` and a free amplitude in
time has the same leak, with `v ∝ B`. Either way the absolute zero of every species
spectrum becomes free. That is a whole spectrum of ambiguity, not a few numbers.
Non-negativity bounds it from one side only.

**What breaks the leak** is a wavelength range `W` where no chemical species absorbs. Inside
`W` the data hold only background and noise, so the background amplitudes are measured
there. Their known shapes carry them across the rest of the spectrum. For flavins, `W` is
above about 520 nm (oxidised VAO ε439 = 12.5 mM⁻¹ cm⁻¹,
Fraaije & van Berkel 1997, doi:10.1074/jbc.272.29.18111).

**What the window cannot do.** EXP0001's window 520–700 nm is not truly empty. A
0.0021–0.0031 AU signal rises in it during each shot. It is absent from the
substrate-only controls (≤ 0.0006 AU) and from the enzyme blank (0.0000 AU). It
correlates 0.90 with the 439 nm loss but has half the half-time. So it is chemistry: a
weak, broad absorber that forms faster than the 364 nm intermediate. It is unassigned. Two
consequences follow:

1. **A rank test of "emptiness" is useless on real instruments.** Inside `W`, EXP0001 has
   rank 6 where the backgrounds explain 2. Static per-shot wiggles alone supply 3. The test
   has to be on **magnitude**.
2. **A per-timepoint background absorbs that chemistry completely.** A flat offset plus
   tilt per frame left nothing of the 0.0026 AU rise. Free-in-time amplitudes are exactly
   as dangerous as an extra species when their shape resembles a species' tail in `W`.

## Decision

1. **A background is a known spectral shape times a fitted amplitude.**
   - It sits outside closure, its amplitude is sign-free, and its spectrum is never
     non-negativity-tested, because it is given.
   - Amplitudes are `"per_run"` (constant within a run; the default) or `"per_time"` (one
     per frame; opt-in).
   - Shapes provided: flat offset, linear tilt, power-law scattering, and a measured
     spectrum on the dataset grid.
   - **There is no background with a free spectrum**, and no free "baseline species".
2. **Zero windows are an assumption passed to `resolve_band`, never stored in the
   dataset.** They are wavelength intervals, in the dataset's unit, where no chemical
   species absorbs. Backgrounds without a zero window raise `ValueError`. Without a window,
   zero is the instrument's zero, and the result says so.
3. **The window's emptiness is tested by magnitude against δ, not by rank.**
4. **δ, the baseline tolerance in AU, is measured in the window.** It is the largest
   time-smoothed leftover after fitting the backgrounds, over every run, frame and window
   channel. That bounds the static wiggle and any drift *together*, and on EXP0001 they
   add: 0.0026 + 0.0013 → measured **0.0039 AU**.
   - δ only relaxes non-negativity: species spectra may dip to −δ/N per unit
     concentration. It never enters the fit, so concentrations do not move with it.
   - A user-given δ smaller than the measured one raises `ValueError`.
5. **The result records its assumptions**: the backgrounds, their fitted amplitudes, the
   windows, δ, and where zero came from.

Validated on EXP0001 with offset + tilt per run and `W` = 520–700 nm:
- feasible, with δ = 0.0039 measured;
- 5.16–7.02 µM converted at 1 s, the three shots within 0.1 µM;
- ε364 of the intermediate 43.7–54.6 mM⁻¹ cm⁻¹, containing Fraaije's 46;
- ε439 of the oxidised enzyme 12.4–12.5, against Fraaije's 12.5.

The hand-tuned version (line baseline at t ≈ 0, guessed δ = 0.003) gave 5.04–6.99 µM and
44.1–55.8.

### Where this deviates from common practice

- **Baseline subtraction before resolution** (fixed windows, polynomial or asymmetric
  least-squares baselines) is the usual preprocessing. That claim is uncited: the vault holds
  no MCR baseline source yet. This ADR keeps the subtraction but makes three changes:
  - the window is a stated, tested assumption;
  - the subtraction's error bound δ travels into the band;
  - the backgrounds are recorded with the result.
- **An extra free component** for a baseline is also common practice (also uncited). It is
  rejected here: per the Context it is not identified, and on real data it absorbs
  chemistry.

## Consequences

- **The band's zero is only as good as the window.** A species that absorbs weakly in `W`
  biases the baseline by about its time-mean and inflates δ by its amplitude. On EXP0001
  that is about 0.003 AU: harmless against 0.1–0.3 AU signals, and it widens the band
  honestly. A species absorbing *strongly* in `W` shows up as a δ comparable to the signal.
  The docs must teach reading δ.
- **No window, no backgrounds.** For chemistry that absorbs across the whole range (some
  cobalamin species may: unverified), zero comes from the instrument or a blank, and δ must
  be given by hand.
- **`per_time` amplitudes are opt-in, and the docs require a control run** that shows the
  background without the chemistry. Probe f's turbidity is the intended use.
- **Reference spectra must already be baseline-free.** A blank is a measurement with its own
  baseline. The same fit is public (`fit_backgrounds`) so a blank goes through the window
  procedure before it becomes a reference.
- **Per-species zero regions** ("cob(II) is zero above 600 nm") are a different thing. They
  constrain rotation, not baseline, and stay deferred (plan 002 spec §7). The
  `zero_windows` name leaves room for them.
- **Assumes** backgrounds are linear in amplitude and that their shapes are linearly
  independent inside `W`. The library checks the second and raises.

## Amendment, 2026-11-19: built on the credible band of ADR 0007

Implemented in `background.py` and `resolve_band`. Three things differ from the Decision
above, each because of something measured while building it.

1. **δ does not bound the baseline error outside the window.** Decision 4 treats δ as the
   error everywhere. A leftover of size δ inside the window biases the fitted amplitudes,
   and the shapes carry that bias outwards; for a tilt it grows with the distance.
   On synthetic data with a 40 nm window, a fixed 0.002 AU wiggle and offset + tilt per
   run, δ was 0.003 AU while the baseline left outside the window reached 0.018 AU.
   The tolerance is therefore δ inside the window and `δ·(1 + leverage(λ))` outside, with
   the leverage the absolute row sum of the least-squares map from window data to the
   background at λ. That is a worst case: 2 δ for an offset, 4–5 δ at 364–439 nm for
   offset + tilt on EXP0001.
2. **The worst case can be replaced by a measurement.** `baseline_tolerance` also takes
   one value per wavelength, used as given. On EXP0001 the buffer blank, which has no
   absorber, leaves 1.4× its in-window tolerance outside the window after the same fit.
3. **δ also widens the spectrum band**, by δ/N on both sides, and not only the
   non-negativity test: a baseline error constant in time is absorbed into the species
   spectra as error/N, so the truth lies that far from the resolved spectrum.

Not built in this pass: `per_time` amplitudes, the scattering and measured shapes (any
shape can be given as a callable), and `noise_by_wavelength` (the noise per wavelength is
inferred by the model since ADR 0007).

**A narrow window cannot pin a tilt per run.** With 11 window channels over 40 nm and 30
spectra, noise alone gives the tilt a standard error of 0.003 AU across the grid, different
in every run, so the runs no longer share one set of spectra and no split is feasible. The
synthetic gate uses offsets only. Fitting the baselines of the runs relative to each other
inside the model, from the whole spectrum, would remove this limit; it belongs with the
kinetic stage.

**Without backgrounds, a baseline mismatch between runs is absorbed as noise.** Since
ADR 0007 infers the noise, EXP0001 without baseline handling no longer fails: it returns a
band with a median inferred noise of 0.0022 AU against 0.0008 AU with backgrounds. The
inflated noise is the only sign.

EXP0001 with the credible band (three 1 s shots, offset + tilt per run, window 520–694.5 nm,
δ = 0.0040 AU measured, R-hat 1.005):

| | worst-case tolerance | tolerance calibrated on the buffer blank | prototype above | Fraaije 1997 |
|---|---|---|---|---|
| converted at 1 s (µM), three shots | 4.4–6.9 | 4.9–6.9 | 5.16–7.02 | — |
| ε364 of the intermediate (mM⁻¹ cm⁻¹) | 41.5–64.1 | 43.7–57.1 | 43.7–54.6 | 46 |
| ε439 of the oxidised enzyme | 10.4–15.1 | 11.9–13.6 | 12.4–12.5 | 12.5 |

The shots agree within 0.1 µM in both runs.

## Amendment, 2026-11-19 (second): baselines between runs are inferred in the fit

The first amendment fitted every run's backgrounds separately in its own zero window.
That used only the window and ignored that the runs share their spectra. It is replaced.

1. **Each run after the first gets one amplitude per background relative to the first run,
   sampled in the NumPyro model** with a flat prior. The runs share `Y`, so a difference
   between runs that is constant in time and has a background's shape is identified from
   the whole spectrum. A start comes from alternating a rank-K approximation with a
   per-run least-squares fit; the model samples around it.
2. **This is the default**, with an offset and a tilt, and needs no window.
   `backgrounds=()` switches it off.
3. **The zero window fixes only what all runs share.** After the alignment one set of
   amplitudes is fitted in the window to the mean over all runs and subtracted. The
   tolerance and its leverage factor are as in the first amendment. Without a window the
   first run's zero is taken as it is, and the result says so.
4. **With a zero window every reference gets background amplitudes of its own**, sampled.
   The window moves the data's zero, and a reference that stays on its own zero then
   contradicts the data: with offset + tilt and a 40 nm window no split was feasible until
   the reference was freed. Its spectrum's spread is taken from the posterior of
   `x_ref @ Y`, which includes that freedom. A blank can then serve as a reference as
   measured. Without a window the references share the data's zero, as before.

**Why the reference is not freed without a window.** On clean single-run data the
reference's own offset came out 0 to 2.4 posterior sd from zero over three noise draws.
An offset moves every channel together, so one unlucky draw puts the whole spectrum of the
referenced species outside its band (0.05–0.81 of its values inside, where 0.98 was the
figure with the reference on the data's zero).

**A common baseline cannot come from a blank.** Every measurement has its own baseline, so
a blank fixes its own zero and nothing else. This corrects a suggestion made while
planning, that blanks as runs with known composition would pin the zero.

Synthetic (two runs 0.010 AU apart, shared 0.002 AU wiggle, two noise seeds):

| | amounts inside | spectrum values inside | offset of run 2 against run 1 (true −0.0100) | inferred noise / true |
|---|---|---|---|---|
| default, no window | 0.94–0.95 | 0.60 (zero unknown) | −0.0102 to −0.0104 | 1.04–1.05 |
| offset, window 660–700 nm | 0.99–1.00 | 1.00 | −0.0099 | 1.00–1.02 |
| offset + tilt, same window | 0.99 | 1.00, band 3 µM wide | −0.0098 to −0.0099 | 1.00–1.02 |
| `backgrounds=()` | no feasible split | | | |

EXP0001, binned by 10, three 1 s shots:

| | offset + tilt, window, blank-calibrated tolerance, enzyme blank as measured | default, no window | `backgrounds=()` |
|---|---|---|---|
| median inferred noise (AU) | 0.00083 | 0.00079 | 0.00224 |
| shots 2 and 3 against shot 1: offset | −0.0046, −0.0044 | −0.0045, −0.0044 | — |
| posterior sd of those offsets / tilts | 0.00002 / 0.0001 | 0.00002 / 0.00009 | — |
| converted at 1 s (µM) | 4.9–6.9 | 5.1–7.2 | 5.2–7.0 |
| ε364 of the intermediate (46) | 43.8–57.8 | 43.3–55.3 | 44.2–53.9 |
| ε439 of the oxidised enzyme (12.5) | 11.4–14.4 | 12.4–12.6 | 12.6–12.8 |

The default removes the mismatch between shots that `backgrounds=()` turns into 2.7 times
the noise. The first column's wider ε439 is the price of not trusting the blank's zero.

**Limits.** Runs that share no species give the alignment nothing to hold on to. The
uncertainty of the references' coordinates reaches the referenced species' spectra only,
not the other species or the concentrations. The baseline is constant within a run.
