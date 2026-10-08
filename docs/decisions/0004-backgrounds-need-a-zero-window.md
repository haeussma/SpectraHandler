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
