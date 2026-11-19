---
title: Curve resolution with honest uncertainty
description: Resolve a reaction time course into pure spectra and concentration profiles, and report the full range the data allow.
sidebar:
  order: 3
---

A UV/Vis time course is a mixture: at every time, the measured absorbance is the sum of
each species' concentration times its pure spectrum. Curve resolution splits the data
back into those two parts.

## Why you get a band, not a single answer

The data fix the **product** of concentrations and spectra, but usually not the split.
Mixing a little of one spectrum into another, and adjusting the concentrations to
match, reproduces the data exactly, and both spectra stay smooth and positive. No
amount of repeating the same measurement removes this; it is a property of the
experiment, not of the noise.

SpectraHandler therefore reports a **credible band**, built in two steps that keep
apart what the data determine and what they do not:

1. **What the data determine.** The product, and the noise level at every wavelength,
   get a Bayesian posterior (sampled with NumPyro). This is where the measurement
   noise enters.
2. **What they do not.** The split gets no prior at all. Every split that keeps
   concentrations and spectra non-negative is kept, where "non-negative" means: not
   below zero by more than the posterior of step 1 can explain.

The band is, for every concentration and every spectrum value, the range of its 95 %
posterior interval over all those splits. Its width is an honest statement of what
your data can and cannot tell you, provided that:

- the number of species you give is right,
- the noise is independent from point to point, with one level per wavelength,
- the absorbance is concentrations times spectra, plus only the backgrounds you name
  (see [Baselines and backgrounds](#baselines-and-backgrounds)), and
- closure holds as given.

The splits are explored by a random walk (hit-and-run), so the band is an inner
approximation that can fall slightly short of the true extremes. On the synthetic
tests the band contained at least 95 % of the true amounts and spectrum values.

Methods that return a single split (classic MCR-ALS, NMF), or one Bayesian posterior
under a prior on the spectra, pick one point or one narrow region inside that band
without telling you that they did.

## What narrows the band

Only information the time course does not already contain:

- **Closure.** The species of one pool sum to a known total, e.g. 12.5 µM cobalamin.
  Taken from `initial_state`: every run sums to `initial_state[run].sum()`.
- **Reference spectra.** Every species you can measure pure pins its own spectrum.
- **Runs that change which species are present.** All runs share one set of pure
  spectra. A run in which a species is absent cuts the band markedly; a run that only
  changes the rates barely helps.

## A runnable example

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from spectrahandler.curve_resolution import (
    SpectralDataset,
    make_realistic_dataset,
    resolve_band,
)

# Synthetic a -> b -> c data at 12.5 uM total, with known true spectra.
data, true_spectra, _ = make_realistic_dataset(jax.random.key(0))

# Suppose species "a" was measured pure: a 50 uM scan with 0.002 AU noise, divided
# by 50, gives its spectrum per uM with noise 0.002 / 50. Other species: NaN.
reference = jnp.full((3, data.n_wavelength), jnp.nan).at[0].set(true_spectra[0])
data = SpectralDataset.create(
    absorbance=data.absorbance,
    time=data.time,
    wavelength=data.wavelength,
    species=data.species,
    initial_state=data.initial_state,
    run_ids=data.run_ids,
    reference_spectra=reference,
    reference_sigma=jnp.array([0.002 / 50, jnp.nan, jnp.nan]),
)

band = resolve_band(data, jax.random.key(1))

print(f"{band.n_free} free re-mixing numbers, R-hat {band.r_hat:.3f}")
print(f"noise {float(band.sigma.min()):.4f} - {float(band.sigma.max()):.4f} AU")

# Only "a" has a reference, so only its name is pinned: the columns for "b" and "c"
# may come back swapped (see below).
lo, hi = band.concentration_lower[0, 10, 0], band.concentration_upper[0, 10, 0]
print(f"a at t = {float(data.time[0, 10]):.1f} h: {float(lo):.2f} - {float(hi):.2f} uM")
```

On this data species `a` at 3.4 h comes out between about 0.05 and 2.16 µM; the true
value is 1.58 µM. The noise comes out between 0.0015 and 0.0028 AU per wavelength (the
true value is 0.002 everywhere; each wavelength has only 30 spectra to estimate it
from). The whole call takes about 8 s on a laptop.

## Reading the result

| Field | What it is |
| --- | --- |
| `concentration_lower`, `concentration_upper` | The band for every amount, in your concentration unit |
| `spectra_lower`, `spectra_upper` | The band for every spectrum value, in absorbance per concentration unit. The run axis has size 1: all runs share one set of spectra |
| `concentration_ambiguity`, `spectra_ambiguity` | The same extremes for the posterior mean alone: the rotational ambiguity without the noise part, still including the small tolerance below zero |
| `concentration_draws`, `spectra_draws` | Draws spread evenly over all splits that fit, with noise. A *typical-solution* summary under a stated flat prior, not a calibrated interval |
| `sigma`, `sigma_draws` | The noise standard deviation at every wavelength: posterior mean and posterior draws, in absorbance units |
| `level` | The credibility level of the band, 0.95 by default |
| `n_free` | How many numbers the data leave undetermined |
| `r_hat` | Agreement between the sampling chains of step 1; close to 1 is good, and above 1.05 you get a warning |
| `background_names`, `background_amplitude` | The backgrounds that were subtracted and their amounts per run, in absorbance units |
| `baseline_tolerance`, `zero_source` | By how much zero absorbance may be off at each wavelength, and where zero came from |

Species **without** a reference are resolved only up to relabelling among themselves:
the band for "b" may be the spectrum you know as "c". Give a reference, or a run where
one of them is absent, to pin the names.

To see the band of one run, `plot_deconvolution(band, data, data.run_ids[0])` from
`spectrahandler.plot` shades it for every species; see
[Plotting results](/SpectraHandler/guides/plotting/).

## Is the number of species right?

Every split in the band reproduces the data through the same best fit, so the fit
cannot tell splits apart. It can tell you whether the number of species is right:
after removing that many components, what is left should be plain noise.

```python
from spectrahandler.curve_resolution import noise_diagnostics
from spectrahandler.plot import plot_noise

diagnostics = noise_diagnostics(data)
print(f"{diagnostics.n_above_noise} components above the noise edge")
plot_noise(diagnostics, data).savefig("noise.png")
```

On the example data this prints `3 components above the noise edge`, and the figure
looks like this:

![Noise diagnostics for three species: three singular values above the noise edge, a residual of even static, and autocorrelation dropping to zero after lag 0](../../../assets/noise-white.png)

- **Left, what the data hold.** Singular values above the dashed white-noise edge
  carry signal. Their count should equal your number of species.
- **Middle, what is left.** The residual after that many components, over time and
  wavelength. Even static is noise; stripes, steps or blobs are something the model
  is missing.
- **Right, is it noise?** The residual's autocorrelation. White noise is 1 at lag 0
  and about 0 after.

The same data with a slowly drifting offset added, which is one more component:

![Noise diagnostics with an unmodelled offset: four singular values above the edge and correlated residuals](../../../assets/noise-offset.png)

What to do with what you see:

- **More components above the edge than species:** an unmodelled species, a baseline,
  or a per-run offset. Each is one more component; the band assumes there are none.
- **Autocorrelation near 1 along wavelength:** the export was smoothed or interpolated
  along wavelength, as JASCO interval scans are. Neighbouring points then share their
  noise, which breaks the independence the band assumes, and the noise level inferred
  from the data is far too low; pass `sigma` measured along time on a stretch where
  nothing reacts.

## Baselines and backgrounds

Real spectra are rarely concentrations times spectra and nothing else. A cuvette sits a
little differently in every run, a lamp drifts, and the whole spectrum is shifted or
tilted by a few thousandths of an absorbance unit.

A baseline with a free shape cannot be separated from the species spectra: any part of
it can be moved into the spectra without changing the fit. What can be separated is a
background with a **known shape**, if you can name a wavelength range where **no species
absorbs**, a zero window. There the data hold only background and noise, so the amount of
each background is measured there, and the known shape carries it across the rest of the
spectrum.

```python
import dataclasses

from spectrahandler.curve_resolution import Background

# The same data, with everything 0.005 AU too high.
shifted = dataclasses.replace(data, absorbance=data.absorbance + 0.005)

band = resolve_band(
    shifted,
    jax.random.key(1),
    backgrounds=(Background.offset(),),
    zero_windows=((660.0, 700.0),),  # nm; no species of this example absorbs here
)

print(band.zero_source)
print(f"offset {float(band.background_amplitude[0, 0]):.4f} AU")
print(f"tolerance {float(band.baseline_tolerance[-1]):.4f} AU in the window")
print(f"tolerance {float(band.baseline_tolerance[0]):.4f} AU outside")
```

This recovers the offset as 0.0050 AU. The **baseline tolerance** is what the backgrounds
could not explain inside the window: here 0.0011 AU, which is just noise. Outside the
window it is 0.0021 AU. The band uses it in two ways: a spectrum may dip below zero by
the tolerance (per unit concentration), and the spectrum band is widened by the same
amount.

- **Backgrounds provided:** `Background.offset()` (flat) and `Background.tilt()` (a
  straight line). Any other known shape is `Background(name, shape)`, with `shape` a
  function of the wavelength grid. Each gets one amount per run.
- **The window is an assumption, and the tolerance tests it.** A species that absorbs
  weakly inside the window raises the tolerance by about its absorbance. A tolerance
  comparable to your signal means the window is not empty.
- **A tilt needs a wide window.** A line fitted over 40 nm and extended over 300 nm
  multiplies every small error. The tolerance outside the window grows accordingly
  (one plus the leverage of the fit, a worst case), and with a narrow window and
  several runs the fit can fail with "no feasible split". Use an offset alone when the
  window is narrow.
- **A blank gives a tighter tolerance.** A buffer blank has no absorber anywhere. Put it
  through `fit_backgrounds` with the same backgrounds and window; what it leaves outside
  the window is the real baseline error there. Pass that as `baseline_tolerance`, one
  value per wavelength, and it replaces the worst case.
- **Reference spectra must be baseline-free.** A blank used as a reference has its own
  baseline: correct it with `fit_backgrounds` first.
- **Without backgrounds, a baseline problem can hide in the noise.** With several runs
  whose baselines differ, the model explains the mismatch as noise and still returns a
  band, which is then wrong. An inferred `sigma` well above what your instrument
  delivers is the sign.

## Tuning

`resolve_band` exposes its knobs with defaults that worked on synthetic data:

- `level`: the credibility level of the band. Default 0.95.
- `sigma`: the noise level, one number or one per wavelength. By default one per
  wavelength is inferred from the data; pass a value measured on a blank if you have
  one.
- `noise_scale`: the scale of the prior on the inferred noise levels. The default is
  the RMS absorbance, which leaves the data to decide.
- `backgrounds`, `zero_windows`, `baseline_tolerance`, `baseline_smoothing`: see
  [Baselines and backgrounds](#baselines-and-backgrounds).
- `num_warmup`, `num_samples`, `num_chains`, `target_accept_prob`: the NUTS sampler of
  step 1. Raise the first two if you get the R-hat warning.
- `n_iter`, `n_burn`, `thin`: length of the walk over the feasible region.
- `n_als`, `n_restart`, `n_search`: the search for a first split that fits.

Current limits: every run must be fully measured, the number of species is given, and
a background has one amount per run, constant in time.

## Reading JASCO exports

`spectrahandler.jasco` reads JASCO CSV exports into plain arrays. An interval-scan
export holds a whole run in one file; a run recorded as one file per timepoint, named
like `run1_15min.csv` or `run1_2h.csv`, is read with `read_spectrum_series`, which
orders the files by the time in their names. Either way you get a `Scan` with
`wavelength_nm`, `time_min` (always in minutes) and `absorbance` shaped
`(n_time, n_wavelength)`. `SpectralDataset.create` wants a leading run axis and times
in the dataset's time unit, hours by default:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from spectrahandler.curve_resolution import SpectralDataset
from spectrahandler.jasco import read_interval_scan, read_spectrum_series

scan = read_interval_scan("my_scan.csv")
# one file per timepoint: scan = read_spectrum_series(["run1_0min.csv", "run1_15min.csv"])

data = SpectralDataset.create(
    absorbance=jnp.asarray(scan.absorbance)[None],  # (1 run, n_time, n_wavelength)
    time=jnp.asarray(scan.time_min)[None] / 60.0,  # minutes -> hours
    wavelength=jnp.asarray(scan.wavelength_nm),
    species=("a", "b", "c"),
    initial_state=jnp.array([[12.5, 0.0, 0.0]]),  # uM at the start, one row per run
    run_ids=("run 1",),
    time_unit="h",
)
```

Several runs stack along the first axis with `jnp.stack`, and need the same wavelength
grid and the same number of timepoints. JASCO interval scans are interpolated along
wavelength, which fools the automatic noise estimate; see
[Is the number of species right?](#is-the-number-of-species-right) above.
