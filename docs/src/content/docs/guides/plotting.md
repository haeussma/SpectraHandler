---
title: Plotting results
description: Draw the data, the deconvolution into species, the residuals and the rate posterior of each run, on axes you lay out yourself.
sidebar:
  order: 5
---

`spectrahandler.plot` draws one run per panel, picked by its run id. Each panel function
draws on axes you pass, or makes a new figure, and returns the axes, so you build the grid
you need: one column per shot, a slide with two panels, a paper figure. `plot_noise` draws
its own three-panel figure over all runs. No function sets a title.

| Function | Draws | Takes |
| --- | --- | --- |
| `plot_spectra` | every measured spectrum of a run, coloured by time | any dataset |
| `plot_deconvolution` | concentrations over time and species spectra | a `KineticFit` or a `FeasibleBand` |
| `plot_residuals` | the residual map and the misfit of every spectrum | a `KineticFit` |
| `plot_rate_posterior` | each run's rate and the pooled posterior of a condition | a `KineticFit` |
| `plot_noise` | singular values, residual and autocorrelation over all runs | `noise_diagnostics` |

## One column per run

```python
import jax

jax.config.update("jax_enable_x64", True)  # spectrahandler needs float64

import matplotlib.pyplot as plt

from spectrahandler.kinetics import Scheme, fit_kinetics, make_kinetic_replicates
from spectrahandler.plot import plot_deconvolution, plot_residuals, plot_spectra

scheme = Scheme(steps=[("A", "B")])
data, _, _ = make_kinetic_replicates(
    jax.random.key(0), scheme, {("A", "B"): 0.8}, initial={"A": 10.0}, between_sd_log=0.02
)
fit = fit_kinetics(data, scheme)

fig, axes = plt.subplots(5, data.n_run, figsize=(16, 16), layout="constrained")
for column, run in enumerate(data.run_ids):
    plot_spectra(data, run, ax=axes[0, column])
    plot_deconvolution(fit, data, run, axes=axes[1:3, column])
    plot_residuals(fit, data, run, axes=axes[3:5, column])
fig.savefig("runs.png")
```

- **Spectra.** Every recorded spectrum, light to dark over time.
- **Concentrations.** Dots are each measured spectrum projected onto that run's fitted
  species spectra (`data.project(fit.spectra)`, see `SpectralDataset.project`): what the
  spectra alone say about the amounts. Lines are the kinetic
  model. Dots on the lines mean the scheme describes what the spectra show.
- **Species spectra.** The fitted spectrum of every species in that run, in absorbance
  per concentration unit. Runs are not averaged: compare them across the columns.
- **Residual map.** Data minus model over wavelength and time, in units of each
  wavelength's noise level. Even static is good; stripes or blocks mean something is
  missing.
- **Misfit.** The root mean square of each spectrum's scaled residual. The noise level
  is fitted from the same residuals, so the run's average is 1 by construction: look
  for where it rises above 1, such as the first spectra after mixing.

## The rate of a condition

```python
from spectrahandler.plot import plot_rate_posterior

ax = plot_rate_posterior(fit, "synthetic", ("A", "B"))
ax.figure.savefig("rate.png")
```

Thin curves are what each run claims on its own, labelled with its run id; they are
typically too narrow. The thick curve is the posterior from the spread between the
replicates, the one behind the reported interval.

## A deconvolution without a kinetic model

`plot_deconvolution` also takes the band from
[curve resolution](/SpectraHandler/guides/curve-resolution/). It shades every split the
data allow instead of drawing one curve. A band has no single residual, so
`plot_residuals` needs a fit.

```python
from spectrahandler.curve_resolution import make_realistic_dataset, resolve_band

band_data, _, _ = make_realistic_dataset(jax.random.key(0))
band = resolve_band(band_data, jax.random.key(1))
plot_deconvolution(band, band_data, band_data.run_ids[0])
```

## Your own names, colours and fonts

```python
labels = {"A": r"E$_\mathrm{ox}$", "B": r"E$_\mathrm{red}$"}
colors = {"A": "#344A9A", "B": "#00A082"}
with plt.style.context({"font.size": 14}):
    plot_deconvolution(fit, data, "replicate1", labels=labels, colors=colors)
```

`labels` and `colors` are keyed by the species names of the dataset. Without `colors`,
species take `spectrahandler.plot.SPECIES_COLORS` in order; a sixth species needs
`colors`. Font, size and line widths come from matplotlib's rcParams.
