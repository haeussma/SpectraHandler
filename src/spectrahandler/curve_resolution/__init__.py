"""Curve resolution: pure spectra and concentration profiles, with the band the data allow."""

from spectrahandler.curve_resolution.background import Background, BackgroundFit, fit_backgrounds
from spectrahandler.curve_resolution.band import CredibleBand, resolve_band
from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics, noise_diagnostics
from spectrahandler.curve_resolution.noise import NoiseCorrelation, measure_noise_correlation
from spectrahandler.curve_resolution.synthetic import (
    make_correlated_noise,
    make_easy_dataset,
    make_realistic_dataset,
)

__all__ = [
    "Background",
    "BackgroundFit",
    "CredibleBand",
    "NoiseCorrelation",
    "NoiseDiagnostics",
    "SpectralDataset",
    "fit_backgrounds",
    "make_correlated_noise",
    "make_easy_dataset",
    "make_realistic_dataset",
    "measure_noise_correlation",
    "noise_diagnostics",
    "resolve_band",
]
