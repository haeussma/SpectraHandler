"""Curve resolution: pure spectra and concentration profiles, with the band the data allow."""

from spectrahandler.curve_resolution.band import FeasibleBand, resolve_band
from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.diagnostics import NoiseDiagnostics, noise_diagnostics
from spectrahandler.curve_resolution.synthetic import make_easy_dataset, make_realistic_dataset

__all__ = [
    "FeasibleBand",
    "NoiseDiagnostics",
    "SpectralDataset",
    "make_easy_dataset",
    "make_realistic_dataset",
    "noise_diagnostics",
    "resolve_band",
]
