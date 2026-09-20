"""Bayesian curve resolution: spectra and concentration profiles, with posteriors."""

from spectrahandler.curve_resolution.dataset import SpectralDataset
from spectrahandler.curve_resolution.synthetic import make_easy_dataset

__all__ = ["SpectralDataset", "make_easy_dataset"]
