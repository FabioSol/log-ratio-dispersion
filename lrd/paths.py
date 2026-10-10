"""Filesystem locations, resolved from this file so nothing depends on the cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the directory that contains lrd/
FIGURES = ROOT / 'figures'
NOISE_SWEEP_JSON = ROOT / 'noise_sweep.json'


def figure(name):
    """Path of a figure, creating the figures directory if needed."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name
