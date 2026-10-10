"""Fixed test inputs and the degeneracy-aware sampler used by the checks and tests."""
import numpy as np

# A fixed 5-class composition used across several checks.
PH5 = np.array([.45, .24, .16, .09, .06])

# Fixed nominals for the breakpoint checks (mixed n = 4, 5).
NOMS = [np.array([.45, .24, .16, .09, .06]), np.array([.55, .25, .13, .07]),
        np.array([.30, .29, .23, .18]), np.array([.28, .27, .20, .15, .10]),
        np.array([.40, .30, .20, .10])]


def degenerate_row(rng, n):
    """One vector of width n from a sampler that REACHES the boundary.

    Gaussian draws are never constant, and constant v is the one input where
    sum-zero actually fails, so the sampler mixes eight input classes: generic,
    constant, near-constant, integer-valued (ties), a tied maximum, rounded,
    large-scale, and log-Dirichlet (sparse compositions).
    """
    return _row_of_class(rng, rng.integers(0, 8), n)


def degenerate_sampler(rng):
    """degenerate_row at a random width n in 3..8."""
    k = rng.integers(0, 8)
    n = int(rng.choice([3, 4, 5, 6, 7, 8]))
    return _row_of_class(rng, k, n)


def _row_of_class(rng, k, n):
    """Draw one row of input class k (see degenerate_row).  RNG order is fixed:
    the checks' quoted numbers depend on it."""
    if k == 0: return rng.normal(size=n)
    if k == 1: return np.ones(n) * rng.normal()
    if k == 2: return np.ones(n) * rng.normal() + 1e-14 * rng.normal(size=n)
    if k == 3: return rng.integers(0, 3, n).astype(float)
    if k == 4:
        v = rng.normal(size=n); v[0] = v[1] = v.max() + 1; return v
    if k == 5: return np.round(rng.normal(size=n), 1)
    if k == 6: return rng.normal(size=n) * 1e6
    return np.log(rng.dirichlet(np.ones(n) * .3))
