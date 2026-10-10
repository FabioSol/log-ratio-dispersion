"""Small probability helpers shared by the solver, the sweep, and the checks."""
import numpy as np


def softmax(z):
    """Softmax along the last axis (works on a vector or on rows)."""
    z = np.asarray(z, float)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def logsumexp(z):
    """log(sum(exp(z))) of a vector, stabilised."""
    z = np.asarray(z, float)
    return float(np.log(np.exp(z - z.max()).sum()) + z.max())


def kl(p, x):
    """KL(p || x): the redundancy of coding p with x.  Infinite where x_i = 0 < p_i."""
    p = np.asarray(p, float)
    x = np.asarray(x, float)
    m = p > 0
    if np.any(x[m] <= 0):
        return np.inf
    return float(np.sum(p[m] * np.log(p[m] / x[m])))


def entropy(x):
    """Shannon entropy in nats, with 0 log 0 = 0."""
    x = np.clip(np.asarray(x, float), 1e-300, 1.0)
    return float(-(x * np.log(x)).sum())
