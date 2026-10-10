"""Beyond norm balls: the keystone construction for a general sum-zero set G.

The ambiguity set is (p_hat + eps*G) intersected with the simplex, with G a compact
convex subset of 1^perp containing 0 in its relative interior.  The penalty is the
support function sigma_G(f) = max_{g in G} <g, f>; norm balls are the case
G = {||g||_{r*} <= 1, 1^T g = 0}, where sigma_G = D_r.

This module gives two ASYMMETRIC landmark sets with closed-form support functions,
subgradients and gauges, and an exact solver, to verify (check V3.9) that the path
relation, the saturation gauge and the distance-equals-radius identity transfer:

  MbLS  (margin-based label smoothing, m = 0):  G = u - Delta = conv{u - e_i}
        sigma_G(v) = mean(v) - min(v),  dsigma_G(v) = u - e_argmin(v),
        gauge_G(w) = n * max(w),   so eps_sat = gauge_G(u - p_hat) = 1 - n*min(p_hat).

  contamination (imprecise Dirichlet):         G = Delta - p_hat = conv{e_i - p_hat}
        sigma_G(v) = max(v) - <p_hat, v>,  dsigma_G(v) = e_argmax(v) - p_hat,
        gauge_G(w) = max_i (-w_i / p_hat_i)  over i with w_i < 0.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def mbls(n, ph=None):
    u = np.ones(n) / n
    return dict(
        name='MbLS (G = u - Delta)',
        sigma=lambda v: float(v.mean() - v.min()),
        subgrad=lambda v: u - np.eye(n)[int(np.argmin(v))],
        gauge=lambda w: float(n * np.max(w)),
        # penalty as a function of z (x = softmax(z)): sigma_G(-z) = max(z) - mean(z)
        pen=lambda z: float(np.max(z) - z.mean()),
        pen_epigraph='max',
    )


def contamination(n, ph):
    ph = np.asarray(ph, float)
    return dict(
        name='contamination (G = Delta - p_hat)',
        sigma=lambda v: float(np.max(v) - ph @ v),
        subgrad=lambda v: np.eye(n)[int(np.argmax(v))] - ph,
        gauge=lambda w: max((-w[i] / ph[i]) for i in range(n) if w[i] < -1e-15),
        # sigma_G(-z) = max(-z) - <ph, -z> = -min(z) + <ph, z>
        pen=lambda z: float(-np.min(z) + ph @ z),
        pen_epigraph='min',
    )


def solve(ph, eps, G, ph_pen=None):
    """Exact minimiser of  <p_hat, -log x> + eps * sigma_G(-log x)  over the simplex,
    via the epigraph form (smooth lse objective + linear constraints), SLSQP.

    Returns x* = softmax(z*).
    """
    ph = np.asarray(ph, float); n = len(ph)
    kind = G['pen_epigraph']
    # variables w = [z (n), s (1)]; constraint sum z = 0; s is the epigraph of max/min(z)
    def unpack(w): return w[:n], w[n]

    if kind == 'max':   # pen = s - mean(z),  s >= z_i
        def obj(w):
            z, s = unpack(w); z = z - z.mean()
            return -ph @ z + logsumexp(z) + eps * (s - z.mean())
        cons = [dict(type='eq', fun=lambda w: w[:n].sum())]
        cons += [dict(type='ineq', fun=lambda w, i=i: w[n] - w[i]) for i in range(n)]
    else:               # pen = -s + <ph,z>,  s <= z_i
        def obj(w):
            z, s = unpack(w); z = z - z.mean()
            return -ph @ z + logsumexp(z) + eps * (-s + ph @ z)
        cons = [dict(type='eq', fun=lambda w: w[:n].sum())]
        cons += [dict(type='ineq', fun=lambda w, i=i: w[i] - w[n]) for i in range(n)]

    best = None
    for seed in range(6):
        rng = np.random.default_rng(seed)
        z0 = rng.normal(size=n) * 0.3; z0 -= z0.mean()
        s0 = z0.max() if kind == 'max' else z0.min()
        res = minimize(obj, np.concatenate([z0, [s0]]), method='SLSQP',
                       constraints=cons, options=dict(ftol=1e-14, maxiter=2000))
        if res.success and (best is None or res.fun < best.fun):
            best = res
    z = best.x[:n]; z = z - z.mean()
    return np.exp(z - logsumexp(z))
