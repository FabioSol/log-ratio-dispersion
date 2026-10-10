"""Exact ground truth for the inner (adversary's) problem

    W(f) = max_q <q, f>   s.t.  q in Delta^n,  ||q - ph||_p <= eps,

against which the closed-form bound <ph, f> + eps * D_{p*}(f) is checked.

    p in {1, inf} : a linear programme (worst_case_LP).
    p = 2         : closed form on the active set, with a KKT certificate (inner_p2;
                    derivation in app:kkt).

worst_case() dispatches on p.  A failed solve RAISES LPFailure: returning nan would
make `abs(nan - bound) < tol` False and silently record a failed solve as a genuine
non-tight case.
"""
import numpy as np
from scipy.optimize import linprog


class LPFailure(RuntimeError):
    """An exact ground-truth solve did not succeed. Never silently nan."""


def worst_case(ph, f, eps, p):
    """W(f) exactly, for p in {1, 2, inf}.  Raises LPFailure on a failed solve."""
    if p == 2:
        val, _, cert = inner_p2(ph, f, eps)
        if not cert['ok']:
            raise LPFailure(f"inner_p2 certificate failed: {cert}")
        return val
    return worst_case_LP(ph, f, eps, p)


# ---------------------------------------------------------------------------
# p in {1, inf}: linear programme
# ---------------------------------------------------------------------------

def worst_case_LP(ph, f, eps, p):
    """max <q,f> over {q in Delta : ||q - ph||_p <= eps}, exactly, for p in {1, inf}."""
    ph = np.asarray(ph, float)
    f = np.asarray(f, float)
    n = len(ph)
    if p == 1:
        # variables (q, s) with |q - ph| <= s, sum(s) <= eps
        c = np.concatenate([-f, np.zeros(n)])
        A = np.vstack([np.hstack([np.eye(n), -np.eye(n)]),
                       np.hstack([-np.eye(n), -np.eye(n)]),
                       np.concatenate([np.zeros(n), np.ones(n)])[None, :]])
        b = np.concatenate([ph, -ph, [eps]])
        Aeq = np.concatenate([np.ones(n), np.zeros(n)])[None, :]
        res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[1.0],
                      bounds=[(0, None)] * (2 * n), method='highs')
    elif np.isinf(p):
        res = linprog(-f, A_eq=np.ones((1, n)), b_eq=[1.0],
                      bounds=[(max(0.0, ph[i] - eps), ph[i] + eps) for i in range(n)],
                      method='highs')
    else:
        raise ValueError("exact LP only at p in {1, inf}; use inner_p2 for p = 2")
    if not res.success:
        raise LPFailure(f"linprog failed: {res.message}")
    return float(-res.fun)


# ---------------------------------------------------------------------------
# p = 2: closed form given the active set, certified by KKT
#
# Write delta = q - ph.  KKT with multipliers mu (sum), lambda >= 0 (ball),
# nu >= 0 (nonnegativity), t := 2*lambda:
#
#     f = mu*1 + t*delta - nu,     nu_i (ph_i + delta_i) = 0.
#
# Let A = {i : q_i = 0} be the active set and m = |A^c|.  Off A, delta_i = (f_i - mu)/t.
# Imposing sum(delta) = 0 and ||delta|| = eps closes the system in one step:
#
#     P  = sum_{i in A} ph_i,   P2 = sum_{i in A} ph_i^2,   d_i = f_i - mean_{A^c}(f)
#     t  = ||d|| / sqrt(eps^2 - P2 - P^2/m)
#     q_i = ph_i + d_i/t + P/m   (i not in A),     q_i = 0   (i in A)
#     mu = mean_{A^c}(f) - t*P/m,     nu_i = mu - f_i - t*ph_i   (i in A)
#
# A is found from the projection view: for t > 0, q(t) = Proj_Delta(ph + f/t) has
# displacement monotone in t, so bisection on t locates A (a discrete object, so the
# bisection only has to be roughly right) and the closed form gives the answer to
# machine precision.  KKT plus convexity certifies the result: a proof, not a residual.
# ---------------------------------------------------------------------------

def proj_simplex(y):
    """Euclidean projection onto the probability simplex (Michelot / Duchi et al.)."""
    n = len(y)
    s = np.sort(y)[::-1]
    css = np.cumsum(s) - 1.0
    rho = np.nonzero(s - css / np.arange(1, n + 1) > 0)[0][-1]
    return np.maximum(y - css[rho] / (rho + 1.0), 0.0)


def _from_active_set(ph, f, eps, A):
    """Closed-form KKT point for a given active set. Returns (q, t, mu) or None."""
    n = len(ph)
    Ac = np.setdiff1d(np.arange(n), A)
    m = len(Ac)
    if m == 0:
        return None
    P, P2 = ph[A].sum(), (ph[A] ** 2).sum()
    disc = eps ** 2 - P2 - P ** 2 / m
    if disc <= 0:
        return None
    fbar = f[Ac].mean()
    d = f[Ac] - fbar
    nd = np.linalg.norm(d)
    if nd < 1e-300:                      # f constant off A: ball inactive in that face
        return None
    t = nd / np.sqrt(disc)
    q = np.zeros(n)
    q[Ac] = ph[Ac] + d / t + P / m
    return q, t, fbar - t * P / m


def inner_p2(ph, f, eps, tol=1e-9):
    """Exact worst case at p = 2 with a KKT certificate.

    Returns (value, q, cert) where cert['ok'] is True only if every KKT condition and
    every feasibility condition holds to `tol`.
    """
    ph = np.asarray(ph, float)
    f = np.asarray(f, float)
    n = len(ph)

    # ball swallows the best vertex -> maximiser is that vertex
    k = int(np.argmax(f))
    ek = np.zeros(n); ek[k] = 1.0
    if np.linalg.norm(ek - ph) <= eps:
        return float(f[k]), ek, dict(ok=True, case='vertex', A=[])

    # unconstrained (lem:keystone) solution: tight iff it stays in the simplex
    fc = f - f.mean()
    nfc = np.linalg.norm(fc)
    if nfc < 1e-300:
        return float(ph @ f), ph.copy(), dict(ok=True, case='constant f', A=[])
    q0 = ph + eps * fc / nfc
    if q0.min() >= -1e-14:
        q0 = np.maximum(q0, 0.0)
        return float(q0 @ f), q0, dict(ok=True, case='interior (bound tight)', A=[])

    # locate the active set by bisection on t (A is piecewise constant in t)
    lo, hi = 1e-12, 1.0
    while np.linalg.norm(proj_simplex(ph + f / hi) - ph) > eps:
        hi *= 2.0
        if hi > 1e14:
            break
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if np.linalg.norm(proj_simplex(ph + f / mid) - ph) > eps:
            lo = mid
        else:
            hi = mid
    A = np.flatnonzero(proj_simplex(ph + f / hi) <= 1e-13)

    # closed form on that active set, then certify; try neighbours if it fails
    order = np.argsort(f + hi * ph)          # KKT orders by f_i + t*ph_i, not by f_i
    cands = [A] + [order[:j] for j in range(n)]
    best = None
    for Ac_try in cands:
        out = _from_active_set(ph, f, eps, np.asarray(Ac_try, int))
        if out is None:
            continue
        q, t, mu = out
        Aset = np.asarray(Ac_try, int)
        nu = mu - f[Aset] - t * ph[Aset] if len(Aset) else np.zeros(0)
        cert = dict(
            primal_nonneg=float(q.min()),
            primal_sum=float(abs(q.sum() - 1.0)),
            ball_active=float(abs(np.linalg.norm(q - ph) - eps)),
            dual_nu=float(nu.min()) if len(nu) else 0.0,
            dual_t=float(t),
            A=sorted(int(i) for i in Aset),
        )
        cert['ok'] = (cert['primal_nonneg'] >= -tol and cert['primal_sum'] <= tol
                      and cert['ball_active'] <= tol and cert['dual_nu'] >= -tol
                      and cert['dual_t'] > 0)
        cert['case'] = 'boundary'
        if cert['ok']:
            return float(q @ f), q, cert
        if best is None or (q @ f) > best[0]:
            best = (float(q @ f), q, cert)
    return best if best else (np.nan, None, dict(ok=False, case='no KKT point'))
