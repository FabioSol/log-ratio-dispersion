"""Reference solver for the penalised problem, using convexity in the softmax
coordinate (lem:paramB, cor:strictconvex):

    min_zeta  -<ph, zeta> + lse(zeta) + eps * D_r(zeta + log o),     x* = softmax(zeta*).

F is constant along span{1}, so we optimise on an orthonormal basis of 1^perp, where
F is strictly convex and coercive: the minimiser is unique and stationarity suffices.

  1 < r < inf : differentiable away from span{1}; analytic gradient, L-BFGS-B
                (dD_r/dv = subgrad(v, r) by the envelope theorem).
  r = inf     : D_inf(v) = (max v - min v)/2  ->  smooth objective + linear
                constraints v_i <= a, v_i >= b; SLSQP.
  r = 1       : D_1(v) = min_c sum|v_i - c|  ->  epigraph variables t_i; SLSQP.

At and past the saturation radius eps_sat = ||ph - u||_{r*} the solution is u exactly
(prop:saturation), returned in closed form.  A tilt o (the tilted-log-loss extension of the
future work, sec:concl-future) disables that shortcut, because the closed form assumes o = 1.
"""
import numpy as np
from scipy.optimize import minimize

from .core import D, subgrad, dual, nrm, is_degenerate
from .prob import softmax, logsumexp


def ones_perp_basis(n):
    """Orthonormal basis of 1^perp, shape (n, n-1).

    The columns of I - 11'/n sum to zero and any n-1 of them are independent, so
    the first n-1 columns of its (unpivoted) QR span 1^perp exactly.
    """
    Q, _ = np.linalg.qr(np.eye(n) - np.ones((n, n)) / n)
    B = Q[:, :n - 1]
    assert np.max(np.abs(B.sum(0))) < 1e-10, "basis not in 1^perp"
    return B


def saturation_radius(ph, r):
    """eps_sat = ||ph - u||_{r*} (prop:saturation)."""
    ph = np.asarray(ph, float)
    return float(nrm(ph - np.ones(len(ph)) / len(ph), dual(r)))


def solve(ph, eps, r, tilt=None, sat_slack=0.0, full=False):
    """The penalised solution x*(eps) at index r.  Returns x*, or (x*, info) if full.

    sat_slack moves the closed-form saturation shortcut to eps >= (1 - sat_slack) *
    eps_sat, so the solver can be probed arbitrarily close to saturation from below.

    info['converged'] and info['message'] report the optimiser.  info['b2_residual']
    is max|x* - (ph + eps*g)| for the canonical g (the path relation,
    prop:pathrelation).  A large value is NOT necessarily failure: past a breakpoint
    at r in {1, inf} the true g is a hull element and the canonical vertex is the
    wrong one.  info['min_gap'] reports the tie gap alongside, so the two are read
    together.
    """
    ph = np.asarray(ph, float)
    n = len(ph)
    lo = np.zeros(n) if tilt is None else np.log(np.asarray(tilt, float))
    u = np.ones(n) / n

    if tilt is None and eps >= saturation_radius(ph, r) * (1.0 - sat_slack):
        x = u.copy()
        return (x, dict(case='saturated', converged=True, b2_residual=0.0,
                        min_gap=0.0, message='closed form (prop:saturation)')) if full else x

    B = ones_perp_basis(n)
    lp = np.log(np.maximum(ph, 1e-12))
    z0 = B.T @ (lp - lp.mean())

    if 1 < r < np.inf:
        def fg(y):
            z = B @ y
            v = z + lo
            f = -ph @ z + logsumexp(z) + eps * D(v, r)
            g = np.zeros(n) if is_degenerate(v) else subgrad(v, r)
            return f, B.T @ (-ph + softmax(z) + eps * g)

        res = minimize(fg, z0, jac=True, method='L-BFGS-B',
                       options=dict(ftol=1e-16, gtol=1e-14, maxiter=20000, maxfun=20000))
        z = B @ res.x

    elif np.isinf(r):
        def fg(w):
            y, a, b = w[:n - 1], w[n - 1], w[n]
            z = B @ y
            f = -ph @ z + logsumexp(z) + eps * 0.5 * (a - b)
            return f, np.concatenate([B.T @ (-ph + softmax(z)), [eps * .5, -eps * .5]])
        v0 = B @ z0 + lo
        w0 = np.concatenate([z0, [v0.max(), v0.min()]])
        A = np.zeros((2 * n, n + 1))
        A[:n, :n - 1], A[:n, n - 1] = B, -1.0            # v_i - a <= 0
        A[n:, :n - 1], A[n:, n] = -B, 1.0                # b - v_i <= 0
        rhs = np.concatenate([-lo, lo])
        res = minimize(fg, w0, jac=True, method='SLSQP', constraints=[
            dict(type='ineq', fun=lambda w: rhs - A @ w, jac=lambda w: -A)],
            options=dict(ftol=1e-16, maxiter=4000))
        z = B @ res.x[:n - 1]

    else:  # r == 1
        def fg(w):
            y, c, t = w[:n - 1], w[n - 1], w[n:]
            z = B @ y
            f = -ph @ z + logsumexp(z) + eps * t.sum()
            return f, np.concatenate([B.T @ (-ph + softmax(z)), [0.0], eps * np.ones(n)])
        v0 = B @ z0 + lo
        c0 = float(np.median(v0))
        w0 = np.concatenate([z0, [c0], np.abs(v0 - c0) + 1e-6])
        A = np.zeros((2 * n, 2 * n))
        A[:n, :n - 1], A[:n, n - 1], A[:n, n:] = B, -1.0, -np.eye(n)
        A[n:, :n - 1], A[n:, n - 1], A[n:, n:] = -B, 1.0, -np.eye(n)
        rhs = np.concatenate([-lo, lo])
        res = minimize(fg, w0, jac=True, method='SLSQP', constraints=[
            dict(type='ineq', fun=lambda w: rhs - A @ w, jac=lambda w: -A)],
            options=dict(ftol=1e-16, maxiter=4000))
        z = B @ res.x[:n - 1]

    x = softmax(z)
    if not full:
        return x

    v = -np.log(x * (np.ones(n) if tilt is None else np.asarray(tilt, float)))
    g = np.zeros(n) if is_degenerate(v) else subgrad(v, r)
    sv = np.sort(v)
    info = dict(case='interior',
                converged=bool(res.success or 'CONVERGENCE' in str(res.message)),
                message=str(res.message),
                nit=int(getattr(res, 'nit', -1)),
                b2_residual=float(np.max(np.abs(x - (ph + eps * g)))),
                distance_error=float(abs(nrm(x - ph, dual(r)) - eps)),
                min_gap=float(np.min(sv[1:] - sv[:-1])))
    return x, info
