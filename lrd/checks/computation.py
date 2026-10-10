"""Chapter 3 checks and Appendices A and B: computation and verification."""
import numpy as np
from scipy.optimize import minimize

from . import SOLVER_TOL, Result, check
from ..core import D, subgrad, nrm, dual
from ..prob import logsumexp, softmax
from ..samplers import PH5, degenerate_sampler
from ..solver import solve, ones_perp_basis
from ..worst_case import inner_p2


@check('V3.4', 'uniqueness of the minimiser', 'cor:strictconvex')
def v3_4():
    """Six derivative-free restarts per problem must reach the same point."""
    rng = np.random.default_rng(41)
    worst = 0.0
    for _ in range(40):
        n = int(rng.choice([3, 4, 5, 6]))
        ph = rng.dirichlet(np.ones(n) * 0.7)
        r = float(rng.choice([1.5, 2.0, 3.0]))
        eps = float(rng.uniform(0.02, 0.8)) * nrm(ph - np.ones(n) / n, dual(r))
        B = ones_perp_basis(n)

        def F(y):
            z = B @ y
            return -ph @ z + logsumexp(z) + eps * D(z, r)
        sols = []
        for _ in range(6):
            y0 = rng.normal(size=n - 1) * 2.0
            res = minimize(F, y0, method='Nelder-Mead',
                           options=dict(xatol=1e-10, fatol=1e-13, maxiter=40000, maxfev=40000))
            sols.append(softmax(B @ res.x))
        sols = np.array(sols)
        worst = max(worst, float(np.max(np.abs(sols - sols.mean(0)))))
    return Result(
        passed=worst < 1e-6,
        summary=f'max spread across 6 restarts x 40 problems = {worst:.2e} (want < 1e-6)',
        metrics=dict(spread=worst))


@check('V3.5', 'D_r(log x) is not convex in x; F(zeta) is', 'obs:nonconvex')
def v3_5():
    rng = np.random.default_rng(42)

    def F(z, ph, eps, r):
        return -ph @ z + logsumexp(z) + eps * D(z, r)
    rs = [1, 1.5, 2, 3, np.inf]
    Fviol = {}
    for r in rs:
        v = 0
        for _ in range(1500):
            n = int(rng.choice([3, 4, 5, 8]))
            ph = rng.dirichlet(np.ones(n)); eps = rng.uniform(0, .5); t = rng.uniform()
            a, b = rng.normal(size=n) * 3, rng.normal(size=n) * 3
            if F(t*a + (1-t)*b, ph, eps, r) - (t*F(a, ph, eps, r) + (1-t)*F(b, ph, eps, r)) > 1e-9:
                v += 1
        Fviol[r] = v
    # one explicit counterexample to convexity of D_r(log x), valid at every r
    a = np.array([1e-4, .49995, .49995]); b = np.array([1e-4, .01, .9899]); t = .5
    gaps = {r: float(D(np.log(t*a + (1-t)*b), r) - (t*D(np.log(a), r) + (1-t)*D(np.log(b), r)))
            for r in rs}
    details = [f'F(zeta) convexity violations / 1500: { {str(k): v for k, v in Fviol.items()} }',
               'D_r(log x) counterexample gaps (positive = non-convex):']
    details += [f'   r = {str(r):>4}:  gap = {g:+.3f}' for r, g in gaps.items()]
    return Result(
        passed=sum(Fviol.values()) == 0 and min(gaps.values()) > 0,
        summary=(f'F convex (0 violations); D_r(log x) gaps '
                 f'{min(gaps.values()):+.2f} to {max(gaps.values()):+.2f}'),
        details=details,
        metrics=dict(F_violations=sum(Fviol.values()), gaps=gaps))


@check('VA.1', 'certified KKT ground truth at r* = 2', 'app:kkt')
def va_1():
    """inner_p2 against an independent SLSQP solve; 40% of instances have a zero in ph."""
    rng = np.random.default_rng(43)
    ncert = below = 0; mx = 0.0
    for _ in range(400):
        n = int(rng.choice([3, 4, 5, 8])); ph = rng.dirichlet(np.ones(n))
        if rng.random() < .4:
            ph[rng.integers(n)] = 0.; ph /= ph.sum()
        f = rng.normal(size=n) * 2; eps = rng.uniform(.01, .6)
        val, q, c = inner_p2(ph, f, eps)
        if not c['ok']:
            ncert += 1; continue
        rs = minimize(lambda z: -(z @ f), ph.copy(), method='SLSQP',
                      constraints=[dict(type='eq', fun=lambda z: z.sum() - 1),
                                   dict(type='ineq', fun=lambda z: eps - np.linalg.norm(z - ph))],
                      bounds=[(0, 1)] * n, options=dict(ftol=1e-14, maxiter=500))
        if rs.success:
            mx = max(mx, val - (-rs.fun))
            if val < -rs.fun - 1e-7:
                below += 1
    return Result(
        passed=ncert == 0 and below == 0,
        summary=f'certificate failures {ncert}/400, below-SLSQP {below}, max(KKT - SLSQP) {mx:.2e}',
        metrics=dict(cert_failures=ncert, below=below, max_excess=mx))


@check('VB.1', 'the subgradient identities, including at the boundary', 'cor:subgrad')
def vb_1():
    """sum(g) = 0, ||g||_{r*} <= 1, <g,v> = D_r(v) on constant, near-constant, tied v."""
    rng = np.random.default_rng(44)
    bad = {}; worst = 0.0
    for r in [1, 1.05, 1.2, 1.5, 2, 3, 6, 20, np.inf]:
        p = dual(r); b = 0
        for _ in range(3000):
            v = degenerate_sampler(rng)
            g = subgrad(v, r); Dv = D(v, r)
            sc = max(1., abs(Dv), float(np.max(np.abs(v))))
            worst = max(worst, abs(g.sum()))
            if (not np.all(np.isfinite(g))) or abs(g.sum()) > 1e-9 \
               or nrm(g, p) > 1 + 1e-9 or abs(g @ v - Dv) > 1e-8 * sc:
                b += 1
        if b:
            bad[r] = b
    return Result(
        passed=not bad,
        summary=f'failures {bad or "none"} / 3000 per r;  max|sum g| = {worst:.1e}',
        metrics=dict(failures=bad, max_sum=worst))


@check('V3.6', 'the saturation radius and the distance to the nominal', 'prop:saturation, prop:distance')
def v3_6():
    """Below eps_sat, ||x* - ph||_{r*} = eps; at and above it, the displacement freezes."""
    u = np.ones(5) / 5
    worst_below = worst_above = 0.0
    for r in [1, 1.5, 2, 3, np.inf]:
        p = dual(r); sat = nrm(PH5 - u, p)
        for fr in [.2, .5, .8, .95]:
            x = solve(PH5, fr * sat, r)
            worst_below = max(worst_below, abs(nrm(x - PH5, p) - fr * sat))
        for fr in [1.01, 1.5, 3.0]:
            x = solve(PH5, fr * sat, r, sat_slack=-2.0)   # push the closed-form shortcut out to
            worst_above = max(worst_above, abs(nrm(x - PH5, p) - sat))   # 3 eps_sat so the optimiser works
    return Result(
        passed=worst_below < SOLVER_TOL and worst_above < SOLVER_TOL,
        summary=f'distance error below saturation {worst_below:.2e}; freeze error above {worst_above:.2e}',
        metrics=dict(distance_error=worst_below, freeze_error=worst_above))
