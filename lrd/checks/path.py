"""Chapter 4 checks: the solution path."""
import numpy as np

from . import EXACT_TOL, SOLVER_TOL, Result, check
from ..core import D, nrm, dual
from ..samplers import PH5, NOMS
from ..solver import solve


def n3_path(ph, eps):
    """prop:n3: the exact n = 3 l1-ball path (r = inf), both regimes, for sorted ph.

    Case A (p2 < 1/3, bottom gap tighter): the extremes move until x2 = x3 fuse, then
    the pair rides the median to u.  Case B (p2 >= 1/3): until x1 = x2 fuse.
    """
    p1, p2, p3 = ph
    if p2 < 1 / 3:
        if eps <= 2 * (p2 - p3):
            return np.array([p1 - eps / 2, p2, p3 + eps / 2])
        if eps <= 2 * (p1 - 1 / 3):
            t = (1 - p1 + eps / 2) / 2
            return np.array([p1 - eps / 2, t, t])
    else:
        if eps <= 2 * (p1 - p2):
            return np.array([p1 - eps / 2, p2, p3 + eps / 2])
        if eps <= 2 * (1 / 3 - p3):
            s = (1 - p3 - eps / 2) / 2
            return np.array([s, s, p3 + eps / 2])
    return np.ones(3) / 3


def n3_breakpoint(ph):
    """(eps_1, fused pair, apex index) of the n = 3 path, for sorted ph."""
    p1, p2, p3 = ph
    if p2 < 1 / 3:
        return 2 * (p2 - p3), (1, 2), 0
    return 2 * (p1 - p2), (0, 1), 2


@check('V4.1', 'the barrier bound', 'lem:barrier')
def v4_1():
    """D_r(log x) >= D_inf(log x), and D_inf ~ |log x_min|/2 as a coordinate vanishes.

    The rate is checked exactly: for x = (t, 1, 1, 1)/(3 + t),
    D_inf(log x) = log(1/t)/2 and |log x_min| = log((3 + t)/t), so the ratio is
    log(1/t)/log((3 + t)/t), which increases to 1 as t -> 0.
    """
    rng = np.random.default_rng(52); dom = 0
    for _ in range(2000):
        n = int(rng.choice([3, 5, 8])); x = rng.dirichlet(np.ones(n) * .2)
        lx = np.log(x); di = D(lx, np.inf)
        dom += any(D(lx, r) < di - 1e-12 for r in [1, 1.5, 2, 3, 6])
    ts = [1e-2, 1e-8, 1e-16, 1e-40]
    ratios, err = [], 0.0
    for t in ts:
        x = np.array([t, 1., 1., 1.]); x /= x.sum()
        ratios.append(float(D(np.log(x), np.inf) / (0.5 * abs(np.log(x.min())))))
        err = max(err, abs(ratios[-1] - np.log(1 / t) / np.log((3 + t) / t)))
    details = [f'cases with D_r < D_inf (want 0): {dom}']
    details += [f't = {t:.0e}:  D_inf / (0.5|log x_min|) = {q:.3f}' for t, q in zip(ts, ratios)]
    details.append(f'max |ratio - closed form| = {err:.1e}')
    return Result(
        passed=dom == 0 and err < EXACT_TOL and bool(np.all(np.diff(ratios) > 0)),
        summary=f'ratio {ratios[0]:.3f} -> {ratios[-1]:.3f} (closed form to {err:.0e}); D_r < D_inf in {dom} cases',
        details=details,
        metrics=dict(dominance_failures=dom, ratios=ratios, closed_form_error=err))


@check('V4.4', 'the first breakpoint in closed form', 'prop:breakpoint, prop:breakpoint1')
def v4_4():
    worst = 0.0
    details = []
    all_ok = True
    for ph in NOMS:
        n = len(ph); u = np.ones(n) / n
        e1 = 2 * min(ph[0] - ph[1], ph[-2] - ph[-1]); sat_inf = nrm(ph - u, 1)
        for fr in [.1, .5, .9, .99]:
            eps = fr * e1
            pred = ph.copy(); pred[0] -= eps / 2; pred[-1] += eps / 2
            worst = max(worst, float(np.max(np.abs(solve(ph, eps, np.inf) - pred))))
        m = n // 2
        if n % 2:
            f1 = min(ph[m-1] - ph[m], ph[m] - ph[m+1])
            pr = lambda e: np.concatenate([ph[:m] - e, [ph[m]], ph[m+1:] + e])
        else:
            f1 = (ph[m-1] - ph[m]) / 2
            pr = lambda e: np.concatenate([ph[:m] - e, ph[m:] + e])
        for fr in [.1, .5, .9, .99]:
            worst = max(worst, float(np.max(np.abs(solve(ph, fr * f1, 1) - pr(fr * f1)))))
        ok = (e1 <= sat_inf + 1e-12) and (f1 <= nrm(ph - u, np.inf) + 1e-12)
        all_ok &= ok
        details.append(f'{np.round(ph, 2)}  eps1(r=inf)={e1:.3f}  eps1(r=1)={f1:.3f}  eps1<=sat: {ok}')
    return Result(
        passed=worst < SOLVER_TOL and all_ok,
        summary=f'max|solver - closed form| = {worst:.2e} (want < 1e-7); eps1 <= eps_sat: {all_ok}',
        details=details,
        metrics=dict(worst=worst))


@check('V4.5', 'the n = 3 l1 path, both regimes', 'prop:n3')
def v4_5():
    """Closed form vs solver at fractions of eps_sat (including past it), and the
    case-dependent eps_sat expression vs ||ph - u||_1 (prop:saturation)."""
    noms = [np.array([.7, .2, .1]), np.array([.5, .3, .2]),
            np.array([.40, .35, .25]), np.array([.45, .34, .21]),
            np.array([.34, .33, .33]), np.array([.5, .34, .16])]
    worst = sat_mismatch = 0.0
    details = []
    for ph in noms:
        sat_b1 = nrm(ph - np.ones(3) / 3, 1)
        sat_case = 2 * (ph[0] - 1 / 3) if ph[1] < 1 / 3 else 2 * (1 / 3 - ph[2])
        sat_mismatch = max(sat_mismatch, abs(sat_b1 - sat_case))
        w = 0.0
        for fr in [.05, .3, .5, .7, .9, .99, 1.05, 2.0]:
            eps = fr * sat_b1
            w = max(w, float(np.max(np.abs(solve(ph, eps, np.inf, sat_slack=-2.0) - n3_path(ph, eps)))))
        worst = max(worst, w)
        details.append(f"{np.round(ph, 3)}  case {'A' if ph[1] < 1/3 else 'B'}  "
                       f"sat={sat_b1:.4f}  max|diff|={w:.1e}")
    return Result(
        passed=worst < SOLVER_TOL and sat_mismatch < EXACT_TOL,
        summary=(f'max|solver - closed form| = {worst:.2e}; '
                 f'max|sat_B1 - sat_case| = {sat_mismatch:.1e}'),
        details=details,
        metrics=dict(worst=worst, sat_mismatch=sat_mismatch))


@check('V4.8', 'gap decay, no interior breakpoint', 'thm:trichotomy')
def v4_8():
    """At interior r the minimum coordinate gap stays positive up to saturation and
    decays linearly in (1 - eps/eps_sat) as eps -> eps_sat: log-log slope 1 over the
    last two decades, to 0.05.  The rate is asymptotic, and members near r = 1 (where
    the path does break) reach it later: at r = 1.2 the decade before is still ~x19.
    That decade is reported, not judged, and so is a dense grid with r = 1.05 added, where
    two coordinates come within machine precision of each other."""
    u = np.ones(5) / 5
    fracs = [.5, .9, .99, .999, .9999]
    decay = {}
    details = []
    slopes = {}
    tail = np.log(1 - np.array(fracs[-3:]))            # the last two decades
    for r in [1.2, 1.5, 2, 3, 6]:
        sat = nrm(PH5 - u, dual(r)); gaps = []
        for fr in fracs:
            x = solve(PH5, fr * sat, r)
            gaps.append(float(np.min(-np.diff(np.sort(x)[::-1]))))
        decay[r] = gaps
        slopes[r] = (float(np.polyfit(tail, np.log(gaps[-3:]), 1)[0])
                     if min(gaps) > 0 else np.nan)
        details.append(f'r={r}: ' + ' '.join(f'{g:.1e}' for g in gaps)
                       + f'   (log-log slope, last 2 decades: {slopes[r]:.3f})')
    positive = all(min(g) > 0 for g in decay.values())
    linear = all(abs(s - 1) < 0.05 for s in slopes.values())
    # Reported, not judged: a dense grid (20 points per decade) with r = 1.05 added, and the
    # gap divided by ||x* - u||_{r*}, which separates fusion from the collapse to u.
    dense = 1 - np.geomspace(0.5, 1e-4, 75)
    raw, rel = {}, {}
    for r in [1.05, 1.2, 1.5, 2, 3, 6]:
        sat = nrm(PH5 - u, dual(r)); raw[r], rel[r] = [], []
        for fr in dense:
            x = solve(PH5, fr * sat, r)
            g = max(0.0, float(np.min(-np.diff(np.sort(x)[::-1]))))
            raw[r].append(g); rel[r].append(g / nrm(x - u, dual(r)))
        details.append(f'dense r={r}: min gap {min(raw[r]):.1e}; gap / ||x - u|| from '
                       f'{rel[r][0]:.1e} to {rel[r][-1]:.1e} (range {min(rel[r]):.1e}-{max(rel[r]):.1e})')
    return Result(
        passed=positive and linear,
        summary=(f'gaps positive: {positive}; log-log slope {min(slopes.values()):.3f}'
                 f'-{max(slopes.values()):.3f} (linear decay = 1); dense grid: min gap '
                 f'{min(raw[1.2]):.1e} at r = 1.2, {min(raw[1.05]):.1e} at r = 1.05'),
        details=details,
        metrics=dict(slopes=slopes),
        data=dict(fracs=fracs, decay=decay, dense=dense.tolist(), raw=raw, rel=rel))


def ho_yeung(ph, eps):
    """Ho & Yeung (2010), Theorem 6: the maximum-entropy distribution within l1 distance eps
    of ph.  Clip the top at mu, raise the bottom to nu, where sum (ph_i - mu)_+ = eps/2 and
    sum (nu - ph_i)_+ = eps/2; uniform once nu >= mu."""
    ph = np.asarray(ph, float); n = len(ph)
    if eps >= 2.0:
        return np.ones(n) / n
    s = np.sort(ph)
    top = lambda m: np.sum(np.maximum(ph - m, 0.0)) - eps / 2       # decreasing in m
    bot = lambda v: np.sum(np.maximum(v - ph, 0.0)) - eps / 2       # increasing in v
    from scipy.optimize import brentq
    mu = brentq(top, s[0] - 1.0, s[-1])
    nu = brentq(bot, s[0], s[-1] + 1.0)
    if nu >= mu:
        return np.ones(n) / n
    return np.clip(ph, nu, mu)


@check('V4.2', 'the r = inf path is the Ho-Yeung maximum-entropy solution', 'prop:breakpoint')
def v4_2():
    """By prop:selfcert the r = inf solution is the maximum-entropy point of the l1 ball, so
    it must equal Ho & Yeung's closed form at every radius, which gives the whole path,
    every breakpoint included.  Falsifier: any gap above SOLVER_TOL."""
    rng = np.random.default_rng(57)
    worst = 0.0; cells = 0
    for _ in range(150):
        n = int(rng.choice([3, 4, 5, 6, 8]))
        ph = rng.dirichlet(np.ones(n) * rng.choice([0.5, 1.0, 3.0]))
        if ph.min() < 1e-3:
            continue
        sat = nrm(ph - np.ones(n) / n, 1)
        for fr in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.2]:
            eps = fr * sat
            worst = max(worst, float(np.max(np.abs(solve(ph, eps, np.inf, sat_slack=-2.0) - ho_yeung(ph, eps)))))
            cells += 1
    return Result(
        passed=worst < SOLVER_TOL,
        summary=f'max|solver - Ho-Yeung| = {worst:.1e} over {cells} (nominal, radius) pairs',
        metrics=dict(worst=worst, cells=cells))


def box_clip(ph, eps):
    """The r = 1 path in closed form: the maximum-entropy point of the box
    {q : |q_i - ph_i| <= eps} intersected with the simplex is q_i = clip(c, ph_i - eps, ph_i + eps)
    (lower end floored at 0), with c chosen so that sum q = 1 (KKT; Abellan & Moral 2003)."""
    from scipy.optimize import brentq
    ph = np.asarray(ph, float)
    lo, hi = np.maximum(ph - eps, 0.0), ph + eps
    f = lambda c: np.sum(np.clip(c, lo, hi)) - 1.0
    return np.clip(brentq(f, -1.0, 2.0, xtol=1e-15), lo, hi)


@check('V4.3', 'the r = 1 path is the box-clip maximum-entropy solution', 'prop:breakpoint1')
def v4_3():
    """The r = 1 solution is the maximum-entropy point of an l_inf box around ph, whose KKT
    solution is a single clipping level.  Checked along the whole path, past saturation."""
    rng = np.random.default_rng(58)
    worst = 0.0; cells = 0
    for _ in range(150):
        n = int(rng.choice([3, 4, 5, 6, 8]))
        ph = rng.dirichlet(np.ones(n) * rng.choice([0.5, 1.0, 3.0]))
        if ph.min() < 1e-3:
            continue
        sat = nrm(ph - np.ones(n) / n, np.inf)
        for fr in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.2]:
            eps = fr * sat
            worst = max(worst, float(np.max(np.abs(solve(ph, eps, 1, sat_slack=-2.0) - box_clip(ph, eps)))))
            cells += 1
    return Result(
        passed=worst < SOLVER_TOL,
        summary=f'max|solver - box clip| = {worst:.1e} over {cells} (nominal, radius) pairs',
        metrics=dict(worst=worst, cells=cells))


def _maxent_in_ball(ph, psi, c, x0):
    """Independent solver: argmax H(q) over the simplex with sum_i psi_i(q_i) <= c (SLSQP)."""
    from scipy.optimize import minimize
    n = len(ph)
    negH = lambda q: float(np.sum(np.clip(q, 1e-300, None) * np.log(np.clip(q, 1e-300, None))))
    cons = [dict(type='eq', fun=lambda q: q.sum() - 1.0),
            dict(type='ineq', fun=lambda q: c - psi(q))]
    res = minimize(negH, x0, method='SLSQP', bounds=[(1e-12, 1.0)] * n, constraints=cons,
                   options=dict(ftol=1e-15, maxiter=3000))
    return res.x


@check('V4.7', 'no fusion in smooth balls; fusion at the kinked endpoints', 'prop:nofusion')
def v4_7():
    """prop:nofusion.  For balls with a smooth, strictly convex, separable defining function
    (l_s with 1 < s < inf; chi^2, Hellinger and KL f-divergences) the maximum-entropy point keeps
    the strict order of ph: no originally distinct pair of coordinates fuses, and ties stay tied.
    For total variation (Ho-Yeung) and l_inf (box clip) fusion must occur on the same nominals.

    Pass: smooth balls: order preserved, and the smallest gap between originally distinct
    coordinates exceeds 1e-5, a hundred times the independent solver's precision (1e-7);
    tied coordinates agree to 1e-6; the KL solution matches the tempered form ph^t / Z to 1e-6.
    Kinked balls: some originally distinct pair is fused (gap < 1e-9) on every nominal.
    """
    rng = np.random.default_rng(59)
    noms = []
    while len(noms) < 20:
        n = int(rng.choice([3, 4, 5, 6]))
        ph = np.sort(rng.dirichlet(np.ones(n) * 2.0))[::-1]
        if np.min(-np.diff(ph)) < 0.01:
            continue
        noms.append(ph)
    noms.append(np.array([.40, .25, .25, .10]))            # a tie that must survive
    u_of = lambda ph: np.ones(len(ph)) / len(ph)
    balls = {
        'l_1.5': lambda ph: (lambda q: np.sum(np.abs(q - ph) ** 1.5), lambda: np.sum(np.abs(u_of(ph) - ph) ** 1.5)),
        'l_2':   lambda ph: (lambda q: np.sum((q - ph) ** 2),          lambda: np.sum((u_of(ph) - ph) ** 2)),
        'l_3':   lambda ph: (lambda q: np.sum(np.abs(q - ph) ** 3),    lambda: np.sum(np.abs(u_of(ph) - ph) ** 3)),
        'chi2':  lambda ph: (lambda q: np.sum((q - ph) ** 2 / ph),     lambda: np.sum((u_of(ph) - ph) ** 2 / ph)),
        'hell':  lambda ph: (lambda q: np.sum((np.sqrt(q) - np.sqrt(ph)) ** 2), lambda: np.sum((np.sqrt(u_of(ph)) - np.sqrt(ph)) ** 2)),
        'KL':    lambda ph: (lambda q: np.sum(q * np.log(np.clip(q, 1e-300, None) / ph)), lambda: np.sum(u_of(ph) * np.log(u_of(ph) / ph))),
    }
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_gap = {k: np.inf for k in balls}; order_bad = {k: 0 for k in balls}; tie_err = {k: 0.0 for k in balls}
    kl_err = 0.0
    for ph in noms:
        distinct = [(i, j) for i in range(len(ph)) for j in range(i + 1, len(ph)) if abs(ph[i] - ph[j]) > 1e-12]
        tied = [(i, j) for i in range(len(ph)) for j in range(i + 1, len(ph)) if abs(ph[i] - ph[j]) <= 1e-12]
        for name, mk in balls.items():
            psi, sat = mk(ph)
            for fr in fracs:
                q = _maxent_in_ball(ph, psi, fr * sat(), ph.copy())
                min_gap[name] = min(min_gap[name], min(abs(q[i] - q[j]) for i, j in distinct))
                order_bad[name] += int(any((q[i] - q[j]) * (ph[i] - ph[j]) <= 0 for i, j in distinct))
                if tied:
                    tie_err[name] = max(tie_err[name], max(abs(q[i] - q[j]) for i, j in tied))
                if name == 'KL':
                    from scipy.optimize import brentq
                    target = fr * sat()
                    def kl_t(t):
                        w = ph ** t; w /= w.sum(); return np.sum(w * np.log(w / ph)) - target
                    t = brentq(kl_t, 1e-9, 1.0)
                    w = ph ** t; w /= w.sum()
                    kl_err = max(kl_err, float(np.max(np.abs(q - w))))
    # contrast: kinked balls must fuse somewhere on every distinct-coordinate nominal
    fused_tv = fused_box = 0; n_distinct = 0
    for ph in noms:
        if np.min(-np.diff(np.sort(ph)[::-1])) <= 1e-12:
            continue
        n_distinct += 1
        gaps_tv = [np.min(-np.diff(np.sort(ho_yeung(ph, e))[::-1])) for e in np.linspace(0.01, 0.99, 60) * nrm(ph - np.ones(len(ph)) / len(ph), 1)]
        gaps_box = [np.min(-np.diff(np.sort(box_clip(ph, e))[::-1])) for e in np.linspace(0.01, 0.99, 60) * nrm(ph - np.ones(len(ph)) / len(ph), np.inf)]
        fused_tv += min(gaps_tv) < 1e-9
        fused_box += min(gaps_box) < 1e-9
    smooth_ok = all(order_bad[k] == 0 and min_gap[k] > 1e-5 and tie_err[k] < 1e-6 for k in balls)
    ok = smooth_ok and kl_err < 1e-6 and fused_tv == n_distinct and fused_box == n_distinct
    details = [f'{len(noms)} nominals (one with a tie), radii at {fracs} of the saturation value']
    details += [f'{k:>6}: order violations {order_bad[k]}, min gap {min_gap[k]:.1e}, tie error {tie_err[k]:.1e}'
                for k in balls]
    details.append(f'KL ball vs tempered ph^t/Z: max error {kl_err:.1e}')
    details.append(f'kinked balls fuse on {fused_tv}/{n_distinct} (TV) and {fused_box}/{n_distinct} (l_inf) distinct nominals')
    return Result(
        passed=ok,
        summary=(f'smooth balls: no fusion (min gap {min(min_gap.values()):.1e}); '
                 f'TV and l_inf fuse on {fused_tv}/{n_distinct} and {fused_box}/{n_distinct}'),
        details=details, metrics=dict(min_gap=min_gap, kl_err=kl_err))


@check('V4.6', 'at n = 3 the endpoint members coincide', 'rem:three')
def v4_6():
    """At n = 3, ||d||_inf = ||d||_1 / 2 for every sum-zero d, so D_1 = 2 D_inf and the r = 1
    path at eps is the r = inf path at 2 eps.  At n = 4 the two members differ."""
    rng = np.random.default_rng(510)
    pen = max(abs(D(v, 1) - 2 * D(v, np.inf)) for v in rng.normal(size=(2000, 3)))
    path = 0.0; cells = 0
    for _ in range(150):
        ph = rng.dirichlet(np.ones(3) * rng.choice([0.5, 1.0, 3.0]))
        if ph.min() < 1e-3:
            continue
        sat = nrm(ph - np.ones(3) / 3, np.inf)
        for fr in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.2]:
            eps = fr * sat
            path = max(path, float(np.max(np.abs(solve(ph, eps, 1) - solve(ph, 2 * eps, np.inf)))))
            cells += 1
    v4 = rng.normal(size=(2000, 4))
    differ = float(np.max(np.abs([D(v, 1) - 2 * D(v, np.inf) for v in v4])))
    return Result(
        passed=pen < EXACT_TOL and path < SOLVER_TOL and differ > 0.1,
        summary=(f'n = 3: max|D_1 - 2 D_inf| = {pen:.1e} over 2000 vectors; '
                 f'max|x_1(eps) - x_inf(2 eps)| = {path:.1e} over {cells} (nominal, radius) pairs; '
                 f'n = 4: max|D_1 - 2 D_inf| = {differ:.2f}'),
        metrics=dict(penalty=pen, path=path, cells=cells, differ_n4=differ))


@check('V4.9', 'the near-endpoint gap is exponentially small in 1/(r-1)', 'rem:fusiongap')
def v4_9():
    """Near the r = 1 endpoint the minimum coordinate gap d obeys d ~ (Delta/(eps C))^{1/(r-1)},
    exponentially small in 1/(r-1): log d is linear in 1/(r-1).  On a fixed nominal at
    eps/eps_sat = 0.9, over r in {1.05, ..., 1.3}, fit log10(d) against 1/(r-1) and check the fit is
    essentially exact; the slope is log10(Delta/(eps C)) for an order-one C.
    Pass (fixed before the run): R^2 > 0.999 and max residual < 0.05 dex.
    """
    p = np.array([0.40, 0.26, 0.17, 0.11, 0.06]); u = np.ones(len(p)) / len(p); frac = 0.9

    def gap(r):
        x = solve(p, frac * nrm(p - u, dual(r)), r)
        return float(np.min(np.diff(np.sort(x))))

    rs = [1.05, 1.1, 1.15, 1.2, 1.3]
    d = [gap(r) for r in rs]
    xx = np.array([1 / (r - 1) for r in rs]); yy = np.log10(np.array(d))
    A = np.polyfit(xx, yy, 1); resid = yy - np.polyval(A, xx)
    ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot; maxres = float(np.max(np.abs(resid)))
    rs_fig = [float(np.round(r, 4)) for r in np.linspace(1.03, 1.5, 24)]
    d_fig = [gap(r) for r in rs_fig]
    return Result(
        passed=(r2 > 0.999 and maxres < 0.05),
        summary=f'log10(d) vs 1/(r-1): R^2 = {r2:.5f}, slope {A[0]:.3f}, max residual {maxres:.3f} dex; '
                f'd is exponentially small in 1/(r-1)',
        details=[f'r={r}: d={dd:.2e}' for r, dd in zip(rs, d)],
        metrics=dict(slope=float(A[0]), intercept=float(A[1]), r2=r2, maxres=maxres),
        data=dict(frac=frac, rs=rs_fig, d=d_fig, slope=float(A[0]), intercept=float(A[1])))
