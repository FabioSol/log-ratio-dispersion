"""Chapter 3 checks: the keystone and the closed form."""
import numpy as np
from scipy.optimize import minimize_scalar

from . import Result, check
from ..core import centre, D, nrm, is_degenerate
from ..samplers import PH5, degenerate_sampler


@check('V3.3', 'the centre kappa_r is not monotone in r', 'obs:kappa')
def v3_3():
    """kappa_r(x) on the grid r = 1, 1.5, 2, 3, 6, inf is not monotone, and its continuous minimiser
    over r lies strictly inside (1.5, 3), close to the geometric mean (r = 2)."""
    rs = [1, 1.5, 2, 3, 6, np.inf]
    kappa = [float(np.exp(centre(np.log(PH5), r))) for r in rs]
    d = np.diff(kappa)
    monotone = bool(np.all(d >= -1e-15) or np.all(d <= 1e-15))
    k = lambda r: float(np.exp(centre(np.log(PH5), r)))
    res = minimize_scalar(k, bounds=(1.2, 6.0), method='bounded', options=dict(xatol=1e-8))
    r_min, k_min = float(res.x), float(res.fun)
    return Result(
        passed=(not monotone) and 1.5 < r_min < 3.0,
        summary=f'non-monotone; minimum {k_min:.5f} at r = {r_min:.3f} (kappa at r = 2: {k(2):.5f})',
        details=[f'r = {str(r):>4}    kappa_r = {kk:.5f}' for r, kk in zip(rs, kappa)],
        metrics=dict(monotone=monotone, r_min=r_min, kappa_min=k_min),
        data=dict(rs=rs, kappa=kappa))


@check('V3.2', 'closed form vs numerical minimisation', 'eq:closedform')
def v3_2():
    rng = np.random.default_rng(32)
    worst = 0.0
    for _ in range(600):
        n = int(rng.choice([3, 5, 7, 8, 9]))
        lx = np.log(rng.dirichlet(np.ones(n) * .6))
        for r in [1, 1.5, 2, 3, np.inf]:
            res = minimize_scalar(lambda c: nrm(lx - c, r), bounds=(lx.min(), lx.max()),
                                  method='bounded', options=dict(xatol=1e-14))
            worst = max(worst, (D(lx, r) - res.fun) / max(1., abs(res.fun)))
    return Result(
        passed=worst < 1e-9,
        summary=f'max relative excess of D over numerical min = {worst:.2e} (want < 1e-9)',
        metrics=dict(worst=worst))


@check('V3.1', 'the r-centre equation', 'cor:centre')
def v3_1():
    rng = np.random.default_rng(33)
    worst = 0.0
    for _ in range(1500):
        v = degenerate_sampler(rng)
        if is_degenerate(v):
            continue
        for r in [1.2, 1.5, 3, 6]:
            res = minimize_scalar(lambda c: nrm(v - c, r), bounds=(v.min(), v.max()),
                                  method='bounded', options=dict(xatol=1e-14))
            worst = max(worst, (nrm(v - centre(v, r), r) - res.fun) / max(1., abs(res.fun)))
    return Result(
        passed=worst < 1e-9,
        summary=f'max relative excess = {worst:.2e} (want < 1e-9)',
        metrics=dict(worst=worst))


@check('V3.9', 'the keystone construction transfers to a general sum-zero set G', 'sec:beyond')
def v3_9():
    """Beyond norm balls: for a compact convex G in 1^perp with 0 in its relative interior the
    penalty is the support function sigma_G, and the path relation, the saturation gauge, and
    distance-equals-radius transfer.  Verified on two ASYMMETRIC landmark sets, MbLS
    (G = u - Delta) and contamination (G = Delta - p_hat), over random nominals:
      * path relation: g = (x* - p_hat)/eps is in the subdifferential, i.e. <g, f> = sigma_G(f),
        f = -log x*;
      * distance equals radius: gauge_G(x* - p_hat) = eps;
      * saturation: the solver returns u for eps >= eps_sat = gauge_G(u - p_hat).
    Pass (fixed before the run): each residual < 1e-5.
    """
    from ..general_g import mbls, contamination, solve
    rng = np.random.default_rng(39)
    worst = dict(path=0.0, gauge=0.0, sat=0.0)
    details = []
    for maker in (mbls, contamination):
        wp = wg = ws = 0.0; m = 0
        for _ in range(12):
            n = int(rng.choice([3, 4, 5])); ph = rng.dirichlet(np.ones(n) * 1.5); u = np.ones(n) / n
            G = maker(n, ph)
            esat = G['gauge'](u - ph); eps = 0.5 * esat
            x = solve(ph, eps, G); f = -np.log(x); g = (x - ph) / eps
            wp = max(wp, abs(g @ f - G['sigma'](f)))
            wg = max(wg, abs(G['gauge'](x - ph) - eps))
            xs = solve(ph, esat * 1.02, G)
            ws = max(ws, float(np.max(np.abs(xs - u))))
            m += 1
        worst['path'] = max(worst['path'], wp); worst['gauge'] = max(worst['gauge'], wg)
        worst['sat'] = max(worst['sat'], ws)
        details.append(f"{G['name']}: {m} nominals; path {wp:.1e}, gauge {wg:.1e}, saturation {ws:.1e}")
    ok = worst['path'] < 1e-5 and worst['gauge'] < 1e-5 and worst['sat'] < 1e-5
    return Result(
        passed=ok,
        summary=(f"general-G path relation {worst['path']:.1e}, gauge=radius {worst['gauge']:.1e}, "
                 f"saturation {worst['sat']:.1e} on MbLS and contamination sets"),
        details=details, metrics=worst)


@check('V3.8', 'off the ray, members and relatives move in different directions', 'sec:beyond')
def v3_8():
    """On p_hat = (0.7, 0.22, 0.08), as the penalty strength -> 0, the share of top-class mass
    that lands on the MIDDLE class has a closed form that distinguishes the methods:
      r=inf: 0 (the first phase moves only the extremes, Prop 4.2);
      r=2:   w2/(w2+w3), w_i = log(kappa_2(p)/p_i)                 (= 4.4%);
      label smoothing: (1/n - p2)/(p1 - 1/n)                       (= 30.9%, constant in alpha);
      confidence penalty: d2/(-d1), d_i = p_i(-log p_i - H(p))     (= 53.5%).
    Pass (fixed before the run): the closed forms equal {0, .044, .309, .535} to 1e-3, the four
    are pairwise distinct by > 4 points, and the solver's small-eps direction (eps = 1e-3)
    approaches the r-limits to 1 point.  (The r=2 share then grows as its path curves.)
    """
    from ..solver import solve as solve_r
    ph = np.array([0.7, 0.22, 0.08]); n = 3; u = np.ones(n) / n
    dshare = lambda d: float(d[1] / -d[0])            # middle share of a direction d with d0 < 0
    lw = np.log(ph); w = lw.mean() - lw               # centred -log p (the r=2 direction)
    H = -(ph * np.log(ph)).sum(); d = ph * (-np.log(ph) - H)
    lim = dict(r_inf=0.0,
               r_2=float(w[1] / (w[1] + w[2])),
               label_smoothing=float((u[0] - ph[1]) / (ph[0] - u[0])),
               confidence=float(d[1] / -d[0]))
    target = dict(r_inf=0.0, r_2=0.044, label_smoothing=0.309, confidence=0.535)
    close = all(abs(lim[k] - target[k]) < 1e-3 for k in lim)
    s = sorted(lim.values()); distinct = all(s[i + 1] - s[i] > 0.04 for i in range(3))
    s_inf = dshare(solve_r(ph, 1e-3, np.inf) - ph)
    s_2 = dshare(solve_r(ph, 1e-3, 2.0) - ph)
    s_2_004 = dshare(solve_r(ph, 0.04, 2.0) - ph)        # the r=2 share grows as the path curves
    approach = abs(s_inf - lim['r_inf']) < 0.01 and abs(s_2 - lim['r_2']) < 0.01
    grows = abs(s_2_004 - 0.12) < 0.02                   # about 12% at eps = 0.04, as the text says
    order = ['r_inf', 'r_2', 'label_smoothing', 'confidence']
    return Result(
        passed=close and distinct and approach and grows,
        summary='eps->0 middle share: '
                + ', '.join(f'{k}={100 * lim[k]:.1f}%' for k in order)
                + f'; r=2 grows to {100 * s_2_004:.0f}% at eps=0.04',
        details=[f'closed forms vs targets ok={close}, distinct={distinct}',
                 f'solver at eps=1e-3: r=inf {100 * s_inf:.1f}%, r=2 {100 * s_2:.1f}% '
                 f'(limits {100 * lim["r_inf"]:.1f}%, {100 * lim["r_2"]:.1f}%); '
                 f'r=2 at eps=0.04: {100 * s_2_004:.1f}%'],
        metrics=dict(lim, r2_at_004=s_2_004))


@check('V3.10', 'the Aitchison geodesic is the KL-ball path, not the r=2 path', 'prop:aitchison')
def v3_10():
    """prop:aitchison is an identity of penalty value only.  The Aitchison geodesic from p_hat
    to u is the straight line in clr coordinates, x proportional to p_hat^s, so a point is on
    it exactly when clr(x) is parallel to clr(p_hat).  Off-geodesic residual of x:
    ||clr(x) - (clr(x).e) e|| / ||clr(x)||, e = clr(p_hat)/||clr(p_hat)||.
      * the maximum-entropy point of the KL ball {q : KL(q || p_hat) <= c} lies on the geodesic
        (the tempered form of prop:nofusion's remark);
      * the r=2 robust path does not;
      * on p_hat = (0.5, 0.25, 0.15, 0.10) the two paths start the second coordinate in
        opposite directions: the r=2 path lowers it (0.25 is above the geometric mean 0.208),
        the geodesic raises it (0.25 is below the p-weighted mean exp(sum p log p) = 0.299).
    Pass (fixed before the run): KL residual < 1e-6 on every nominal and radius (the
    independent solver's precision, as in V4.7); r=2 residual > 1e-2 at half the saturation
    radius on the example; the two signs at eps = 1e-3 * eps_sat opposite.
    """
    from .path import _maxent_in_ball
    from ..solver import solve as solve_r, saturation_radius

    def clr(x):
        lx = np.log(x)
        return lx - lx.mean()

    def off(x, ph):
        e = clr(ph) / np.linalg.norm(clr(ph)); c = clr(x)
        return float(np.linalg.norm(c - (c @ e) * e) / np.linalg.norm(c))

    rng = np.random.default_rng(310)
    noms = [np.array([0.5, 0.25, 0.15, 0.10]), np.array([0.7, 0.22, 0.08])]
    noms += [rng.dirichlet(np.ones(n) * 2.0) for n in (3, 4, 5, 6)]
    fracs = [0.1, 0.5, 0.9]
    kl_off = 0.0; r2_off = 0.0
    for ph in noms:
        u = np.ones(len(ph)) / len(ph)
        psi = lambda q, ph=ph: np.sum(q * np.log(np.clip(q, 1e-300, None) / ph))
        csat = float(np.sum(u * np.log(u / ph)))
        for fr in fracs:
            kl_off = max(kl_off, off(_maxent_in_ball(ph, psi, fr * csat, ph.copy()), ph))
            r2_off = max(r2_off, off(solve_r(ph, fr * saturation_radius(ph, 2.0), 2.0), ph))
    ph = noms[0]; u = np.ones(4) / 4
    ex_off = off(solve_r(ph, 0.5 * saturation_radius(ph, 2.0), 2.0), ph)
    d_r2 = solve_r(ph, 1e-3 * saturation_radius(ph, 2.0), 2.0)[1] - ph[1]
    psi = lambda q: np.sum(q * np.log(np.clip(q, 1e-300, None) / ph))
    d_kl = _maxent_in_ball(ph, psi, 1e-3 * float(np.sum(u * np.log(u / ph))), ph.copy())[1] - ph[1]
    ok = kl_off < 1e-6 and ex_off > 1e-2 and d_r2 < 0 < d_kl
    return Result(
        passed=ok,
        summary=(f'KL-ball path off the geodesic by {kl_off:.1e}; r=2 path by up to {r2_off:.2f}; '
                 f'second coordinate r=2 {d_r2:+.1e}, geodesic {d_kl:+.1e}'),
        details=[f'{len(noms)} nominals, radii at {fracs} of saturation',
                 f'KL-ball max-entropy point: max off-geodesic residual {kl_off:.1e}',
                 f'r=2 robust path: max off-geodesic residual {r2_off:.3f} '
                 f'({ex_off:.3f} on (0.5, 0.25, 0.15, 0.10) at half saturation)',
                 f'second coordinate at eps = 1e-3 eps_sat: r=2 {d_r2:+.2e}, KL/geodesic {d_kl:+.2e}'],
        metrics=dict(kl_off=kl_off, r2_off=r2_off, ex_off=ex_off, d_r2=d_r2, d_kl=d_kl))
