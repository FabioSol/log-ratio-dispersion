"""Chapter 6 checks: the radius for estimation.

V6.1, V6.3, V6.4 and V6.6-V6.10 are the slow experiments (minutes each).
"""
import numpy as np

from . import Result, check
from ..core import subgrad, centre, D, dual, nrm, is_degenerate
from ..prob import kl
from ..solver import solve


def _eps_hat(ph, N, r, B=200, rng=None):
    """Parametric-bootstrap estimate of E||p_hat - p||_{r*} at sample size N.

    Resamples from p_hat, so it uses only what the estimator has.  It UNDERESTIMATES
    the true expectation when p_hat is degenerate (few observed classes), the small-N
    regime the robust code is supposed to help in, so it is the conservative choice.
    """
    rng = rng or np.random.default_rng(0)
    p = dual(r)
    draws = rng.multinomial(N, ph, size=B) / N
    return float(np.mean([nrm(d - ph, p) for d in draws]))


@check('V6.1', 'the robust code loses to KT', slow=True)
def v6_1(quick=True):
    """Robust code at radius c * E||p_hat - p|| vs plug-in, KT (add-1/2), add-1.

    quick=True is the design quoted in the thesis (n in {4, 8}, N in {8, 32, 128},
    120 draws); quick=False is the fuller design (n in {4, 8, 16}, N in {8, ..., 128},
    400 draws).  The draws are unchanged from the original check; the per-draw losses
    are kept so the comparison with KT can be paired.

    Reported per member r and per c: raw wins (cell mean below KT) and significant wins
    (the whole 95% paired-bootstrap interval of the mean difference robust - KT below
    zero), significant losses, and the median relative margin (KT - robust) / KT of the
    significant wins.  Also the best-of-three count, where the member is chosen per cell
    after seeing the results (the count the thesis first quoted).

    The certificate reading predicts that robust at c = 1 beats KT in most cells.  The
    check PASSES when that prediction fails for every member: significant wins at c = 1
    in at most half the cells.
    """
    ns = [4, 8] if quick else [4, 8, 16]
    Ns = [8, 32, 128] if quick else [8, 16, 32, 64, 128]
    alphas = [0.3, 1.0, 3.0]
    draws = 120 if quick else 400
    rs = [1, 2, np.inf]
    cs = [0.5, 1.0, 2.0]
    rng = np.random.default_rng(72)
    brng = np.random.default_rng(720)
    B = 2000

    rows = []
    cells = 0
    plugin_inf = plugin_tot = 0
    raw = {(r, c): 0 for r in rs for c in cs}
    sig = {(r, c): 0 for r in rs for c in cs}
    sigloss = {(r, c): 0 for r in rs for c in cs}
    margins = {(r, c): [] for r in rs for c in cs}
    where = {(r, c): [] for r in rs for c in cs}
    best = {c: 0 for c in cs}
    for n in ns:
        for alpha in alphas:
            for N in Ns:
                acc = {('plugin',): [], ('kt',): [], ('add1',): []}
                for r in rs:
                    for c in cs:
                        acc[(r, c)] = []
                for _ in range(draws):
                    p = rng.dirichlet(np.ones(n) * alpha)
                    cnt = rng.multinomial(N, p)
                    ph = cnt / N
                    k = kl(p, ph)
                    plugin_tot += 1
                    plugin_inf += np.isinf(k)
                    acc[('plugin',)].append(k)
                    acc[('kt',)].append(kl(p, (cnt + 0.5) / (N + n * 0.5)))
                    acc[('add1',)].append(kl(p, (cnt + 1.0) / (N + n)))
                    for r in rs:
                        e1 = _eps_hat(ph, N, r, B=60, rng=rng)
                        for c in cs:
                            acc[(r, c)].append(kl(p, solve(ph, c * e1, r)))
                A = {k: np.array(v) for k, v in acc.items()}
                m = {k: float(np.mean(v)) for k, v in A.items()}
                cells += 1
                idx = brng.integers(0, draws, (B, draws))
                for r in rs:
                    for c in cs:
                        d = A[(r, c)] - A[('kt',)]
                        lo, hi = np.quantile(d[idx].mean(1), [.025, .975])
                        raw[(r, c)] += d.mean() < 0
                        if hi < 0:
                            sig[(r, c)] += 1
                            margins[(r, c)].append(-d.mean() / m[('kt',)])
                            where[(r, c)].append(alpha)
                        if lo > 0:
                            sigloss[(r, c)] += 1
                for c in cs:
                    best[c] += (min(m[(r, c)] for r in rs) < m[('kt',)])
                rows.append((n, alpha, N, m))

    details = [f"{'n':>3} {'alpha':>6} {'N':>4} {'plugin':>9} {'KT':>8} {'add1':>8} "
               f"{'best c=.5':>9} {'best c=1':>9} {'best c=2':>9}"]
    for n, alpha, N, m in rows:
        pl = 'inf' if np.isinf(m[('plugin',)]) else f"{m[('plugin',)]:.4f}"
        details.append(f"{n:>3} {alpha:>6} {N:>4} {pl:>9} {m[('kt',)]:>8.4f} "
                       f"{m[('add1',)]:>8.4f} "
                       + " ".join(f"{min(m[(r, c)] for r in rs):>9.4f}" for c in cs))
    details.append(f'plug-in infinite in {plugin_inf}/{plugin_tot} draws '
                   f'({plugin_inf / plugin_tot:.1%}), which is why beating plug-in is not evidence')
    details.append('per member, against KT (raw wins / significant wins / significant losses; '
                   'median margin of significant wins; alpha of significant wins):')
    for r in rs:
        for c in cs:
            mg = f'{np.median(margins[(r, c)]):.1%}' if margins[(r, c)] else '-'
            details.append(f'  r={str(r):>3} c={c}: {raw[(r, c)]:>2} / {sig[(r, c)]:>2} / '
                           f'{sigloss[(r, c)]:>2} of {cells};  margin {mg};  alpha {where[(r, c)]}')
    certificate_reading = any(sig[(r, 1.0)] > cells / 2 for r in rs)
    return Result(
        passed=not certificate_reading,
        summary=('best of three beats KT in ' + ', '.join(f'{best[c]}/{cells} at c={c}' for c in cs)
                 + '; significant wins at c=1 per member: '
                 + ', '.join(f'r={r}: {sig[(r, 1.0)]}' for r in rs)),
        details=details,
        metrics=dict(best=best, cells=cells, raw={str(k): v for k, v in raw.items()},
                     sig={str(k): v for k, v in sig.items()},
                     sigloss={str(k): v for k, v in sigloss.items()},
                     margins={str(k): v for k, v in margins.items()},
                     where={str(k): v for k, v in where.items()},
                     plugin_inf=plugin_inf / plugin_tot))


def _refined_log_argmin(log_grid, risk):
    """argmin of a risk curve on a uniform log grid, refined by the parabola through
    the grid minimum and its two neighbours (the grid alone quantises eps* to a
    factor of ~1.4).  Falls back to the grid point at an edge or a flat/concave triple."""
    i = int(np.argmin(risk))
    if i == 0 or i == len(risk) - 1:
        return float(log_grid[i])
    y0, y1, y2 = risk[i - 1:i + 2]
    curv = y0 - 2 * y1 + y2
    if curv <= 0:
        return float(log_grid[i])
    h = log_grid[i + 1] - log_grid[i]
    return float(log_grid[i] + 0.5 * h * (y0 - y2) / curv)


def rate_constant(p, r, h=1e-6):
    """The prop:rate constant c_r(p) with eps*(N) ~ c_r / N.

    Expanding E KL(p || x*(eps)) to second order gives R(eps) ~ V - (a/N) eps + b eps^2,
    with g the subgradient of D_r at -log p and Sigma = diag(p) - pp':
        a  = sum_i g_i / p_i  +  tr(H M),   M = diag(1/p) Sigma diag(1/p),
        2b = sum_i g_i^2 / p_i,
    H the Hessian of D_r at -log p (1 < r < inf; at r in {1, inf} g is locally constant and
    the trace term vanishes).  sum g/p > 0 by Chebyshev's sum inequality and tr(HM) >= 0
    since H, M are PSD, so c_r = a / (2b) > 0 at every r.  Returns (a, 2b, c_r).
    """
    p = np.asarray(p, float); n = len(p); v = -np.log(p)
    g = subgrad(v, r)
    tr = 0.0
    if 1 < r < np.inf:
        H = np.column_stack([(subgrad(v + h * np.eye(n)[j], r) - subgrad(v - h * np.eye(n)[j], r)) / (2 * h)
                             for j in range(n)])
        M = np.diag(1 / p) @ (np.diag(p) - np.outer(p, p)) @ np.diag(1 / p)
        tr = float(np.trace(0.5 * (H + H.T) @ M))
    a = float(np.sum(g / p)) + tr
    twob = float(np.sum(g ** 2 / p))
    return a, twob, a / twob


def _slope(logN, logy):
    return float(np.polyfit(logN, logy, 1)[0])


@check('V6.6', 'the rate mismatch', 'prop:rate', slow=True)
def v6_6(B=2000):
    """The certificate radius E||p_hat - p||_{r*} decays like N^{-1/2}; the risk-optimal
    radius eps* = argmin_eps E KL(p || x*(eps)) like N^{-1}.

    eps* is the refined argmin of the draw-averaged risk curve (_refined_log_argmin).
    Both slopes get a 90% bootstrap interval by resampling draws within each N (B
    resamples; the per-draw losses are kept, so no re-solving).

    Pass, for EVERY member r (the chapter claims every member, not an average):
      * certificate slope within 0.05 of -1/2 (the rate is asymptotic; the allowance
        covers finite-N bias), and
      * the upper end of the eps* slope interval below -1/2, i.e. eps* decays
        significantly faster than any radius that must contain the error, and
      * every eps* strictly inside the grid (an argmin on the edge is censored), and
      * the measured N eps* at the largest N within 30% of the predicted constant
        rate_constant(p, r).  (Rule set after the V6.6 data were seen; the allowance
        covers the endpoints' decaying order-flip correction.)

    The fitted slopes are pre-asymptotic at N <= 1024 (r = 2 approaches -1 from above),
    so the N^{-1} rate is tested through its constant instead: N eps* against c_r(p).
    """
    rng = np.random.default_rng(7)
    p = np.array([.5, .25, .15, .10])
    Ns = [16, 32, 64, 128, 256, 512, 1024]
    grid = np.geomspace(3e-5, 0.6, 29)      # log step ~0.35; eps* at N = 1024 is ~4e-4
    lg = np.log(grid)
    logN = np.log(Ns)
    draws = 1000
    brng = np.random.default_rng(73)
    details = []
    cert_all, star_all, fits, band_all = {}, {}, {}, {}
    ok = True
    for r in [1, 2, np.inf]:
        rs = dual(r)
        norms = np.empty((len(Ns), draws))
        loss = np.empty((len(Ns), draws, len(grid)))
        for ni, N in enumerate(Ns):
            cnts = rng.multinomial(N, p, size=draws) / N
            norms[ni] = [nrm(ph - p, rs) for ph in cnts]
            for di, ph in enumerate(cnts):
                loss[ni, di] = [kl(p, solve(ph, float(e), r)) for e in grid]
        cert = norms.mean(1)
        log_star = np.array([_refined_log_argmin(lg, loss[ni].mean(0)) for ni in range(len(Ns))])
        interior = all(0 < int(np.argmin(loss[ni].mean(0))) < len(grid) - 1 for ni in range(len(Ns)))
        lc, ls = _slope(logN, np.log(cert)), _slope(logN, log_star)
        bc, bs, bper = [], [], []
        for _ in range(B):
            idx = brng.integers(0, draws, (len(Ns), draws))
            bc.append(_slope(logN, np.log(np.take_along_axis(norms, idx, 1).mean(1))))
            per = [_refined_log_argmin(lg, loss[ni, idx[ni]].mean(0)) for ni in range(len(Ns))]
            bs.append(_slope(logN, per)); bper.append(per)
        band = np.exp(np.quantile(bper, [.05, .95], axis=0))
        cci = tuple(float(q) for q in np.quantile(bc, [.05, .95]))
        sci = tuple(float(q) for q in np.quantile(bs, [.05, .95]))
        c_pred = rate_constant(p, r)[2]
        c_meas = Ns[-1] * float(np.exp(log_star[-1]))
        member_ok = interior and abs(lc + 0.5) <= 0.05 and sci[1] < -0.5 and abs(c_meas / c_pred - 1) <= 0.3
        ok &= member_ok
        cert_all[r], star_all[r] = cert.tolist(), np.exp(log_star).tolist()
        band_all[r] = band.tolist()
        fits[r] = dict(cert=lc, cert_ci=cci, star=ls, star_ci=sci, c_pred=c_pred, c_meas=c_meas)
        details.append(f'r={str(r):>4}: slope_cert={lc:+.3f} [{cci[0]:+.3f}, {cci[1]:+.3f}]   '
                       f'slope_eps*={ls:+.3f} [{sci[0]:+.3f}, {sci[1]:+.3f}]   '
                       f"N eps* = {c_meas:.2f} vs predicted {c_pred:.2f}   "
                       f"{'ok' if member_ok else 'FAILS'}")
    return Result(
        passed=ok,
        summary=('eps* slopes ' + ', '.join(f'{f["star"]:+.2f}' for f in fits.values())
                 + ' (r = 1, 2, inf) vs certificate ' + ', '.join(f'{f["cert"]:+.2f}' for f in fits.values())),
        details=['90% bootstrap intervals over draws; pass needs every member:'] + details,
        metrics=dict(fits=fits),
        data=dict(Ns=Ns, cert=cert_all, star=star_all, star_band=band_all))


def hausser_strimmer(counts):
    """Hausser & Strimmer (2009), eq. (5): James-Stein shrinkage of the empirical frequencies
    toward the uniform target, with the data-driven intensity
        lambda = (1 - sum th^2) / ((N - 1) sum (1/n - th)^2),  truncated to [0, 1]."""
    counts = np.asarray(counts, float); N = counts.sum(); n = len(counts)
    th = counts / N
    den = (N - 1) * np.sum((1.0 / n - th) ** 2)
    lam = 1.0 if den <= 0 else float(np.clip((1 - np.sum(th ** 2)) / den, 0.0, 1.0))
    return lam / n + (1 - lam) * th


@check('V6.10', 'the geometry adds nothing over shrinkage', 'obs:geom', slow=True)
def v6_10():
    """Oracle-tuned family vs oracle-tuned uniform shrinkage toward u.

        family*  min over r, eps  of mean_draws KL(p || solve(p_hat, eps, r))
        shrink*  min over beta    of mean_draws KL(p || (1-beta) p_hat + beta u)

    Both hyperparameters use the true p, so adaptivity cancels and the only difference
    is the geometry.  Falsifier: family* < 0.98 * shrink* in ANY cell.

    Also reported, not judged: the Hausser-Strimmer estimator, the practical (data-driven)
    version of uniform shrinkage, against oracle shrinkage.  Its intensity estimate is 0 when
    every draw falls in one class, which returns the plug-in frequencies and an infinite
    redundancy; those cells are counted, not averaged.
    """
    ns = [4, 8]
    Ns = [8, 32, 128]
    alphas = [0.3, 1.0, 3.0]
    rs = [1, 2, np.inf]
    draws = 200
    egrid = np.geomspace(1e-4, 1.0, 14)
    bgrid = np.concatenate([[0.0], np.geomspace(1e-4, 0.99, 18)])
    rng = np.random.default_rng(99)
    details = []
    ratios = []
    hs_ratios = []
    geom_wins = 0
    for n in ns:
        u = np.ones(n) / n
        for alpha in alphas:
            for N in Ns:
                P = rng.dirichlet(np.ones(n) * alpha, size=draws)
                frisk = {r: np.zeros(len(egrid)) for r in rs}
                srisk = np.zeros(len(bgrid))
                hs_risk = 0.0
                for di in range(draws):
                    p = P[di]
                    cnt = rng.multinomial(N, p)
                    ph = cnt / N
                    hs_risk += kl(p, hausser_strimmer(cnt))
                    for gi, b in enumerate(bgrid):
                        srisk[gi] += kl(p, (1 - b) * ph + b * u)
                    for r in rs:
                        for gi, e in enumerate(egrid):
                            frisk[r][gi] += kl(p, solve(ph, float(e), r))
                shrink_o = float(np.min(srisk / draws))
                fam_by_r = {r: float(np.min(frisk[r] / draws)) for r in rs}
                best_r = min(rs, key=lambda r: fam_by_r[r])
                ratio = fam_by_r[best_r] / shrink_o if shrink_o > 0 else np.nan
                ratios.append(ratio)
                hs = hs_risk / draws
                hs_ratios.append(hs / shrink_o)          # inf when some sample is single-class
                geom_wins += ratio < 0.98
                details.append(f'n={n} a={alpha} N={N:>3}: shrink*={shrink_o:.4f} '
                               f'family*={fam_by_r[best_r]:.4f} ratio={ratio:.3f} best_r={best_r!s} '
                            f'HS={hs:.4f}')
    return Result(
        passed=geom_wins == 0,
        summary=(f'family beats oracle shrinkage by >2% in {geom_wins}/{len(ratios)} cells; '
                 f'family*/shrink* in [{np.min(ratios):.3f}, {np.max(ratios):.3f}]; '
                 f'Hausser-Strimmer / oracle shrinkage in '
                 f'[{np.min([x for x in hs_ratios if np.isfinite(x)]):.3f}, '
                 f'{np.max([x for x in hs_ratios if np.isfinite(x)]):.3f}] where finite, infinite in '
                 f'{sum(not np.isfinite(x) for x in hs_ratios)}/{len(hs_ratios)} cells (single-class samples '
                 f'give intensity 0)'),
        details=details,
        metrics=dict(geom_wins=geom_wins, ratios=ratios, hs_ratios=hs_ratios))


def _risk_matrix(p, N, r, grid, draws, rng):
    """Per-draw losses KL(p || x*(eps)) - eps * <p_hat - p, g>_Phi on a grid of eps.

    The subtracted term uses g, the subgradient at the TRUE p, so it has mean exactly zero
    (E p_hat = p) and leaves every expectation unchanged.  It is a control variate: it removes
    the per-draw fluctuation of order eps / sqrt(N), which otherwise swamps the O(eps / N)
    benefit being measured once N is large.
    """
    g = subgrad(-np.log(p), r)
    cnts = rng.multinomial(N, p, size=draws) / N
    L = np.empty((draws, len(grid)))
    for d, ph in enumerate(cnts):
        cv = float(np.sum((ph - p) * g / p))
        L[d] = [kl(p, solve(ph, float(e), r)) - float(e) * cv for e in grid]
    return L


_STUDY_SOURCES = [np.array([.5, .25, .15, .10]), np.array([.45, .30, .17, .08])]
_STUDY_NS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
_STUDY_GRID = np.geomspace(1e-6, 0.6, 55)
_STUDY_DRAWS, _STUDY_BATCHES = 4000, 8
_T90 = {7: 1.895}                                   # t quantile, 0.95, for 8 batches


def _study_cell(job):
    """One (source, member, N): the risk-optimal radius from all draws, and the same estimate
    on each of _STUDY_BATCHES independent batches of draws."""
    si, ri, ni = job
    p, r, N = _STUDY_SOURCES[si], [1, 2, np.inf][ri], _STUDY_NS[ni]
    rng = np.random.default_rng(7400 + 100 * si + 10 * ri + ni)
    L = _risk_matrix(p, N, r, _STUDY_GRID, _STUDY_DRAWS, rng)
    lg = np.log(_STUDY_GRID)
    point = float(np.exp(_refined_log_argmin(lg, L.mean(0))))
    batches = [float(np.exp(_refined_log_argmin(lg, b.mean(0))))
               for b in np.array_split(L, _STUDY_BATCHES)]
    return si, ri, ni, point, batches


_STUDY = {}


def _eps_star_study(jobs=8):
    """N eps*(N) on two sources, three members, N = 16 .. 8192, shared by V6.4 and V6.7 so the
    two report one set of numbers.  eps* is the refined argmin of the draw-averaged
    control-variate risk (_refined_log_argmin); 4000 draws per point.  Intervals come from
    8 independent batches of draws (t-interval on the batch estimates), not from a bootstrap,
    which understates the noise of an argmin.  Computed once per process."""
    if not _STUDY:
        from concurrent.futures import ProcessPoolExecutor
        todo = [(si, ri, ni) for si in range(2) for ri in range(3) for ni in range(len(_STUDY_NS))]
        with ProcessPoolExecutor(jobs) as ex:
            for si, ri, ni, point, batches in ex.map(_study_cell, todo):
                N = _STUDY_NS[ni]; b = N * np.array(batches)
                half = _T90[_STUDY_BATCHES - 1] * b.std(ddof=1) / np.sqrt(_STUDY_BATCHES)
                cell = _STUDY.setdefault((si, str([1, 2, np.inf][ri])), dict(
                    c=rate_constant(_STUDY_SOURCES[si], [1, 2, np.inf][ri])[2],
                    n_eps=[None] * len(_STUDY_NS), lo=[None] * len(_STUDY_NS), hi=[None] * len(_STUDY_NS)))
                cell['n_eps'][ni] = N * point; cell['lo'][ni] = N * point - half; cell['hi'][ni] = N * point + half
    return _STUDY


@check('V6.4', 'the rate constant: N eps* converges to c_r(p)', 'prop:rate', slow=True)
def v6_4():
    """Regime (a) of prop:rate at large N, on two sources: V6.6's p, whose order flips die out
    only near N = 1000, and a well-separated one (the shared study, _eps_star_study).

    Pass (fixed before the run), for every source and member r in {1, 2, inf}:
      * |N eps* / c_r(p) - 1| <= 0.15 at N = 8192, and
      * that relative error is at most the error at N = 1024 plus 0.05 (the gap does not grow).
    """
    st = _eps_star_study()
    i0, i1 = _STUDY_NS.index(1024), len(_STUDY_NS)
    ok = True; details = []
    for (si, r), v in st.items():
        errs = [abs(x / v['c'] - 1) for x in v['n_eps'][i0:i1]]
        good = errs[-1] <= 0.15 and errs[-1] <= errs[0] + 0.05
        ok &= good
        details.append(f'source {si + 1}, r={r:>3}: c_r = {v["c"]:.3f};  N eps* at N = {_STUDY_NS[i0:]}: '
                       + '; '.join(f'{x:.2f} [{lo:.2f}, {hi:.2f}]' for x, lo, hi in
                                   zip(v['n_eps'][i0:], v['lo'][i0:], v['hi'][i0:]))
                       + f'   rel. error {errs[0]:.2f} -> {errs[-1]:.2f}  ' + ('ok' if good else 'FAILS'))
    return Result(
        passed=ok,
        summary='N eps* at N = 8192 against c_r(p): '
                + ', '.join(f"{v['n_eps'][-1] / v['c'] - 1:+.1%}" for v in st.values()),
        details=['90% intervals from 8 independent batches of 500 draws'] + details,
        metrics=dict(table={k: dict(c=v['c'], n_eps=v['n_eps']) for k, v in st.items()}))


@check('V6.9', 'the local regime: near u the risk-optimal radius is N^{-1/2}', 'prop:rate', slow=True)
def v6_9():
    """Regime (b): p_N = u + h d / sqrt(N) with a fixed direction d (sum zero).  The fixed-p
    constant blows up as p -> u (at interior r like 1/|p - u|), so eps* should decay like
    N^{-1/2} here, the concentration scale.

    Pass (fixed before the first run): for every member r in {1, 2, inf}, the fitted slope of
    log eps* against log N lies within 0.15 of -1/2.

    h must put the optimum inside the path.  Oracle uniform shrinkage has optimal fraction
    (n-1) / (n h^2 sum d^2) in this regime; if that exceeds 1, the best move is to shrink all the
    way, every radius past p_hat's saturation has the same risk, and eps* is not identified.  The
    first run used h = 1 (fraction ~7) and failed for exactly that reason; h = 6 gives ~0.2.
    """
    n = 4; u = np.ones(n) / n
    d = np.array([.25, .05, -.1, -.2]); d -= d.mean()
    h = 6.0
    Ns = [64, 128, 256, 512, 1024, 2048, 4096]
    grid = np.geomspace(1e-5, 0.6, 50)
    lg = np.log(grid); logN = np.log(Ns)
    draws = 1000
    rng = np.random.default_rng(76); brng = np.random.default_rng(760)
    ok = True; details = []; slopes = {}
    for r in [1, 2, np.inf]:
        stars, bs_rows = [], []
        for N in Ns:
            p = u + h * d / np.sqrt(N)
            L = _risk_matrix(p, N, r, grid, draws, rng)
            pt = float(np.exp(_refined_log_argmin(lg, L.mean(0))))
            stars.append(pt)
            bs_rows.append([float(np.exp(_refined_log_argmin(lg, L[brng.integers(0, draws, draws)].mean(0))))
                            for _ in range(300)])
        sl = _slope(logN, np.log(stars))
        bs = np.array(bs_rows)
        bsl = [_slope(logN, np.log(bs[:, b])) for b in range(bs.shape[1])]
        ci = tuple(float(q) for q in np.quantile(bsl, [.05, .95]))
        good = abs(sl + 0.5) <= 0.15
        ok &= good
        slopes[str(r)] = dict(slope=sl, ci=ci, sqrtN_eps=[float(np.sqrt(N) * s) for N, s in zip(Ns, stars)])
        details.append(f'r={str(r):>3}: slope {sl:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}];  sqrt(N) eps* = '
                       + ' '.join(f'{np.sqrt(N) * s:.3f}' for N, s in zip(Ns, stars))
                       + ('   ok' if good else '   FAILS'))
    return Result(
        passed=ok,
        summary='slopes ' + ', '.join(f"{v['slope']:+.2f}" for v in slopes.values()) + ' (r = 1, 2, inf; want -0.5)',
        details=details, metrics=dict(slopes=slopes))


def _derivative_draws(p, N, r, draws, rng, h=1e-6):
    """Per-draw first and second eps-derivatives of the redundancy at eps = 0.

    With v = -log p_hat and g = g(p_hat) the subgradient there, the path gives x'(0) = g and
    x''(0) = -2 H diag(1/p_hat) g (H the Hessian of D_r at v; 0 at the endpoints), so
        L'(0)  = -sum p_i g_i / p_hat_i = sum eta_i g_i / p_hat_i        (sum g = 0),
        L''(0) = sum p_i g_i^2 / p_hat_i^2 - sum p_i x''_i / p_hat_i.
    a_N = -N E[L'(0) - X] with the control variate X = sum eta_i g_i(p) / p_i (mean zero), and
    2 b_N = E[L''(0)].  Draws outside the good event {min p_hat >= min p / 2} are skipped and
    counted (there KL(p || p_hat) is not defined).
    """
    n = len(p); g0 = subgrad(-np.log(p), r)
    cnts = rng.multinomial(N, p, size=draws) / N
    A, B = [], []
    for ph in cnts:
        if ph.min() < p.min() / 2:
            continue
        v = -np.log(ph); g = subgrad(v, r); eta = ph - p
        A.append(-N * (float(np.sum(eta * g / ph)) - float(np.sum(eta * g0 / p))))
        if 1 < r < np.inf:
            H = np.column_stack([(subgrad(v + h * np.eye(n)[j], r) - subgrad(v - h * np.eye(n)[j], r)) / (2 * h)
                                 for j in range(n)])
            x2 = -2 * (0.5 * (H + H.T)) @ (g / ph)
        else:
            x2 = np.zeros(n)
        B.append(float(np.sum(p * g ** 2 / ph ** 2) - np.sum(p * x2 / ph)))
    return np.array(A), np.array(B), 1 - len(A) / draws


@check('V6.3', 'the rate constant from the derivatives at eps = 0', 'prop:rate', slow=True)
def v6_3():
    """prop:rate tested at its own level: estimate the linear and quadratic coefficients of the
    risk at eps = 0 directly, a_N and 2 b_N, and compare c_N = a_N / (2 b_N) with c_r(p).  No
    argmin of a flat curve is involved, so this is sharper than V6.4.

    Pass (fixed before the run): at N = 8192, for both sources and r in {1, 2, inf},
    |c_N / c_r - 1| <= 0.03 and the 90% bootstrap interval of c_N contains c_r.
    Reported: a_N / a at every N, which shows the endpoint order-flip term dying out.
    """
    sources = [np.array([.5, .25, .15, .10]), np.array([.45, .30, .17, .08])]
    Ns = [256, 512, 1024, 2048, 4096, 8192]
    draws = 40000
    rng = np.random.default_rng(77); brng = np.random.default_rng(770)
    ok = True; details = []; table = {}
    for si, p in enumerate(sources):
        for r in [1, 2, np.inf]:
            a_inf, twob_inf, c = rate_constant(p, r)
            row, last = [], None
            for N in Ns:
                A, B, skipped = _derivative_draws(p, N, r, draws, rng)
                cN = A.mean() / B.mean()
                bs = []
                for _ in range(400):
                    idx = brng.integers(0, len(A), len(A))
                    bs.append(A[idx].mean() / B[idx].mean())
                lo, hi = np.quantile(bs, [.05, .95])
                row.append(f'N={N}: c_N={cN:.3f} a_N/a={A.mean() / a_inf:.3f}')
                last = (cN, lo, hi, skipped)
            cN, lo, hi, skipped = last
            good = abs(cN / c - 1) <= 0.03 and lo <= c <= hi
            ok &= good
            table[(si, str(r))] = dict(c=c, c_N=cN, ci=(float(lo), float(hi)))
            details.append(f'source {si + 1}, r={str(r):>3}: c_r = {c:.3f}; at N = 8192 c_N = {cN:.3f} '
                           f'[{lo:.3f}, {hi:.3f}]  ' + ('ok' if good else 'FAILS'))
            details.append('    ' + ';  '.join(row))
    return Result(
        passed=ok,
        summary='c_N = a_N / 2b_N against c_r(p) at N = 8192: '
                + ', '.join(f"{v['c_N'] / v['c'] - 1:+.3f}" for v in table.values()),
        details=details, metrics=dict(table=table))


def shrinkage_constant(p):
    """Uniform shrinkage x = p_hat + beta (u - p_hat): the same second-order expansion as prop:rate
    gives beta* = A / (2 B N) with A = mean(1/p) - 1 (quadratic and cubic terms together) and
    2B = sum (p_i - 1/n)^2 / p_i.  Returns (A, 2B, N beta*)."""
    p = np.asarray(p, float); n = len(p)
    A = float(np.mean(1 / p) - 1); twoB = float(np.sum((p - 1 / n) ** 2 / p))
    return A, twoB, A / twoB


def _shrinkage_draws(p, N, draws, rng):
    """Per-draw derivatives at beta = 0 for x = p_hat + beta (u - p_hat), on the good event:
    L'(0) = sum eta_i (u_i - p_hat_i) / p_hat_i (control variate sum eta_i (u_i - p_i) / p_i
    subtracted), L''(0) = sum p_i (u_i - p_hat_i)^2 / p_hat_i^2."""
    n = len(p); u = np.ones(n) / n
    cnts = rng.multinomial(N, p, size=draws) / N
    A, B = [], []
    for ph in cnts:
        if ph.min() < p.min() / 2:
            continue
        eta = ph - p
        A.append(-N * (float(np.sum(eta * (u - ph) / ph)) - float(np.sum(eta * (u - p) / p))))
        B.append(float(np.sum(p * (u - ph) ** 2 / ph ** 2)))
    return np.array(A), np.array(B)


@check('V6.8', 'uniform shrinkage: the constant, and the reductions against the family', 'obs:geom', slow=True)
def v6_8():
    """(a) The corrected shrinkage constant N beta* = A / 2B against a derivative estimate at
    N = 8192 on three sources.  (b) The optimal risk reductions, a^2 / (4b) per N^2 for the r = 2
    member and A^2 / (4B) for uniform shrinkage, from the formulas and from the estimates.

    Pass (fixed before the run): (a) every estimate within 3% of the formula; (b) the estimated
    reductions within 5% of the formulas, and the winner the same by both: the r = 2 member on
    p = (0.7, 0.11, 0.10, 0.09), uniform shrinkage on p = (0.5, 0.25, 0.15, 0.10).
    """
    sources = [np.array([.5, .25, .15, .10]), np.array([.45, .30, .17, .08]), np.array([.7, .11, .10, .09])]
    N, draws = 8192, 40000
    rng = np.random.default_rng(78)
    ok = True; details = []; metrics = {}
    for p in sources:
        A0, twoB0, nb0 = shrinkage_constant(p)
        A, B = _shrinkage_draws(p, N, draws, rng)
        nb = A.mean() / B.mean()
        red_s_f = A0 ** 2 / (2 * twoB0); red_s_e = A.mean() ** 2 / (2 * B.mean())
        a0, tb0, _ = rate_constant(p, 2)
        Af, Bf, _ = _derivative_draws(p, N, 2, draws, rng)
        red_f_f = a0 ** 2 / (2 * tb0); red_f_e = Af.mean() ** 2 / (2 * Bf.mean())
        good_a = abs(nb / nb0 - 1) <= 0.03
        good_b = (abs(red_s_e / red_s_f - 1) <= 0.05 and abs(red_f_e / red_f_f - 1) <= 0.05
                  and (red_f_f > red_s_f) == (red_f_e > red_s_e))
        ok &= good_a and good_b
        metrics[str(np.round(p, 2))] = dict(N_beta=nb0, reduction_shrink=red_s_f, reduction_r2=red_f_f)
        details.append(f'p = {np.round(p, 2)}: N beta* formula {nb0:.2f}, estimate {nb:.2f};  reduction x N^2: '
                       f'shrinkage {red_s_f:.1f} (est. {red_s_e:.1f}), r = 2 {red_f_f:.1f} (est. {red_f_e:.1f})  '
                       + ('ok' if good_a and good_b else 'FAILS'))
    want = [('0.7' in k) == (v['reduction_r2'] > v['reduction_shrink']) for k, v in metrics.items()]
    ok &= all(want)
    return Result(
        passed=ok,
        summary='corrected shrinkage constant confirmed; r = 2 beats shrinkage on (0.7, 0.11, 0.10, 0.09) only',
        details=details, metrics=metrics)


@check('V6.5', 'coverage: the risk-optimal ball stops containing the truth', 'prop:rate')
def v6_5():
    """How often the ball around p_hat contains p, at the risk-optimal radius c_r(p)/N of
    prop:rate and at the certificate radius eps_cert(N) = E||p_hat - p||_{r*}, on V6.6's source.

    Pass (fixed before the run), for every member r in {1, 2, inf}: the coverage at c_r(p)/N
    falls from the smallest to the largest N and never rises by more than 0.01 between
    consecutive N (Monte Carlo noise), and it is below the coverage at eps_cert(N) at every N.
    """
    p = np.array([.5, .25, .15, .10])
    Ns = [16, 32, 64, 128, 256, 512, 1024]
    draws = 20000
    rng = np.random.default_rng(79)
    ok = True; details = []; table = {}
    for r in [1, 2, np.inf]:
        rs = dual(r); c = rate_constant(p, r)[2]
        star, cert = [], []
        for N in Ns:
            norms = np.array([nrm(ph - p, rs) for ph in rng.multinomial(N, p, size=draws) / N])
            star.append(float(np.mean(norms <= c / N)))
            cert.append(float(np.mean(norms <= norms.mean())))
        good = (star[-1] < star[0] and all(b <= a + 0.01 for a, b in zip(star, star[1:]))
                and all(s < t for s, t in zip(star, cert)))
        ok &= good
        table[str(r)] = dict(star=star, cert=cert)
        details.append(f'r={str(r):>3}: coverage at c_r/N ' + ' '.join(f'{v:.3f}' for v in star)
                       + ';  at eps_cert ' + ' '.join(f'{v:.2f}' for v in cert) + ('  ok' if good else '  FAILS'))
    return Result(
        passed=ok,
        summary=f'N = {Ns[0]}..{Ns[-1]}: coverage at c_r/N falls to '
                + ', '.join(f"{v['star'][-1]:.3f}" for v in table.values())
                + ' (r = 1, 2, inf); at eps_cert ' + ', '.join(f"{v['cert'][-1]:.2f}" for v in table.values()),
        details=details, metrics=dict(table=table, Ns=Ns))


def _rate_constant_closed(p, r):
    """c_r(p) from explicit formulas, independent of rate_constant's numerical Hessian:
    g at r = inf is -1/2 at the largest p_i and +1/2 at the smallest; at r = 1 (even n) -1 on
    the larger half and +1 on the smaller; at r = 2, g = Pv/||Pv|| and H = (P - gg')/||Pv||."""
    p = np.asarray(p, float); n = len(p); v = -np.log(p)
    M = np.diag(1 / p) @ (np.diag(p) - np.outer(p, p)) @ np.diag(1 / p)
    H = np.zeros((n, n)); g = np.zeros(n); o = np.argsort(p)
    if np.isinf(r):
        g[o[-1]], g[o[0]] = -.5, .5
    elif r == 1:
        g[o[n // 2:]], g[o[:n // 2]] = -1., 1.
    else:
        P = np.eye(n) - np.ones((n, n)) / n; w = P @ v; g = w / np.linalg.norm(w)
        H = (P - np.outer(g, g)) / np.linalg.norm(w)
    return float((np.sum(g / p) + np.trace(H @ M)) / np.sum(g ** 2 / p))


@check('V6.2', 'the rate constants c_r(p), two ways', 'prop:rate')
def v6_2():
    """c_r(p) as used by V6.3, V6.4, V6.6 and V6.9 (rate_constant, numerical Hessian) against the explicit
    formulas, on the sources of V6.4/V6.3 and of V6.8.  Pass: relative agreement 1e-5."""
    sources = [np.array([.5, .25, .15, .10]), np.array([.45, .30, .17, .08]), np.array([.7, .11, .10, .09])]
    worst = 0.0; details = []; table = {}
    for p in sources:
        row = {}
        for r in [1, 2, np.inf]:
            c, cc = rate_constant(p, r)[2], _rate_constant_closed(p, r)
            worst = max(worst, abs(c / cc - 1)); row[str(r)] = c
        table[str(np.round(p, 2))] = row
        details.append(f'p = {np.round(p, 2)}: ' + ', '.join(f'c_{k} = {v:.3f}' for k, v in row.items()))
    return Result(
        passed=worst < 1e-5,
        summary=f'max relative difference {worst:.1e} over three sources and three members',
        details=details, metrics=dict(table=table))


@check('V6.7', 'N eps*(N) approaches c_r(p), N = 16 to 8192', 'prop:rate', slow=True)
def v6_7():
    """The data behind the convergence figure: the shared study (_eps_star_study) over the
    whole range N = 16 .. 8192, with 90% batch intervals.

    Pass (V6.4's tolerance, fixed before the run): at N = 8192, |N eps* / c_r(p) - 1| <= 0.15
    for every source and member.
    """
    st = _eps_star_study()
    ok = True; details = []
    for (si, r), v in st.items():
        err = abs(v['n_eps'][-1] / v['c'] - 1)
        ok &= err <= 0.15
        details.append(f'source {si + 1}, r={r:>3}: c_r = {v["c"]:.3f};  N eps* = '
                       + ' '.join(f'{x:.2f}' for x in v['n_eps']) + f';  rel. error at 8192 {err:.1%}'
                       + ('  ok' if err <= 0.15 else '  FAILS'))
    return Result(
        passed=ok,
        summary='N eps* at N = 8192 against c_r(p): '
                + ', '.join(f"{v['n_eps'][-1] / v['c'] - 1:+.1%}" for v in st.values()),
        details=details, metrics=dict(Ns=_STUDY_NS),
        data=dict(Ns=_STUDY_NS, curves=dict(st), sources=[x.tolist() for x in _STUDY_SOURCES]))


@check('V6.11', 'the Hessian of D_r is bounded at a single centre coordinate', 'lem:hessian')
def v6_11():
    """lem:hessian: for 1 < r < 2 the Hessian of D_r stays bounded as a SINGLE coordinate reaches
    the centre, and is unbounded when TWO coordinates meet it together.  At r in {1.2, 1.5, 1.8}:
      (a) the analytic grounded Laplacian r(r-1)(diag(a) - a a^T / 1^T a), a_i = |v_i - c|^{r-2},
          against a central finite difference of grad D_r^r at a safe step;
      (b) convergence: as one coordinate of v = (0, .6, 1.3, 2.1) approaches the centre, from
          t = 1e-4 to 1e-12, the ANALYTIC max|H(D_r^r)| changes by < 1%.  (The finite-difference
          Hessian blows up spuriously below t ~ 1e-5 at r = 1.2 because the centre equation is
          ill-conditioned, so convergence is read from the analytic form, not from differences.)
      (c) negative control: with two coordinates on the centre, v = (-1, 0, 0, 1), the second
          difference of D_r along the splitting direction grows as the step shrinks (unbounded),
          exactly the case the lemma excludes.
    Pass (fixed before the run): (a) < 1e-5; (b) relative change < 1%; (c) grows > 5x from
    t = 1e-2 to 1e-5.
    """
    base = np.array([0.0, 0.6, 1.3, 2.1]); n = len(base); I = np.eye(n)

    def grad_power(v, r):
        c = float(centre(v, r)); return r * np.abs(v - c) ** (r - 1) * np.sign(v - c)

    def jac(f, v, h):
        return np.column_stack([(f(v + h * I[j]) - f(v - h * I[j])) / (2 * h) for j in range(n)])

    def hess_power_an(v, r):
        c = float(centre(v, r)); a = np.abs(v - c) ** (r - 2)
        return r * (r - 1) * (np.diag(a) - np.outer(a, a) / a.sum())

    def second_diff(v, r, direction, t):
        return (D(v + t * direction, r) - 2 * D(v, r) + D(v - t * direction, r)) / t ** 2

    form_err = max_rel = 0.0
    details = []
    for r in (1.2, 1.5, 1.8):
        c = float(centre(base, r))
        vp = base.copy(); vp[0] = c + 1e-2            # safe step for finite differences
        form_err = max(form_err, float(np.max(np.abs(
            hess_power_an(vp, r) - jac(lambda w: grad_power(w, r), vp, 1e-4)))))
        vals = []
        for e in (4, 6, 8, 10, 12):
            v = base.copy(); v[0] = c + 10.0 ** (-e)
            vals.append(float(np.max(np.abs(hess_power_an(v, r)))))
        rel = abs(vals[-1] - vals[0]) / abs(vals[0]); max_rel = max(max_rel, rel)
        details.append(f'r={r}: single coordinate, analytic max|H(D_r^r)| -> {vals[-1]:.4f}, '
                       f'rel change {rel:.2%}')
    v2 = np.array([-1.0, 0.0, 0.0, 1.0]); d2 = np.array([0.0, 1.0, -1.0, 0.0])
    neg = [second_diff(v2, 1.5, d2, 10.0 ** (-e)) for e in (2, 5)]
    grow = neg[1] / neg[0]
    details.append(f'two coordinates on the centre (r=1.5): second difference {neg[0]:.0f} -> '
                   f'{neg[1]:.0f} as step shrinks, growing {grow:.0f}x (unbounded, as excluded)')
    passed = form_err < 1e-5 and max_rel < 0.01 and grow > 5
    return Result(
        passed=passed,
        summary=(f'grounded Laplacian vs finite diff {form_err:.1e}; single-coordinate '
                 f'max|H(D_r^r)| converges (rel change <= {max_rel:.2%}); two tied coordinates '
                 f'blow up ({grow:.0f}x), the case the lemma excludes'),
        details=details,
        metrics=dict(form_err=form_err, max_rel=max_rel, neg_growth=grow))


@check('V6.14', 'the eps^3 remainder bounded generically; D_r regularity fails on the centre',
       'lem:remainder')
def v6_14():
    """lem:remainder.  The lemma needs D_r in C^4 near the path (so the implicit function theorem
    gives a C^3 path and a bounded 3rd eps-derivative of K).  Two deterministic parts, at eta = 0
    (p_hat = p), so K(0) = K'(0) = 0 and K(eps) = KL(p || x*(eps)) = c2 eps^2 + rho(eps) on the
    exact solver path.

    (1) generic O(eps^3): on a generic source p = (0.5, 0.25, 0.15, 0.10), every coordinate of
        -log p off the centre so D_r is analytic there, rho(eps) / eps^3 stays bounded (it converges
        to the cubic coefficient) as eps -> 0, at r in {1.5, 2, 2.5, 4}.  c2 is read as the limit
        K(eps)/eps^2 (Richardson from the two smallest eps).  This is the remainder the lemma proves.
    (2) on-centre breakdown of the C^4 regularity (why asm:rate keeps the on-centre case).  The k-th
        derivative of D_r^r carries |v_i - c|^{r-k}, so as one coordinate meets the centre:
          1 < r < 3 (r = 1.5, 2.5): the THIRD derivative diverges -- max|d(Hessian)/dv0| grows;
          3 <= r < 4 (r = 3.5): the third derivative stays bounded but the FOURTH diverges --
            max|d^2(Hessian)/dv0^2| grows while max|d(Hessian)/dv0| does not.  This documents why the
            residual extends past r = 3: C^3 is not enough, C^4 is needed.
        (The grounded Hessian itself stays bounded on the centre, V6.11; here its derivatives blow
        up, one and two orders up.)

    Pass (fixed before the run):
      (1) rho/eps^3 finite with small/large-eps ratio in [0.5, 2.5] at every tested r;
      (2) at r in {1.5, 2.5}, the 3rd-derivative proxy is bounded at t ~ 0.2 (< 10) and grows > 10x
          by t = 1e-4; at r = 3.5, the 3rd-derivative proxy does not grow (< 2x) while the
          4th-derivative proxy grows > 10x.
    """
    # part (1): the generic eps^3 remainder, eta = 0
    p = np.array([0.5, 0.25, 0.15, 0.10])
    es = np.geomspace(1e-3, 5e-2, 10)

    def K_of(pp, r):
        return np.array([kl(pp, solve(pp, float(e), r)) for e in es])

    def c2_limit(pp, r):                                # c2 = lim K(eps)/eps^2, linear Richardson
        e2 = np.array([5e-4, 1e-3])
        q = np.array([kl(pp, solve(pp, float(e), r)) for e in e2]) / e2 ** 2
        return 2 * q[0] - q[1]

    ratios = {}; p1_ok = True; details = []
    for r in (1.5, 2.0, 2.5, 4.0):
        v = -np.log(p); c = float(centre(v, r)); mind = float(np.min(np.abs(v - c)))
        prof = (K_of(p, r) - c2_limit(p, r) * es ** 2) / es ** 3
        rat = float(prof[0] / prof[-1])
        ratios[r] = rat
        ok = bool(np.all(np.isfinite(prof)) and 0.5 <= rat <= 2.5)
        p1_ok &= ok
        details.append(f'(1) r={r}: min|coord-centre|={mind:.2f}, rho/eps^3 in '
                       f'[{prof.min():.1f}, {prof.max():.1f}], small/large ratio {rat:.2f} '
                       f'{"ok" if ok else "FAILS"}')

    # part (2): on-centre breakdown of the C^4 regularity the lemma needs (path C^3)
    base = np.array([0.0, 0.6, 1.3, 2.1])

    def hess_power_an(v, r):
        c = float(centre(v, r)); a = np.abs(v - c) ** (r - 2)
        return r * (r - 1) * (np.diag(a) - np.outer(a, a) / a.sum())

    def dH_mag(v, r, h):                                 # ~ third derivative of D_r^r
        vp = v.copy(); vp[0] += h; vm = v.copy(); vm[0] -= h
        return float(np.max(np.abs((hess_power_an(vp, r) - hess_power_an(vm, r)) / (2 * h))))

    def d2H_mag(v, r, h):                                # ~ fourth derivative of D_r^r
        vp = v.copy(); vp[0] += h; vm = v.copy(); vm[0] -= h
        return float(np.max(np.abs(
            (hess_power_an(vp, r) - 2 * hess_power_an(v, r) + hess_power_an(vm, r)) / h ** 2)))

    def at_dist(r, t):
        v = base.copy()
        for _ in range(60):                             # fixed point: coord0 lands at distance t
            c = float(centre(v, r)); v[0] = c + t
        return v

    p2_ok = True; grow = {}
    for r in (1.5, 2.5):                                 # 1 < r < 3: the third derivative diverges
        gen = dH_mag(at_dist(r, 0.2), r, 0.2 / 10.0)
        onc = dH_mag(at_dist(r, 1e-4), r, 1e-4 / 10.0)
        g = onc / gen; grow[r] = g
        ok = bool(gen < 10.0 and g > 10.0)
        p2_ok &= ok
        details.append(f'(2) r={r}: max|dH/dv0| (3rd deriv) generic(t~0.2)={gen:.1f}, '
                       f'on-centre(t~1e-4)={onc:.0f}, growth {g:.0f}x {"ok" if ok else "FAILS"}')

    r = 3.5                                              # 3 <= r < 4: 3rd deriv bounded, 4th diverges
    d3g = dH_mag(at_dist(r, 0.2), r, 0.2 / 10.0); d3o = dH_mag(at_dist(r, 1e-4), r, 1e-4 / 10.0)
    d4g = d2H_mag(at_dist(r, 0.2), r, 0.2 / 10.0); d4o = d2H_mag(at_dist(r, 1e-4), r, 1e-4 / 10.0)
    g3 = d3o / d3g; g4 = d4o / d4g; grow[r] = g4
    ok = bool(g3 < 2.0 and g4 > 10.0)
    p2_ok &= ok
    details.append(f'(2) r={r}: 3rd deriv bounded on-centre (growth {g3:.2f}x), '
                   f'4th deriv diverges (growth {g4:.0f}x) {"ok" if ok else "FAILS"}')

    return Result(
        passed=p1_ok and p2_ok,
        summary=('generic rho/eps^3 bounded (small/large ratios '
                 + ', '.join(f'r{k}:{v:.2f}' for k, v in ratios.items())
                 + '); on the centre D_r loses C^4 (derivative growth '
                 + ', '.join(f'r{k}:{v:.0f}x' for k, v in grow.items())
                 + '; at r3.5 the 4th derivative, 3rd stays bounded)'),
        details=details,
        metrics=dict(ratios=ratios, growth=grow))


@check('V6.12', 'the crossover sample size', 'prop:rate')
def v6_12():
    """N_x = (c_r(p) / c_cert)^2 with c_cert = E||Z||_{r*}, Z ~ N(0, Sigma): the sample size where
    the certificate radius E||p_hat - p||_{r*} ~ c_cert N^{-1/2} meets the risk-optimal radius
    c_r/N.  Below N_x the certificate radius is the smaller one; above it the certificate
    over-shrinks.  On both test sources N_x <= 16 at every member, so the over-shrinking regime
    covers all practical N.
    Checks (fixed before the run): (a) c_cert grounds the sampling: sqrt(N) E||p_hat - p|| -> c_cert,
    agreeing within 3% at N = 4096; (b) every N_x < 20; (c) the asymptotic certificate radius
    c_cert/sqrt(N) is larger than c_r/N for N = 64 (> all N_x) and, at r = 2 where N_x > 2, smaller
    for N = 4 (< N_x).
    """
    sources = [np.array([.5, .25, .15, .10]), np.array([.45, .30, .17, .08])]
    rng = np.random.default_rng(612)
    worst_cc = 0.0; maxNx = 0.0; order_ok = True; details = []
    for si, p in enumerate(sources):
        n = len(p); Sig = np.diag(p) - np.outer(p, p)
        Z = rng.multivariate_normal(np.zeros(n), Sig, size=200000)
        for r in (1, 2, np.inf):
            pq = dual(r)
            c_cert = float(np.mean([nrm(z, pq) for z in Z]))
            c_r = rate_constant(p, r)[2]
            Nx = (c_r / c_cert) ** 2; maxNx = max(maxNx, Nx)
            # (a) ground c_cert on real multinomial sampling at N = 4096
            N = 4096; draws = rng.multinomial(N, p, size=4000) / N
            emp = np.sqrt(N) * float(np.mean([nrm(d - p, pq) for d in draws]))
            worst_cc = max(worst_cc, abs(emp - c_cert) / c_cert)
            # (c) ordering of the asymptotic radii
            big = (c_cert / np.sqrt(64)) > (c_r / 64)                 # certificate larger above N_x
            order_ok &= big
            if r == 2:
                order_ok &= (c_cert / np.sqrt(4)) < (c_r / 4)         # certificate smaller below N_x
            details.append(f'source {si+1}, r={str(r):>3}: c_r={c_r:.3f}, c_cert={c_cert:.3f}, '
                           f'N_x={Nx:.1f}, sqrt(N)E||eta||@4096={emp:.3f}')
    passed = worst_cc < 0.03 and maxNx < 20 and order_ok
    return Result(
        passed=passed,
        summary=f'N_x <= {maxNx:.1f} at every member (both sources); c_cert grounds sampling to '
                f'{worst_cc:.1%}; certificate radius larger than risk-optimal above N_x',
        details=details, metrics=dict(maxNx=maxNx, cc_err=worst_cc))


@check('V6.13', 'the plug-in rate rule does not beat KT', 'obs:geom', slow=True)
def v6_13():
    """A concrete estimator from the rate: eps_hat = c_r(p_hat)/N, the plug-in of the
    risk-optimal constant.  c_r needs full support, so it is evaluated at the KT-smoothed
    estimate; the robust code is solved from the raw p_hat (the barrier handles zeros).  On the
    18-cell coding design this is compared to KT and to Hausser-Strimmer, paired, with the same
    95% bootstrap protocol as V6.1.
    Reported whatever the outcome; the thesis predicts it does not beat KT.  Pass (fixed before
    the run): significant wins over KT in at most half the cells (<= 9/18) for every member.
    """
    from ..core import is_degenerate
    ns = [4, 8]; Ns = [8, 32, 128]; alphas = [0.3, 1.0, 3.0]; rs = [1, 2, np.inf]; draws = 120
    rng = np.random.default_rng(73); brng = np.random.default_rng(730); B = 1500

    def c_r_hat(ps, r):
        if is_degenerate(-np.log(ps)):
            return 0.0
        try:
            return rate_constant(ps, r)[2]
        except ZeroDivisionError:
            return 0.0

    sig_kt = {r: 0 for r in rs}; loss_kt = {r: 0 for r in rs}; sig_hs = {r: 0 for r in rs}
    cells = 0; hs_inf = 0
    for n in ns:
        for a in alphas:
            for N in Ns:
                acc = {'kt': [], 'hs': []}
                for r in rs:
                    acc[r] = []
                for _ in range(draws):
                    p = rng.dirichlet(np.ones(n) * a); cnt = rng.multinomial(N, p); ph = cnt / N
                    ps = (cnt + 0.5) / (N + n * 0.5)
                    acc['kt'].append(kl(p, ps))
                    hs = kl(p, hausser_strimmer(cnt)); acc['hs'].append(hs); hs_inf += np.isinf(hs)
                    for r in rs:
                        eh = c_r_hat(ps, r) / N
                        acc[r].append(kl(p, solve(ph, eh, r) if eh > 0 else ph))
                A = {k: np.array(v) for k, v in acc.items()}; cells += 1
                idx = brng.integers(0, draws, (B, draws))
                for r in rs:
                    d = A[r] - A['kt']; lo, hi = np.quantile(d[idx].mean(1), [.025, .975])
                    sig_kt[r] += hi < 0; loss_kt[r] += lo > 0
                    dh = A[r] - A['hs']
                    if np.isfinite(dh).all():
                        _, hih = np.quantile(dh[idx].mean(1), [.025, .975]); sig_hs[r] += hih < 0
    passed = all(sig_kt[r] <= cells // 2 for r in rs)
    return Result(
        passed=passed,
        summary='plug-in eps=c_r(p_hat)/N vs KT: significant wins '
                + ', '.join(f'{sig_kt[r]}/{cells}' for r in rs)
                + ' (losses ' + ', '.join(f'{loss_kt[r]}' for r in rs) + ') at r=1,2,inf',
        details=[f'r={str(r):>3}: wins vs KT {sig_kt[r]}/{cells}, losses {loss_kt[r]}/{cells}, '
                 f'wins vs Hausser-Strimmer (finite) {sig_hs[r]}/{cells}' for r in rs]
                + [f'Hausser-Strimmer infinite in {hs_inf} draws'],
        metrics=dict(sig_kt=sig_kt, loss_kt=loss_kt, sig_hs=sig_hs))
