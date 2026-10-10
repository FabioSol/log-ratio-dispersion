"""Chapter 5 checks: the budget is valid, and tight exactly when."""
import numpy as np
from scipy.optimize import minimize

from . import SOLVER_TOL, TIE_MARGIN, Result, check
from ..core import D, subgrad, dual, nrm, centre, face_feasible
from ..prob import entropy
from ..solver import solve
from ..worst_case import LPFailure, inner_p2, worst_case, worst_case_LP


def _crosstab():
    return dict(tt=0, tf=0, ft=0, ff=0)


def _tally(c, exact, feasible):
    """tt = tight & feasible, ff = loose & infeasible; tf / ft are off-diagonal."""
    c['tt' if (exact and feasible) else 'tf' if exact else 'ft' if feasible else 'ff'] += 1


@check('V5.1', 'the bound is never below the truth', 'lem:upperbound')
def v5_1():
    """<ph,f> + eps D_r(f) >= W(f), against exact ground truth at r* in {1, 2, inf}."""
    rng = np.random.default_rng(61); below = tot = lpfail = 0; gap = 0.0
    for _ in range(500):
        n = int(rng.choice([3, 4, 5, 8])); ph = rng.dirichlet(np.ones(n))
        f = rng.normal(size=n) * 2; eps = rng.uniform(.005, .5)
        for p in [1, np.inf, 2]:
            try:
                truth = worst_case(ph, f, eps, p)
            except LPFailure:
                lpfail += 1; continue
            cf = ph @ f + eps * D(f, dual(p)); tot += 1
            below += (cf < truth - 1e-8); gap = max(gap, cf - truth)
    return Result(
        passed=below == 0 and lpfail == 0,
        summary=(f'{tot} instances across r* in 1,2,inf: closed form below truth = {below}, '
                 f'solver failures = {lpfail}, max gap = {gap:.3f} nats'),
        metrics=dict(instances=tot, below=below, failures=lpfail, max_gap=gap))


@check('V5.3', 'tightness criterion, generic regime', 'prop:tightness')
def v5_3():
    """On generic data the subgradient is a singleton, so the single-vertex check is exact."""
    rng = np.random.default_rng(62); out = {}
    for p in [1, np.inf]:
        r = dual(p); c = _crosstab()
        for _ in range(600):
            n = int(rng.choice([3, 4, 5, 8])); ph = rng.dirichlet(np.ones(n))
            f = rng.normal(size=n) * 2; eps = rng.uniform(.005, .5)
            ex = abs(worst_case_LP(ph, f, eps, p) - (ph @ f + eps * D(f, r))) < 1e-8
            fe = (ph + eps * subgrad(f, r)).min() >= -1e-12
            _tally(c, ex, fe)
        out[p] = c
    off = sum(c['tf'] + c['ft'] for c in out.values())
    return Result(
        passed=off == 0,
        summary=f'off-diagonal (criterion disagrees with LP) = {off}',
        details=['tt = tight & feasible, ff = loose & infeasible, tf/ft = off-diagonal (must be 0)']
                + [f'r*={p}:  {c}' for p, c in out.items()],
        metrics=dict(crosstab={str(p): c for p, c in out.items()}))


@check('V5.2', 'tightness, tied regime: the criterion is existential', 'prop:tightness')
def v5_2():
    """With ties the face is set-valued: the full-face test agrees with the LP, the
    canonical single vertex does not."""
    rng = np.random.default_rng(63)
    c = _crosstab(); canon_misses = 0
    for _ in range(400):
        n = int(rng.choice([4, 5, 6]))
        f = rng.integers(0, 3, n).astype(float)
        ph = rng.dirichlet(np.ones(n)); ph[rng.integers(n)] = 0.; ph /= ph.sum()
        eps = rng.uniform(.05, .6)
        ex = abs(worst_case_LP(ph, f, eps, 1) - (ph @ f + eps * D(f, np.inf))) < 1e-8
        fe = face_feasible(ph, f, eps, np.inf)
        cn = (ph + eps * subgrad(f, np.inf)).min() >= -1e-12
        _tally(c, ex, fe)
        canon_misses += (fe and not cn)
    # the hand-built counterexample
    ph = np.array([0., .5, .3, .2]); f = np.array([0., 0., 1., 1.]); eps = .4
    ce_cn = bool((ph + eps * subgrad(f, np.inf)).min() >= -1e-12)
    ce_fe = face_feasible(ph, f, eps, np.inf)
    lp = worst_case_LP(ph, f, eps, 1); cf = ph @ f + eps * D(f, np.inf)
    off = c['tf'] + c['ft']
    return Result(
        passed=off == 0 and canon_misses > 0 and ce_fe and not ce_cn and abs(lp - cf) < 1e-8,
        summary=f'face test exact (off-diagonal {off}); canonical vertex misses {canon_misses}/400',
        details=[f'face test vs LP: {c}   (off-diagonal must be 0)',
                 f'canonical single-vertex check missed {canon_misses} face-feasible instances',
                 'counterexample p=(0,.5,.3,.2), f=(0,0,1,1), eps=.4:',
                 f'  canonical vertex feasible? {ce_cn};  face feasible? {ce_fe};  '
                 f'LP {lp:.6f} vs closed form {cf:.6f}'],
        metrics=dict(crosstab=c, canon_misses=canon_misses))


@check('V5.4', 'tightness at r* = 2 vs certified KKT', 'prop:tightness')
def v5_4():
    rng = np.random.default_rng(64); dis = n_ok = 0
    for _ in range(400):
        n = int(rng.choice([3, 4, 5, 8])); ph = rng.dirichlet(np.ones(n))
        f = rng.normal(size=n) * 2; eps = rng.uniform(.01, .5)
        val, q, c = inner_p2(ph, f, eps)
        if not c['ok']:
            continue
        n_ok += 1
        ex = abs(val - (ph @ f + eps * D(f, 2))) < 1e-8
        fe = (ph + eps * subgrad(f, 2)).min() >= -1e-10
        dis += (ex != fe)
    return Result(
        passed=dis == 0 and n_ok == 400,
        summary=f'{n_ok}/400 certified instances, criterion-vs-truth disagreements {dis}',
        metrics=dict(certified=n_ok, disagreements=dis))


@check('V5.5', 'distance to degeneracy on synthetic logits', 'prop:tightness')
def v5_5():
    """Ties are not lurking arbitrarily close: the 1st-percentile distance to the
    tied set stays well above zero on synthetic logit draws."""
    rng = np.random.default_rng(65); out = {}
    for r in [1, np.inf]:
        d = []
        for _ in range(2000):
            n = int(rng.choice([4, 6, 10]))
            z = rng.normal(size=n) * rng.choice([.5, 2., 5.]); z[0] += 2.
            x = np.exp(z - z.max()); x /= x.sum(); f = -np.log(x)
            if np.isinf(r):
                s = np.sort(f); d.append(min(s[1] - s[0], s[-1] - s[-2]))
            else:
                d.append(float(np.min(np.abs(f - centre(f, 1)))))
        d = np.array(d); out[r] = (float(np.quantile(d, .01)), float(np.median(d)))
    q1 = min(v[0] for v in out.values())
    return Result(
        passed=q1 > TIE_MARGIN,
        summary=f'1st percentile distance >= {q1:.1e} (tie margin {TIE_MARGIN:.0e})',
        details=[f'r={str(r):>4}:  1st percentile distance {a:.1e},  median {m:.2e}'
                 for r, (a, m) in out.items()],
        metrics=dict(q01={str(r): v[0] for r, v in out.items()}))


@check('V5.6', 'one-hot tightness equals correct classification', 'prop:onehot')
def v5_6():
    """For ph = e_k the l1 certificate is tight iff argmax x = k and eps <= 2; at eps = 1
    the tightness rate therefore equals the accuracy."""
    rng = np.random.default_rng(66); dis = tot = 0; offs = []; acc = []; tight = []
    for offset in np.linspace(-1.0, 4.0, 11):
        hits = corr = 0
        for _ in range(60):
            n = int(rng.choice([4, 6, 10])); k = int(rng.integers(n))
            z = rng.normal(size=n); z[k] += offset
            x = np.exp(z - z.max()); x /= x.sum(); f = -np.log(x)
            ph = np.zeros(n); ph[k] = 1.0
            eps = float(rng.choice([0.2, 1.0, 1.9, 2.5]))
            ex = abs(worst_case_LP(ph, f, eps, 1) - (ph @ f + eps * D(f, np.inf))) < 1e-8
            pred = (int(np.argmax(x)) == k) and (eps <= 2.0 + 1e-12)
            dis += (ex != pred); tot += 1
            ex2 = abs(worst_case_LP(ph, f, 1.0, 1) - (ph @ f + 1.0 * D(f, np.inf))) < 1e-8
            hits += ex2; corr += (int(np.argmax(x)) == k)
        offs.append(float(offset)); acc.append(corr / 60); tight.append(hits / 60)
    identity = float(np.max(np.abs(np.array(acc) - np.array(tight))))
    return Result(
        passed=dis == 0 and identity == 0.0,
        summary=(f'{tot} instances, criterion-vs-LP disagreements {dis}; '
                 f'max|tightness rate - accuracy| at eps=1 = {identity}'),
        metrics=dict(instances=tot, disagreements=dis),
        data=dict(offsets=offs, accuracy=acc, tightness=tight))


def _maxent_in_ball(ph, eps, p, n, rng):
    """An independent argmax of H(q) over {q in simplex, ||q - ph||_p <= eps}, by
    multistart SLSQP.  Returns the best point that is feasible to 1e-10, or None."""
    negH = lambda q: -entropy(q)
    cons = [dict(type='eq', fun=lambda q: q.sum() - 1.0)]
    if np.isinf(p):
        bnds = [(max(0.0, ph[i] - eps), min(1.0, ph[i] + eps)) for i in range(n)]
    else:
        bnds = [(0.0, 1.0)] * n
        cons.append(dict(type='ineq', fun=lambda q: eps - np.linalg.norm(q - ph, p)))
    best = None
    for s in [ph.copy()] + [None] * 5:
        x0 = s if s is not None else np.clip(ph + 0.02 * rng.standard_normal(n), 1e-6, None)
        x0 = np.clip(x0, 1e-9, None); x0 /= x0.sum()
        res = minimize(negH, x0, method='SLSQP', bounds=bnds, constraints=cons,
                       options=dict(ftol=1e-15, maxiter=2000))
        q = res.x
        feas = (q.min() >= -1e-10 and abs(q.sum() - 1) <= 1e-10
                and nrm(q - ph, p) <= eps + 1e-10)
        if feas and (best is None or res.fun < best.fun):
            best = res
    return None if best is None else best.x


def _nominal(rng, n):
    """A nominal from one of three regimes: spread, near-sparse, or with an exact zero
    (where the barrier, lem:barrier, keeps x* interior)."""
    kind = int(rng.integers(3))
    if kind == 0:
        return rng.dirichlet(np.ones(n) * 1.5)
    ph = rng.dirichlet(np.ones(n) * 0.3)
    if kind == 2:
        ph[rng.integers(n)] = 0.0
        ph /= ph.sum()
    return ph


@check('V3.7', 'the solution is self-certifying', 'prop:selfcert', slow=True)
def v3_7():
    """prop:selfcert, each part tested against an independent instrument:

      in ball    ||x* - ph||_{r*} <= eps                          (the proof uses it)
      identity   U(f*) = W(f*) = H(x*), W from the exact LP / KKT solver
      minimax    no challenger x' (random perturbations of x*, plus ph and u) has a
                 lower exact worst case W(-log x') than x*
      maxent     an independent maximiser of H over the ball finds no entropy above
                 H(x*).  Entropy, not position: H is flat at its maximum, so this is
                 limited only by x*'s own accuracy (~1e-9), where the position is only good to
                 SLSQP's ~1e-5.  All four are judged against SOLVER_TOL.

    eps is drawn in [0.01, 0.999] eps_sat; nominals are spread, near-sparse, or have
    an exact zero; n in {3, 4, 5, 8}.
    """
    rng = np.random.default_rng(67)
    ball = ident = minimax = ent = 0.0
    cells = no_maxent = 0
    for _ in range(60):
        n = int(rng.choice([3, 4, 5, 8]))
        ph = _nominal(rng, n)
        u = np.ones(n) / n
        for p in [1, np.inf, 2]:
            r = dual(p); sat = nrm(ph - u, p)
            eps = rng.uniform(0.01, 0.999) * sat
            x = solve(ph, eps, r); f = -np.log(x)
            H = entropy(x)
            W = worst_case(ph, f, eps, p)
            U = float(ph @ f + eps * D(f, r))
            ball = max(ball, nrm(x - ph, p) - eps)
            ident = max(ident, abs(U - H), abs(W - H))
            challengers = [x * np.exp(0.05 * rng.standard_normal(n)) for _ in range(20)]
            challengers += [np.where(ph > 0, ph, 1e-3), u]
            for c in challengers:
                c = c / c.sum()
                minimax = max(minimax, W - worst_case(ph, -np.log(c), eps, p))
            xm = _maxent_in_ball(ph, eps, p, n, rng)
            if xm is None:
                no_maxent += 1
            else:
                ent = max(ent, entropy(xm) - H)
            cells += 1
    passed = max(ball, ident, minimax, ent) <= SOLVER_TOL
    return Result(
        passed=passed,
        summary=(f'identity {ident:.1e} nats; no challenger beats x* (max gain {minimax:.1e}); '
                 f'no feasible point has higher entropy (max excess {ent:.1e})'),
        details=[f'{cells} instances across r* in 1,2,inf; eps in [0.01, 0.999] eps_sat',
                 f'max ||x* - ph|| - eps           = {ball:.1e}  (want <= SOLVER_TOL)',
                 f'max |U - H|, |W - H|            = {ident:.2e} nats  (want <= SOLVER_TOL)',
                 f'max W(x*) - W(challenger)       = {minimax:.1e}  (want <= SOLVER_TOL)',
                 f'max H(maxent point) - H(x*)     = {ent:.1e}  (want <= SOLVER_TOL)',
                 f'instances where SLSQP found no feasible point: {no_maxent}'],
        metrics=dict(instances=cells, ball=ball, identity=ident, minimax=minimax,
                     entropy_excess=ent, no_maxent=no_maxent))


@check('V5.7', 'MbLS (m=0) one-hot tightness equals accuracy', 'sec:beyond')
def v5_7():
    """Mirrors V5.6 for the MbLS member (G = u - Delta), a member with an asymmetric set.
    On a one-hot target p_hat = e_k the bound is tight iff argmax x = k and eps <= n/(n-1)
    (correctly classified, below the threshold), so at a fixed eps below the threshold the
    tightness rate equals the accuracy.  The one-hot MbLS path is label smoothing with
    alpha = eps, exactly.
    Pass (fixed before the run): criterion vs LP truth 0 disagreements; tightness rate = accuracy;
    one-hot path = label smoothing to 1e-6.
    """
    from scipy.optimize import linprog
    from ..general_g import mbls, solve as solve_g
    n = 4; rng = np.random.default_rng(57); G = mbls(n); eps = 1.0   # < n/(n-1) = 1.333

    def worst(ek, f):
        ub = (ek + eps / n) / eps
        res = linprog(f, bounds=[(0, float(b)) for b in ub],
                      A_eq=np.ones((1, n)), b_eq=[1.0], method='highs')
        lam = res.x
        return float((ek + eps / n - eps * lam) @ f)

    dis = tight = correct = tot = 0
    for _ in range(400):
        k = int(rng.integers(n)); ek = np.eye(n)[k]
        x = rng.dirichlet(np.ones(n)); f = -np.log(x)
        U = ek @ f + eps * G['sigma'](f)
        is_tight = abs(U - worst(ek, f)) < 1e-7
        crit = (int(np.argmax(x)) == k)           # eps < threshold, so the criterion is correctness
        dis += is_tight != crit; tight += is_tight; correct += crit; tot += 1
    pm = 0.0
    for _ in range(30):
        k = int(rng.integers(n)); ek = np.eye(n)[k]; e = float(rng.uniform(0.05, 0.9))
        pm = max(pm, float(np.max(np.abs(solve_g(ek, e, mbls(n)) - ((1 - e) * ek + e * np.ones(n) / n)))))
    return Result(
        passed=(dis == 0 and tight == correct and pm < 1e-6),
        summary=f'tightness rate = accuracy = {tight}/{tot}; criterion vs LP {dis} disagreements; '
                f'one-hot path = label smoothing to {pm:.0e}',
        details=[f'eps = {eps} (< n/(n-1)): tight {tight}/{tot}, correct {correct}/{tot}, '
                 f'LP disagreements {dis}; path vs label smoothing {pm:.1e}'],
        metrics=dict(disagreements=dis, tight=tight, correct=correct, path_err=pm))
