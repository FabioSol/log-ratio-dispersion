"""Chapter 8 checks: the logit corollary and the budget's boundary."""
import numpy as np
from scipy.stats import multivariate_normal

from . import EXACT_TOL, SOLVER_TOL, Result, check
from .. import noise_sweep
from ..core import D, subgrad, D_rows, subgrad_rows, nrm, dual
from ..samplers import degenerate_row
from ..prob import softmax
from ..solver import solve


@check('VB.2', 'row-wise forms equal the scalar forms')
def vb_2():
    """The training sweep uses D_rows / subgrad_rows; they must equal D / subgrad."""
    rng = np.random.default_rng(80); wd = wg = 0.0
    for _ in range(300):
        n = int(rng.choice([3, 5, 8, 10]))
        V = np.array([degenerate_row(rng, n) for _ in range(12)])
        for r in [1, 2, np.inf]:
            wd = max(wd, float(np.max(np.abs(D_rows(V, r) - np.array([D(v, r) for v in V])))))
            wg = max(wg, float(np.max(np.abs(subgrad_rows(V, r) - np.array([subgrad(v, r) for v in V])))))
    return Result(
        passed=wd < 1e-8 and wg < 1e-8,
        summary=f'max|D_rows - D| = {wd:.2e};  max|subgrad_rows - subgrad| = {wg:.2e}',
        metrics=dict(D_error=wd, subgrad_error=wg))


@check('V7.1', 'the label-noise identity, per-arm radii', 'thm:neg')
def v7_1():
    """Symmetric noise at rate rho gives ph = (1-lam) p + lam u, lam = n rho/(n-1); the
    displacement lam ||u - e_k||_{r*} differs across members."""
    n = 10; worst = 0.0; details = []; radii = {}
    for rho in [.1, .2, .4]:
        p = np.zeros(n); p[0] = 1.
        ph = (1 - rho) * p + rho / (n - 1) * (1 - p)
        lam = n * rho / (n - 1)
        worst = max(worst, float(np.max(np.abs(ph - ((1 - lam) * p + lam * np.ones(n) / n)))))
        radii[rho] = [lam * nrm(np.ones(n) / n - p, dual(r)) for r in [1, 2, np.inf]]
        details.append(f'rho={rho}:  radii (r=1,2,inf) = ' + '  '.join(f'{v:.4f}' for v in radii[rho]))
    return Result(
        passed=worst < EXACT_TOL,
        summary=f'max identity residual = {worst:.1e}',
        details=details,
        metrics=dict(residual=worst, radii=radii))


@check('V7.2', 'the closed-form path on the e_k-u line', 'thm:neg')
def v7_2():
    """With nominal (1-lam) e_k + lam u the solution is (1-mu) e_k + mu u,
    mu = min(1, lam + eps/||u - e_k||_{r*}), including past saturation; also at lam = 0
    (a one-hot nominal, eps > 0), where the solution is a label-smoothed target."""
    worst = 0.0
    for n in [5, 10]:
        for rho in [.1, .2, .4]:
            p = np.zeros(n); p[0] = 1.
            lam = n * rho / (n - 1); pu = (1 - lam) * p + lam * np.ones(n) / n
            for r in [1, 2, np.inf]:
                ps = dual(r); den = nrm(np.ones(n) / n - p, ps)
                sat = nrm(pu - np.ones(n) / n, ps)
                for fr in [.25, .6, .95, 1.2]:
                    eps = fr * sat; mu = min(1., lam + eps / den)
                    pred = (1 - mu) * p + mu * np.ones(n) / n
                    worst = max(worst, float(np.max(np.abs(solve(pu, eps, r) - pred))))
    onehot = 0.0                                   # lam = 0 (cor:onehotls), eps > 0
    for n in [3, 5, 10]:
        e = np.zeros(n); e[1] = 1.; u = np.ones(n) / n
        for r in [1, 1.5, 2, 3, np.inf]:
            den = nrm(u - e, dual(r))
            for fr in [.05, .2, .5, .8, .99, 1.3]:
                mu = min(1., fr)
                onehot = max(onehot, float(np.max(np.abs(solve(e, fr * den, r) - ((1 - mu) * e + mu * u)))))
    return Result(
        passed=worst < SOLVER_TOL and onehot < SOLVER_TOL,
        summary=(f'max|solver - closed form| = {worst:.2e} over (n, rho) x 3 members x 4 radii; '
                 f'one-hot nominal (lam = 0): {onehot:.2e} over n in {{3, 5, 10}} x 5 members x 6 radii'),
        metrics=dict(worst=worst, onehot=onehot))


@check('V7.3', 'the derivative sign (where the adversary sits)', 'thm:neg')
def v7_3():
    """Expected log-loss increases along the e_k-u ray exactly when the model puts
    above-average log-mass on the true class; report how often that precondition fails."""
    rng = np.random.default_rng(82); bad = neg = tot = 0
    for _ in range(4000):
        n = int(rng.choice([3, 5, 10])); k = int(rng.integers(n))
        z = rng.normal(size=n) * rng.choice([.2, 1., 4.])
        x = np.exp(z - z.max()); x /= x.sum()
        lhs = np.mean(-np.log(x)) + np.log(x[k]); rhs = np.log(x[k]) - np.mean(np.log(x))
        bad += (np.sign(lhs) != np.sign(rhs)); neg += (lhs <= 0); tot += 1
    return Result(
        passed=bad == 0,
        summary=f'sign disagreements {bad}/{tot}; precondition fails on {neg / tot:.1%} of random x',
        metrics=dict(disagreements=int(bad), precondition_fails=neg / tot))


@check('V7.4', 'collapse: past an exact threshold the minimiser is W = 0', 'obs:collapse', slow=True)
def v7_4():
    """On one sweep instance (rho = 0.4, seed 1000):

      threshold  eps_c exactly: by the LP of collapse_threshold for r in {1, inf}, and by
                 the convex programme of collapse_threshold_2 for r = 2;
      solver     L-BFGS starts at W = 0, where only the cross-entropy descent direction is
                 visible, so it can report collapse from eps_G = ||G||^2 / mean D_r(Xb G)
                 (where that direction stops descending) although eps_G <= eps_c.  A fit
                 at 0.9 eps_G must not be collapsed; fits in [eps_G, eps_c) are reported;
      above      fits at 1.1 eps_c and at eps = 3 are collapsed (|W|max < COLLAPSE_TOL),
                 and their predictions are the uniform distribution to 1e-10;
      residue    where the optimiser stops short of 0, the residual W points along the
                 cross-entropy descent direction at 0 (cosine > 1 - 1e-6): the argmax
                 of that residue is the 'plateau' an accuracy curve reports there.
    """
    n, d, N, sep, l2 = 10, 20, 400, 0.55, 1e-3
    rng = np.random.default_rng(1000)
    M = rng.normal(size=(n, d)) * sep
    Xtr, ytr, yc = noise_sweep.make_data(n, d, N, 0.4, rng, M, sep)
    Xte, yte, _ = noise_sweep.make_data(n, d, 10000, 0.0, rng, M, sep)
    Xb = np.hstack([Xtr, np.ones((N, 1))])
    G = -(Xb.T @ (np.ones((N, n)) / n - np.eye(n)[yc])) / N
    Xte_b = np.hstack([Xte, np.ones((len(Xte), 1))])
    ok = True
    details = []
    residue_acc = []
    for r in [1, 2, np.inf]:
        ec = noise_sweep.exact_threshold(Xtr, yc, n, r)
        eg = float((G * G).sum() / D_rows(Xb @ G, r).mean())
        below = float(np.abs(noise_sweep.fit(Xtr, yc, n, r, 0.9 * eg, l2=l2)).max())
        band = float(np.abs(noise_sweep.fit(Xtr, yc, n, r, 0.5 * (eg + ec), l2=l2)).max())
        warm = float(np.abs(noise_sweep.fit(Xtr, yc, n, r, 0.97 * ec, l2=l2,
                                            w0=noise_sweep.fit(Xtr, yc, n, r, 0.0, l2=l2))).max())
        row = (f'r={str(r):>3}: eps_c = {ec:.4f} (exact), eps_G = {eg:.4f};  |W|max at 0.9 eps_G = '
               f'{below:.1e}; inside [eps_G, eps_c) from 0: {band:.1e}, at 0.97 eps_c warm-started: {warm:.1e}')
        ok &= below > 1e-4 and eg <= ec + 1e-9 and warm > 1e-4
        for e in [1.1 * ec, 3.0]:
            W = noise_sweep.fit(Xtr, yc, n, r, e, l2=l2)
            wm = float(np.abs(W).max())
            unif = float(np.abs(softmax(Xte_b @ W) - 1 / n).max())
            ok &= wm < noise_sweep.COLLAPSE_TOL and unif < 1e-10
            if wm > 0:
                cos = float((W * G).sum() / np.linalg.norm(W) / np.linalg.norm(G))
                ok &= cos > 1 - 1e-6
                residue_acc.append(noise_sweep.evaluate(W, Xte, yte))
                row += f';  at {e:.2f}: |W|max {wm:.0e}, cos(W, descent) {cos:.7f}'
            else:
                row += f';  at {e:.2f}: W = 0 exactly'
        details.append(row)
    proto = noise_sweep.evaluate(noise_sweep.fit(Xtr, yc, n, 'prototype', 0.0), Xte, yte)
    details.append(f'argmax of the residue: accuracy {min(residue_acc):.4f}-{max(residue_acc):.4f}; '
                   f'prototype rule <x, mu_k>: {proto:.4f}')
    return Result(
        passed=ok,
        summary=('past eps_c the minimiser is W = 0 (uniform predictions); the reported '
                 f'"plateau" {residue_acc[0]:.2f} is the argmax of the optimiser\'s residue'),
        details=details,
        metrics=dict(residue_accuracy=residue_acc, prototype=proto))


@check('V7.6', 'the calibration overshoot depends on how the strength was chosen', 'tab:cal', slow=True)
def v7_6():
    """Held-out log-loss and ECE for every one-hot arm, under two selection rules, against the
    unpenalised MLE (eps = 0).  For each arm the full per-arm grid of the sweep is refitted (the
    D_r arms and MbLS are non-smooth at the origin, so the chain is warm-started to avoid the W = 0
    false-stop of V7.4; the smooth arms converge regardless), and two strengths are read off the
    mean validation curves: the one that maximises validation accuracy (what tab:sweep uses) and
    the one that minimises validation log-loss on the noisy validation labels.  ECE uses 15
    equal-width confidence bins.  MbLS (a general-G member, V7.8) is reported alongside the five.

    The point (review): the overshoot of tab:cal is largely a property of the accuracy rule.
    Accuracy is flat in eps, so it lands high on a plateau; the log-loss rule picks a much smaller
    eps and the overshoot mostly disappears, reversing at rho = 0 (the penalty then helps).  The
    certificate reading would set eps to the displacement lam||u - e_k|| (the smallest ball
    containing the truth, tab:strengths); but thm:neg moves the estimate away from the truth for
    every eps > 0, so neither rule should approach the displacement, and the log-loss-chosen eps
    stays well below it.  The positive eps it picks is finite-sample capacity control, not coverage.

    Pass, at every one-hot noise rate rho in {0, 0.1, 0.2, 0.4}:
      blind             the five penalised arms' mean accuracy (accuracy rule) spans below 0.03;
      degrades          under the accuracy rule, mean ECE exceeds the baseline for every arm;
      underconfident    under the accuracy rule, mean confidence is below mean accuracy for every arm;
      ordered           under the accuracy rule, the norm balls raise ECE and log-loss less than
                        either relative (max over D_r < min over {ls, cp}), both metrics;
      log-loss helps    the mean ECE over the arms is lower under the log-loss rule than the
                        accuracy rule;
      zero reversal     at rho = 0 the log-loss rule puts both log-loss and ECE below the baseline
                        for every arm;
      below displacement the log-loss-chosen eps is below the displacement for every D_r arm at
                        every positive rate (neither rule approaches the coverage radius).
    """
    sw = noise_sweep.load_sweep()
    n, d, seeds = sw['n'], sw['d'], sw['seeds']
    ntr, nte, nva, sep, l2 = sw['ntr'], sw['nte'], sw['nva'], sw['sep'], sw['l2']
    DR = ['D_1', 'D_2', 'D_inf']; REL = ['ls', 'cp']; PEN = DR + REL; ALL = PEN + ['mbls']
    KEY = {'D_1': 1, 'D_2': 2, 'D_inf': np.inf, 'ls': 'ls', 'cp': 'cp', 'mbls': 'mbls'}
    DISP = {'D_1': 1, 'D_2': 2, 'D_inf': np.inf}
    rhos = [0.0, 0.1, 0.2, 0.4]
    BASE = np.geomspace(0.02, 4.0, 27)

    def arm_grid(a):
        g = noise_sweep.GRIDS[a] if a in noise_sweep.GRIDS else BASE
        return np.concatenate([[0.0], np.asarray(g, float)])

    checks = dict.fromkeys(['blind', 'degrades', 'underconfident', 'ordered',
                            'log-loss helps', 'zero reversal', 'below displacement'], True)
    details = [f'{seeds} seeds; held-out metrics under two selection rules (acc | log-loss):']
    table = {}
    for rho in rhos:
        lam = n * rho / (n - 1)
        G = {a: arm_grid(a) for a in ALL}
        # per arm, per grid point, across seeds
        vacc = {a: np.full((seeds, len(G[a])), np.nan) for a in ALL}
        vll = {a: v.copy() for a, v in vacc.items()}; tacc = {a: v.copy() for a, v in vacc.items()}
        tll = {a: v.copy() for a, v in vacc.items()}; tece = {a: v.copy() for a, v in vacc.items()}
        tconf = {a: v.copy() for a, v in vacc.items()}
        base_ll = []; base_ece = []
        for si in range(seeds):
            rng = np.random.default_rng(1000 + si)
            M = rng.normal(size=(n, d)) * sep
            Xtr, ytr, yc = noise_sweep.make_data(n, d, ntr, rho, rng, M, sep)
            Xte, yte, _ = noise_sweep.make_data(n, d, nte, 0.0, rng, M, sep)
            Xva, _, yva = noise_sweep.make_data(n, d, nva, rho, rng, M, sep)
            W0 = noise_sweep.fit(Xtr, yc, n, 1, 0.0, nominal='onehot', lam=lam, l2=l2)
            _, l0, e0 = noise_sweep.evaluate_full(W0, Xte, yte)
            base_ll.append(l0); base_ece.append(e0)
            for a in ALL:
                W = W0
                for gi, s in enumerate(G[a]):
                    if gi:
                        W = noise_sweep.fit(Xtr, yc, n, KEY[a], float(s),
                                            nominal='onehot', lam=lam, l2=l2, w0=W)
                    if float(np.abs(W).max()) < noise_sweep.COLLAPSE_TOL:
                        break
                    av, lv, ev = noise_sweep.evaluate_full(W, Xte, yte)
                    Pv = noise_sweep._probs(W, Xva)
                    vacc[a][si, gi] = (Pv.argmax(1) == yva).mean()
                    vll[a][si, gi] = -np.mean(np.log(np.clip(Pv[np.arange(len(yva)), yva], 1e-12, 1.0)))
                    tacc[a][si, gi] = av; tll[a][si, gi] = lv; tece[a][si, gi] = ev
                    tconf[a][si, gi] = float(noise_sweep._probs(W, Xte).max(1).mean())
        bll = float(np.mean(base_ll)); bece = float(np.mean(base_ece))
        rec = {}
        for a in ALL:
            dead = np.isnan(tacc[a]).any(0)
            ga = int(np.nanargmax(np.where(dead, np.nan, np.nanmean(vacc[a], 0))))
            gl = int(np.nanargmin(np.where(dead, np.nan, np.nanmean(vll[a], 0))))
            rec[a] = dict(
                eps_acc=float(G[a][ga]), eps_ll=float(G[a][gl]),
                acc=dict(acc=float(np.nanmean(tacc[a][:, ga])), ll=float(np.nanmean(tll[a][:, ga])),
                         ece=float(np.nanmean(tece[a][:, ga])), conf=float(np.nanmean(tconf[a][:, ga]))),
                ll=dict(acc=float(np.nanmean(tacc[a][:, gl])), ll=float(np.nanmean(tll[a][:, gl])),
                        ece=float(np.nanmean(tece[a][:, gl]))),
                disp=(noise_sweep.predicted_radius(n, rho, DISP[a]) if a in DISP else None))
        accm = {a: rec[a]['acc']['acc'] for a in PEN}
        spread = max(accm.values()) - min(accm.values())
        maxD_ece = max(rec[a]['acc']['ece'] for a in DR); minREL_ece = min(rec[a]['acc']['ece'] for a in REL)
        maxD_ll = max(rec[a]['acc']['ll'] for a in DR); minREL_ll = min(rec[a]['acc']['ll'] for a in REL)
        checks['blind'] &= spread < 0.03
        checks['degrades'] &= all(rec[a]['acc']['ece'] > bece for a in ALL)
        checks['underconfident'] &= all(rec[a]['acc']['conf'] < rec[a]['acc']['acc'] for a in ALL)
        checks['ordered'] &= maxD_ece < minREL_ece and maxD_ll < minREL_ll
        checks['log-loss helps'] &= (np.mean([rec[a]['ll']['ece'] for a in ALL])
                                     < np.mean([rec[a]['acc']['ece'] for a in ALL]))
        if rho == 0.0:
            checks['zero reversal'] &= all(rec[a]['ll']['ll'] < bll and rec[a]['ll']['ece'] < bece for a in ALL)
        else:
            checks['below displacement'] &= all(rec[a]['eps_ll'] < rec[a]['disp'] for a in DR)
        table[rho] = dict(base_ll=bll, base_ece=bece, arms={a: rec[a] for a in ALL})
        details.append(f'rho={rho}: baseline ll {bll:.3f} ece {bece:.3f}; acc spread {spread:.3f}')
        for a in ALL:
            r = rec[a]; ds = f"{r['disp']:.3f}" if r['disp'] is not None else "  -  "
            details.append(
                f'  {a:6s} acc: e={r["eps_acc"]:.3f} ll={r["acc"]["ll"]:.3f} ece={r["acc"]["ece"]:.3f} '
                f'conf={r["acc"]["conf"]:.3f} | ll: e={r["eps_ll"]:.3f} ll={r["ll"]["ll"]:.3f} '
                f'ece={r["ll"]["ece"]:.3f} | disp={ds}')
    details += [f"{k}: {'ok' if v else 'FAILS'}" for k, v in checks.items()]
    return Result(
        passed=all(checks.values()),
        summary=('the accuracy rule overshoots (ECE/ll above the MLE, norm balls least); the '
                 'log-loss rule shrinks it, reverses at rho=0, and stays below the coverage '
                 'displacement (neither rule approaches it); '
                 + ', '.join(f"{k} {'ok' if v else 'FAILS'}" for k, v in checks.items())),
        details=details,
        metrics=dict(checks=checks, table=table))


@check('V7.7', 'the chosen strength sits below collapse and ignores the displacement', 'tab:strengths')
def v7_7():
    """For each member, the accuracy-chosen strength eps* against the exact collapse threshold
    eps_c (smallest over seeds) and the coverage displacement lam||u - e_k||_{r*}, the smallest
    radius whose ball contains the truth.  All read from the frozen sweep (noise_sweep.json,
    one-hot nominal); no fitting.  (The log-loss-chosen eps*_LL shown in tab:strengths is computed
    and checked in V7.6, which also asserts it stays below the displacement.)  The point of
    tab:strengths: the chosen strength is capacity control, well inside collapse and far from the
    coverage radius.

    Pass, over the members r in {1, 2, infinity} and rho in {0, 0.1, 0.2, 0.4}:
      below collapse   eps* < eps_c for every member and rate (by roughly two to seven times);
      threshold falls  eps_c (smallest over seeds) is strictly decreasing in rho for every
                       member, so more label noise collapses the model at a smaller strength;
      not coverage     eps* does not match the coverage displacement: at rho = 0 the displacement
                       is zero yet eps* > 0, and at rho = 0.4 eps* is below it and at most twice
                       its value at rho = 0.
    """
    sw = noise_sweep.load_sweep()
    n = sw['n']; D = ['D_1', 'D_2', 'D_inf']; KEY = {'D_1': 1, 'D_2': 2, 'D_inf': np.inf}
    rhos = [0.0, 0.1, 0.2, 0.4]
    es = {a: [sw['res'][f'onehot|{rho}'][a]['val_star'] for rho in rhos] for a in D}
    ec = {a: [min(sw['res'][f'onehot|{rho}'][a]['eps_c']) for rho in rhos] for a in D}
    disp = {a: [noise_sweep.predicted_radius(n, rho, KEY[a]) for rho in rhos] for a in D}
    checks = dict(
        below_collapse=all(es[a][i] < ec[a][i] for a in D for i in range(4)),
        threshold_falls=all(all(x > y for x, y in zip(ec[a], ec[a][1:])) for a in D),
        not_coverage=all(disp[a][0] == 0 and es[a][0] > 0 for a in D)
        and all(es[a][3] < disp[a][3] and es[a][3] <= 2 * es[a][0] for a in D))
    details = ['chosen strength eps*, exact threshold eps_c (min over seeds), displacement '
               'lam||u - e_k|| per member and rho (0, 0.1, 0.2, 0.4):']
    for a in D:
        details.append(f'  {a:6s} eps*  ' + ' '.join(f'{v:.3f}' for v in es[a]))
        details.append(f'  {"":6s} eps_c ' + ' '.join(f'{v:.3f}' for v in ec[a]))
        details.append(f'  {"":6s} disp  ' + ' '.join(f'{v:.3f}' for v in disp[a])
                       + f'   (eps_c/eps* min {min(ec[a][i]/es[a][i] for i in range(4)):.1f})')
    details += [f"{k}: {'ok' if v else 'FAILS'}" for k, v in checks.items()]
    return Result(
        passed=all(checks.values()),
        summary=('eps* is below the exact collapse threshold (2-7x), the threshold shrinks with '
                 'noise, and eps* does not match the coverage displacement; '
                 + ', '.join(f"{k} {'ok' if v else 'FAILS'}" for k, v in checks.items())),
        details=details,
        metrics=dict(eps_star=es, eps_c=ec, displacement=disp, checks=checks))


@check('V7.8', 'MbLS (m = 0), a general-G member, reproduces the box member', 'sec:logit-sweep', slow=True)
def v7_8():
    """A member of the beyond-norm-balls construction (sec:beyond) on the classification sweep.
    MbLS with margin m = 0 takes G = u - simplex, whose support function is sigma_G(v) = mean(v)
    - min(v); it is not a norm ball.  We run it on the one-hot sweep exactly as the other arms
    (warm-started along a grid, strength chosen on the noisy validation set, scored on clean
    held-out data; collapse flagged by |W|max below COLLAPSE_TOL rather than an exact threshold),
    and compare it to the box member r = 1, the endpoint it is closest to.

    Pass, at every one-hot noise rate rho in {0, 0.1, 0.2, 0.4}:
      a real member   the chosen strength is strictly inside the grid (a penalised operating
                      point, neither the unpenalised model nor a collapsed one);
      matches the box its mean held-out accuracy is within one point of the box member r = 1 on
                      the same seeds (paired);
      overshoots      like every norm-ball member, at its chosen strength it raises the ECE above
                      the unpenalised MLE and leaves the model underconfident (mean confidence
                      below mean accuracy).
    """
    T95 = 2.131                                   # t quantile, 15 degrees of freedom
    sw = noise_sweep.load_sweep()
    n, d, seeds = sw['n'], sw['d'], sw['seeds']
    ntr, nte, nva, sep, l2 = sw['ntr'], sw['nte'], sw['nva'], sw['sep'], sw['l2']
    rhos = [0.0, 0.1, 0.2, 0.4]
    grid = np.concatenate([[0.0], np.geomspace(0.02, 4.0, 27)])
    checks = dict.fromkeys(['a real member', 'matches the box', 'overshoots'], True)
    details = [f'{seeds} seeds; MbLS (m = 0) at its validation strength against the box member:']
    for rho in rhos:
        lam = n * rho / (n - 1)
        te = np.full((seeds, len(grid)), np.nan); va = te.copy()
        ll = te.copy(); ece_ = te.copy(); conf = te.copy()
        base_ece = []
        for si in range(seeds):
            rng = np.random.default_rng(1000 + si)
            M = rng.normal(size=(n, d)) * sep
            Xtr, ytr, yc = noise_sweep.make_data(n, d, ntr, rho, rng, M, sep)
            Xte, yte, _ = noise_sweep.make_data(n, d, nte, 0.0, rng, M, sep)
            Xva, _, yva = noise_sweep.make_data(n, d, nva, rho, rng, M, sep)
            W0 = noise_sweep.fit(Xtr, yc, n, 'mbls', 0.0, nominal='onehot', lam=lam, l2=l2)
            base_ece.append(noise_sweep.evaluate_full(W0, Xte, yte)[2])
            W = W0
            for gi, s in enumerate(grid):
                if gi:
                    W = noise_sweep.fit(Xtr, yc, n, 'mbls', float(s), nominal='onehot',
                                        lam=lam, l2=l2, w0=W)
                if float(np.abs(W).max()) < noise_sweep.COLLAPSE_TOL:
                    break
                a, l, e = noise_sweep.evaluate_full(W, Xte, yte)
                te[si, gi] = a; ll[si, gi] = l; ece_[si, gi] = e
                conf[si, gi] = float(noise_sweep._probs(W, Xte).max(1).mean())
                va[si, gi] = noise_sweep.evaluate(W, Xva, yva)
        live = ~np.isnan(te).any(0)
        gv = int(np.flatnonzero(live)[np.nanargmax(np.nanmean(va[:, live], 0))])
        box = np.array(sw['res'][f'onehot|{rho}']['D_1']['test_at_val'], float)
        diff = te[:, gv] - box
        m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(seeds)
        acc_m = float(np.nanmean(te[:, gv])); conf_m = float(np.nanmean(conf[:, gv]))
        ece_m = float(np.nanmean(ece_[:, gv])); base_m = float(np.mean(base_ece))
        checks['a real member'] &= 0 < gv < int(np.flatnonzero(live)[-1])
        checks['matches the box'] &= abs(m) < 0.01
        checks['overshoots'] &= ece_m > base_m and conf_m < acc_m
        details.append(
            f'rho={rho}: eps*={grid[gv]:.3f}  acc {acc_m:.3f} (box {box.mean():.3f}, '
            f'diff {m:+.4f} [{m - T95 * se:+.4f}, {m + T95 * se:+.4f}])  '
            f'll {float(np.nanmean(ll[:, gv])):.3f}  ece {ece_m:.3f} (MLE {base_m:.3f})  conf {conf_m:.3f}')
    details += [f"{k}: {'ok' if v else 'FAILS'}" for k, v in checks.items()]
    return Result(
        passed=all(checks.values()),
        summary=('MbLS (m = 0), a non-norm general-G member, lands on the box member within one '
                 'point of accuracy and overshoots calibration the same way; '
                 + ', '.join(f"{k} {'ok' if v else 'FAILS'}" for k, v in checks.items())),
        details=details,
        metrics=dict(checks=checks))


@check('V7.9', 'a second setting: anisotropic classes dethrone the NAIVE prototype', 'tab:aniso', slow=True)
def v7_9():
    """The sweep's findings on a second data model.  The isotropic classes of sec:logit-sweep
    make the naive prototype rule <x, mu_k> the Bayes rule, which is why it is hard to beat there.
    Here each seed draws a shared anisotropic covariance Sigma = A A^T (a random rotation of
    eigenvalues spanning a factor of 25, rescaled to unit average variance so the total noise power
    matches the isotropic sweep), so the Bayes rule is <x, Sigma^{-1} mu_k> and the naive prototype
    is no longer optimal.  Everything else is as in sec:logit-sweep: one-hot nominal, the same arms
    and grid, strength chosen on a noisy validation set, scored on clean held-out data, collapse by
    |W|max.

    Reference rules per seed: Bayes (true means and Sigma); the whitened prototype
    <x, Sigma^{-1} mu_hat_k>, which uses the TRUE Sigma and is therefore partly an oracle; LDA with
    Sigma estimated from the training set (pooled within-class covariance), the honest generative
    baseline; and the naive prototype <x, mu_hat_k>.

    The point (review): anisotropy removes only the NAIVE prototype's lead, which was an artefact of
    isotropy.  A correctly specified generative rule still beats every discriminative arm, oracle or
    not, and the gap widens with noise (Efron 1975: LDA is more efficient than logistic regression
    when the Gaussian model holds).  The capacity-control and calibration findings carry over.

    Pass, at every one-hot noise rate rho in {0, 0.1, 0.2, 0.4}:
      naive prototype falls  the best penalised arm beats the naive prototype by more than five
                             points, paired interval clear of zero (it dominated under isotropy);
      generative wins        both the whitened prototype and LDA beat the best penalised arm,
                             paired interval clear of zero;
      no member special      the six arms' mean accuracy spans less than five points;
      penalty helps          the best arm is at least as accurate as the unpenalised MLE;
      overshoots             every arm raises the ECE above the MLE and is underconfident (the same
                             calibration overshoot as the isotropic sweep), norm balls less than the
                             relatives on both ECE and log-loss;
      bayes verified         the linear 'Bayes (true Sigma)' rule agrees, on every test point, with
                             the argmax of the exact Gaussian class log-posterior computed
                             independently from scipy.stats.multivariate_normal.logpdf plus equal
                             log-priors (0 disagreements), confirming the bias term is correct.
    """
    T95 = 2.131
    sw = noise_sweep.load_sweep()
    n, d, seeds = sw['n'], sw['d'], sw['seeds']
    ntr, nte, nva, sep, l2 = sw['ntr'], sw['nte'], sw['nva'], sw['sep'], sw['l2']
    DR = ['D_1', 'D_2', 'D_inf']; REL = ['ls', 'cp']; ALL = DR + REL + ['mbls']
    KEY = {'D_1': 1, 'D_2': 2, 'D_inf': np.inf, 'ls': 'ls', 'cp': 'cp', 'mbls': 'mbls'}
    rhos = [0.0, 0.1, 0.2, 0.4]
    grid = np.concatenate([[0.0], np.geomspace(0.02, 4.0, 27)])

    def cov_factor(rng):                              # Sigma = A A^T, mean eigenvalue 1
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        lam = np.geomspace(0.2, 5.0, d); lam *= d / lam.sum()
        return Q * np.sqrt(lam)

    checks = dict.fromkeys(['naive prototype falls', 'generative wins', 'no member special',
                            'penalty helps', 'overshoots', 'bayes verified'], True)
    details = [f'{seeds} seeds; anisotropic classes (condition number 25), one-hot nominal:']
    table = {}
    bayes_disagree = 0                                # linear Bayes rule vs exact MVN log-posterior
    for rho in rhos:
        lam = n * rho / (n - 1)
        acc = {a: [] for a in ALL}; ll = {a: [] for a in ALL}
        ece_ = {a: [] for a in ALL}; conf = {a: [] for a in ALL}
        mle_acc = []; mle_ece = []; proto = []; wproto = []; lda = []; bayes = []
        for si in range(seeds):
            rng = np.random.default_rng(2000 + si)
            M = rng.normal(size=(n, d)) * sep; A = cov_factor(rng)
            Sinv = np.linalg.inv(A @ A.T)
            Xtr, ytr, yc = noise_sweep.make_data_cov(n, d, ntr, rho, rng, M, A)
            Xte, yte, _ = noise_sweep.make_data_cov(n, d, nte, 0.0, rng, M, A)
            Xva, _, yva = noise_sweep.make_data_cov(n, d, nva, rho, rng, M, A)
            mu = np.array([Xtr[yc == k].mean(0) if (yc == k).any() else np.zeros(d) for k in range(n)])
            proto.append(float((np.argmax(Xte @ mu.T, 1) == yte).mean()))
            wb = -0.5 * np.einsum('ki,ij,kj->k', mu, Sinv, mu)
            wproto.append(float((np.argmax(Xte @ (Sinv @ mu.T) + wb, 1) == yte).mean()))
            Shat = np.zeros((d, d)); cnt = 0                     # LDA: pooled within-class cov, estimated
            for k in range(n):
                Xk = Xtr[yc == k]
                if len(Xk):
                    C = Xk - mu[k]; Shat += C.T @ C; cnt += len(Xk)
            Shinv = np.linalg.inv(Shat / cnt + 1e-6 * np.eye(d))
            lb = -0.5 * np.einsum('ki,ij,kj->k', mu, Shinv, mu)
            lda.append(float((np.argmax(Xte @ (Shinv @ mu.T) + lb, 1) == yte).mean()))
            bb = -0.5 * np.einsum('ki,ij,kj->k', M, Sinv, M)
            bayes_lin = np.argmax(Xte @ (Sinv @ M.T) + bb, 1)
            bayes.append(float((bayes_lin == yte).mean()))
            if rho == rhos[0]:                        # Bayes uses the true M, Sigma: rho-independent
                Sig = A @ A.T                         # exact Gaussian log-posterior, equal priors
                logpost = np.column_stack([multivariate_normal.logpdf(Xte, mean=M[k], cov=Sig)
                                           for k in range(n)]) + np.log(1.0 / n)
                bayes_disagree += int((bayes_lin != logpost.argmax(1)).sum())
            W0 = noise_sweep.fit(Xtr, yc, n, 1, 0.0, nominal='onehot', lam=lam, l2=l2)
            a0, _, e0 = noise_sweep.evaluate_full(W0, Xte, yte)
            mle_acc.append(a0); mle_ece.append(e0)
            for a in ALL:
                W = W0; ws = [W0]
                for gi in range(1, len(grid)):
                    W = noise_sweep.fit(Xtr, yc, n, KEY[a], float(grid[gi]),
                                        nominal='onehot', lam=lam, l2=l2, w0=W)
                    if float(np.abs(W).max()) < noise_sweep.COLLAPSE_TOL:
                        break
                    ws.append(W)
                gv = int(np.argmax([noise_sweep.evaluate(w, Xva, yva) for w in ws]))
                av, lv, ev = noise_sweep.evaluate_full(ws[gv], Xte, yte)
                acc[a].append(av); ll[a].append(lv); ece_[a].append(ev)
                conf[a].append(float(noise_sweep._probs(ws[gv], Xte).max(1).mean()))
        accm = {a: float(np.mean(acc[a])) for a in ALL}
        best = max(ALL, key=lambda a: accm[a])
        gap = np.array(acc[best]) - np.array(proto); gm = gap.mean()
        gse = gap.std(ddof=1) / np.sqrt(seeds)
        spread = max(accm.values()) - min(accm.values())
        maxD_ece = max(np.mean(ece_[a]) for a in DR); minREL_ece = min(np.mean(ece_[a]) for a in REL)
        maxD_ll = max(np.mean(ll[a]) for a in DR); minREL_ll = min(np.mean(ll[a]) for a in REL)
        # the generative rules beat the best discriminative arm, paired by seed
        gwin = min((np.array(gen) - np.array(acc[best])).mean()
                   - T95 * (np.array(gen) - np.array(acc[best])).std(ddof=1) / np.sqrt(seeds)
                   for gen in (wproto, lda))
        checks['naive prototype falls'] &= gm > 0.05 and gm - T95 * gse > 0
        checks['generative wins'] &= gwin > 0
        checks['no member special'] &= spread < 0.05
        checks['penalty helps'] &= accm[best] >= np.mean(mle_acc)
        checks['overshoots'] &= (all(np.mean(ece_[a]) > np.mean(mle_ece) for a in ALL)
                                 and all(np.mean(conf[a]) < accm[a] for a in ALL)
                                 and maxD_ece < minREL_ece and maxD_ll < minREL_ll)
        table[rho] = dict(acc=accm, ll={a: float(np.mean(ll[a])) for a in ALL},
                          ece={a: float(np.mean(ece_[a])) for a in ALL},
                          mle_acc=float(np.mean(mle_acc)), mle_ece=float(np.mean(mle_ece)),
                          proto=float(np.mean(proto)), wproto=float(np.mean(wproto)),
                          lda=float(np.mean(lda)), bayes=float(np.mean(bayes)))
        details.append(
            f'rho={rho}: Bayes {np.mean(bayes):.3f}, whitened-proto(trueSig) {np.mean(wproto):.3f}, '
            f'LDA(estSig) {np.mean(lda):.3f}, naive-proto {np.mean(proto):.3f}, '
            f'MLE {np.mean(mle_acc):.3f} (ece {np.mean(mle_ece):.3f}); best arm {best} {accm[best]:.3f} '
            f'beats naive proto by {gm:+.3f} [{gm - T95 * gse:+.3f}, {gm + T95 * gse:+.3f}]; '
            f'worst generative margin over best arm {gwin:+.3f}; accuracy spread {spread:.3f}')
        for a in ALL:
            details.append(f'  {a:6s} acc {accm[a]:.3f}  ll {np.mean(ll[a]):.3f}  '
                           f'ece {np.mean(ece_[a]):.3f}  conf {np.mean(conf[a]):.3f}')
    checks['bayes verified'] = (bayes_disagree == 0)
    details.append(f"Bayes (true Sigma) linear rule vs exact MVN log-posterior argmax over "
                   f"{seeds} seeds x {nte} points: {bayes_disagree} disagreements")
    details += [f"{k}: {'ok' if v else 'FAILS'}" for k, v in checks.items()]
    return Result(
        passed=all(checks.values()),
        summary=('under anisotropy only the NAIVE prototype falls; a correctly specified generative '
                 'rule (whitened prototype, LDA) still beats every discriminative arm, and the '
                 'capacity-control and calibration-overshoot findings persist; '
                 + ', '.join(f"{k} {'ok' if v else 'FAILS'}" for k, v in checks.items())),
        details=details,
        metrics=dict(checks=checks, table=table))


@check('V7.5', 'the sweep: capacity control, not a certificate', 'fig:sweep')
def v7_5():
    """Reads the frozen 16-seed sweep (noise_sweep.json, one-hot nominal).  Each arm's strength
    is chosen on a noisy validation set; its test accuracy there is compared across arms with
    95% t-intervals paired by seed (the same data for every arm).  Accuracy is reported only at
    strengths below every seed's exact collapse threshold.

    Pass, matching the statements of sec:logit-sweep and fig:sweep (written after the data
    were seen, so this confirms the text rather than an advance prediction):
      helps at rho = 0   every D_r arm beats the unpenalised model, interval above zero;
      does not track     every D_r arm's chosen strength at rho = 0.4 is at most twice its
                         choice at rho = 0, and below the displacement of the nominal at 0.4;
      not special        at every rho, label smoothing is within noise of the best D_r arm
                         or significantly above it (upper end of the interval above zero);
      collapse           every seed's exact eps_c is at most 1.2 for every D_r arm and rho;
                         label smoothing and the confidence penalty never collapse;
      inside the range   every arm's chosen strength is strictly inside its grid;
      prototype          within noise of the best penalised arm at rho = 0, significantly
                         above it at every rho > 0.
    """
    T95 = 2.131                                   # t quantile, 15 degrees of freedom
    sw = noise_sweep.load_sweep()
    D = ['D_1', 'D_2', 'D_inf']; P = D + ['ls', 'cp']
    rhos = [0.0, 0.1, 0.2, 0.4]
    res = {rho: sw['res'][f'onehot|{rho}'] for rho in rhos}

    def acc(R, a):
        return np.array(R[a]['test_at_val'] if a in P else R['prototype']['per_seed'], float)

    def ci(d):
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        return m, m - T95 * se, m + T95 * se

    checks = dict.fromkeys(['helps at rho = 0', 'does not track', 'not special', 'collapse',
                            'inside the range', 'prototype'], True)
    details = [f"{sw['seeds']} seeds; test accuracy at the validation choice, and difference "
               'from the best arm [95% paired interval]:']
    for rho in rhos:
        R = res[rho]
        none = np.array(R['D_1']['per_seed'], float)[:, 0]
        best = max(P, key=lambda a: acc(R, a).mean())
        bestD = max(D, key=lambda a: acc(R, a).mean())
        row = []
        for a in P + ['prototype']:
            m, lo, hi = ci(acc(R, a) - acc(R, best))
            row.append(f'{a} {acc(R, a).mean():.4f} ({m:+.4f} [{lo:+.4f}, {hi:+.4f}])')
        details.append(f'rho={rho}: unpenalised {none.mean():.4f};  ' + ';  '.join(row))
        if rho == 0.0:
            checks['helps at rho = 0'] = all(ci(acc(R, a) - none)[1] > 0 for a in D)
            details.append('  gain over unpenalised at rho = 0 (points): ' + ';  '.join(
                f'{a} {100 * m:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]'
                for a in P for m, lo, hi in [ci(acc(R, a) - none)]))
        details.append('  members, paired: ' + ';  '.join(
            f'{x} - {y} {100 * m:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]'
            for x, y in [('D_2', 'D_1'), ('D_2', 'D_inf'), ('D_inf', 'D_1')]
            for m, lo, hi in [ci(acc(R, x) - acc(R, y))]))
        checks['not special'] &= ci(acc(R, 'ls') - acc(R, bestD))[2] > 0
        checks['collapse'] &= all(max(R[a]['eps_c']) <= 1.2 for a in D)
        checks['collapse'] &= all(not any(R[a]['collapsed_frac']) for a in ['ls', 'cp'])
        checks['inside the range'] &= all(0 < R[a]['val_star_index'] and not R[a]['val_star_at_edge'] for a in P)
        _, lo, hi = ci(acc(R, 'prototype') - acc(R, best))
        checks['prototype'] &= (lo < 0 < hi) if rho == 0.0 else lo > 0
    checks['does not track'] = all(
        res[0.4][a]['val_star'] <= 2 * res[0.0][a]['val_star']
        and res[0.4][a]['val_star'] < noise_sweep.predicted_radius(10, 0.4, {'D_1': 1, 'D_2': 2, 'D_inf': np.inf}[a])
        for a in D)
    details.append('chosen strength per rho: ' + ';  '.join(
        f"{a} " + ', '.join(f"{res[r][a]['val_star']:.3g}" for r in rhos) for a in P))
    details.append('exact eps_c, smallest over seeds, per rho: ' + ';  '.join(
        f"{a} " + ', '.join(f"{min(res[r][a]['eps_c']):.2f}" for r in rhos) for a in D))
    details += [f"{k}: {'ok' if v else 'FAILS'}" for k, v in checks.items()]
    return Result(
        passed=all(checks.values()),
        summary=', '.join(f"{k} {'ok' if v else 'FAILS'}" for k, v in checks.items()),
        details=details,
        metrics=dict(checks=checks))
