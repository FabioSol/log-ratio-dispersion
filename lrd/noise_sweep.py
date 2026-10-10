#!/usr/bin/env python3
"""The label-noise sweep behind Chapter 7 (fig:sweep, obs:collapse).

A linear softmax classifier is trained on Gaussian class clusters with symmetric
label noise at rate rho, under each penalty arm, across a grid of strengths eps,
and scored on clean held-out data.

Two nominals, because they answer different questions:

    onehot  -- p_hat = e_{y~}, the noisy label.  This is what a practitioner trains
               on.  thm:neg says nothing about it (a one-hot nominal is at l1
               distance 0 or 2 from the truth); the only question this arm answers
               is whether eps* depends on rho.
    soft    -- p_hat = (1-lam) e_{y~} + lam u with lam = n rho/(n-1) known.  This is
               the object thm:neg is about; the per-arm radius lam ||u - e_k||_{r*}
               is the true displacement of the nominal.  It is also label smoothing
               at alpha = lam plus the penalty.  thm:neg (ii) predicts this arm is
               flat in eps up to finite-sample noise.

Each arm's strength is chosen on a validation set drawn like the training data (noisy
labels, N_va = 2000) and scored on clean held-out data; per-seed accuracies are stored so
that differences between arms can be given intervals paired by seed.

Collapse is decided exactly: for a one-hot nominal, each seed's threshold eps_c is computed
(an LP at r in {1, inf}, a convex programme at r = 2), strengths at or past it are not
fitted, and fits below it are warm-started along the grid, because L-BFGS started at W = 0
stops there for eps in [eps_G, eps_c) although the minimiser is not 0.

Design choices that the numbers depend on:
  * D_r and its subgradient come from core (D_rows / subgrad_rows), never a copy.
  * the class-mean matrix M is redrawn per seed and SHARED across rho and arms, so
    a seed is one problem instance seen at every noise rate.
  * l2 (default 1e-3) is a second capacity control present in every arm, reported
    in the header; l2 = 0 is supported.  The bias column is not L2-penalised.
  * Strength grids are per arm (GRIDS); the label-smoothing weight stays below 1.

    python -m lrd.noise_sweep --quick     # 4 seeds
    python -m lrd.noise_sweep             # 16 seeds -> noise_sweep.json
"""
import argparse, json, sys
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, hstack as sphstack, identity, lil_matrix, vstack as spvstack
from .core import D_rows, subgrad_rows, dual, nrm
from .paths import NOISE_SWEEP_JSON
from .prob import softmax

ARMS = [('D_1', 1), ('D_2', 2), ('D_inf', np.inf), ('ls', 'ls'), ('cp', 'cp'),
        ('prototype', 'prototype')]

# 'prototype' is not a penalty.  It scores class k by <x, mu_k>, mu_k the mean of the
# training inputs labelled k (a prototype or matched-filter rule; NOT nearest-centroid,
# which would subtract ||mu_k||^2 / 2).  It is the baseline for obs:collapse: past a
# finite threshold eps_c (collapse_threshold) the exact minimiser of every D_r arm is
# W = 0, a model that predicts the uniform distribution.  The optimiser approaches 0
# along the cross-entropy descent direction at W = 0, a frequency-weighted, centred
# version of this prototype rule, and stops at |W| ~ 1e-13; the argmax of that residue
# is what an accuracy curve reports there.  Strengths at or past a seed's exact eps_c are
# therefore not fitted, and (for the soft nominal, which has no exact threshold here) fits
# with |W|max < COLLAPSE_TOL are flagged; neither is reported as the model's accuracy.
COLLAPSE_TOL = 1e-8


def make_data(n, d, N, rho, rng, M, sep):
    """M is passed in, never drawn here: a seed is one problem instance, seen at
    every rho and by every arm."""
    y = rng.integers(0, n, N)
    X = M[y] + rng.normal(size=(N, d))
    yc = y.copy()
    flip = rng.random(N) < rho
    if flip.any():
        alt = rng.integers(0, n - 1, flip.sum())
        yc[flip] = alt + (alt >= y[flip])          # uniform over the n-1 wrong classes
    return X, y, yc


def make_data_cov(n, d, N, rho, rng, M, A):
    """Like make_data, but the class noise is correlated: X = M[y] + z A^T with
    z standard normal, so the shared covariance is Sigma = A A^T.  Same draw order as
    make_data (labels, noise, flips), used by the anisotropic second setting (V7.9)."""
    y = rng.integers(0, n, N)
    X = M[y] + rng.normal(size=(N, d)) @ A.T
    yc = y.copy()
    flip = rng.random(N) < rho
    if flip.any():
        alt = rng.integers(0, n - 1, flip.sum())
        yc[flip] = alt + (alt >= y[flip])
    return X, y, yc


def fit(X, y, n, arm, s, nominal='onehot', lam=0.0, l2=1e-3, w0=None):
    """nominal='onehot' -> p_hat = e_y.  nominal='soft' -> (1-lam) e_y + lam u.

    w0 is the starting point (default W = 0).  Started at 0, L-BFGS sees only the
    cross-entropy descent direction and can stop there for eps in [eps_G, eps_c) although
    the minimiser is not 0 (check V7.4); the sweep therefore warm-starts along the grid."""
    N, d = X.shape
    Xb = np.hstack([X, np.ones((N, 1))])
    Dd = d + 1
    Y = np.zeros((N, n)); Y[np.arange(N), y] = 1.0
    if nominal == 'soft':
        Y = (1 - lam) * Y + lam / n
    elif nominal != 'onehot':
        raise ValueError(nominal)
    if arm == 'prototype':
        W = np.zeros((Dd, n))
        W[:d] = np.array([X[y == k].mean(0) if (y == k).any() else np.zeros(d)
                          for k in range(n)]).T
        return W
    mask = np.ones((Dd, 1)); mask[-1, 0] = 0.0     # do not L2 the bias

    def obj(w):
        W = w.reshape(Dd, n)
        Z = Xb @ W
        P = softmax(Z)
        lse = np.log(np.exp(Z - Z.max(-1, keepdims=True)).sum(-1)) + Z.max(-1)
        if arm == 'ls':
            T = (1 - s) * Y + s / n
            loss = np.mean(lse - (Z * T).sum(-1)); G = (P - T) / N
        elif arm == 'cp':
            S = (P * np.log(np.maximum(P, 1e-300))).sum(-1)
            loss = np.mean(lse - (Z * Y).sum(-1) + s * S)
            G = ((P - Y) + s * P * (np.log(np.maximum(P, 1e-300)) - S[:, None])) / N
        elif arm == 'mbls':                           # MbLS with m = 0: G = u - simplex,
            jmax = Z.argmax(-1)                       # penalty sigma_G(-Z) = max(Z) - mean(Z)
            loss = np.mean(lse - (Z * Y).sum(-1) + s * (Z.max(-1) - Z.mean(-1)))
            sg = np.full((N, n), -1.0 / n); sg[np.arange(N), jmax] += 1.0
            G = ((P - Y) + s * sg) / N
        else:
            loss = np.mean(lse - (Z * Y).sum(-1) + s * D_rows(Z, arm))
            G = ((P - Y) + s * subgrad_rows(Z, arm)) / N
        Wm = W * mask
        loss += l2 * np.sum(Wm * Wm)
        return loss, (Xb.T @ G + 2 * l2 * Wm).ravel()

    x0 = np.zeros(Dd * n) if w0 is None else np.asarray(w0, float).ravel()
    res = minimize(obj, x0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=600, ftol=1e-12, gtol=1e-10))
    return res.x.reshape(Dd, n)


def evaluate(W, X, y):
    Z = np.hstack([X, np.ones((len(X), 1))]) @ W
    return float(np.mean(np.argmax(Z, -1) == y))


def _probs(W, X):
    Z = np.hstack([X, np.ones((len(X), 1))]) @ W
    Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def ece(P, y, n_bins=15):
    """Expected calibration error: |confidence - accuracy| averaged over confidence bins."""
    conf = P.max(1); correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(y)
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            e += m.sum() / N * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def evaluate_full(W, X, y, n_bins=15):
    """Held-out accuracy, log-loss and ECE for a fitted W."""
    P = _probs(W, X)
    acc = float((P.argmax(1) == y).mean())
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))
    return acc, ll, ece(P, y, n_bins)


def predicted_radius(n, rho, r):
    """The displacement of the SOFT nominal from the truth, per arm (thm:neg)."""
    lam = n * rho / (n - 1)
    p = np.zeros(n); p[0] = 1.0
    return lam * nrm(np.ones(n) / n - p, dual(r))


def bootstrap_argmax(acc, grid, B=4000, rng=None):
    """acc: (n_seeds, n_grid). Resample seeds; report the eps* interval.

    The point of reporting an INTERVAL rather than a grid argmax: if the curve is
    flat near its optimum, the interval is wide, and 'eps* does not track rho'
    is then a statement about the data rather than about the estimator.
    """
    rng = rng or np.random.default_rng(0)
    ns = acc.shape[0]
    stars = []
    for _ in range(B):
        idx = rng.integers(0, ns, ns)
        m = np.nanmean(acc[idx], axis=0)
        stars.append(grid[int(np.nanargmax(m))])
    stars = np.array(stars)
    return (float(np.nanmean(np.nanmean(acc, 0))), float(grid[int(np.nanargmax(np.nanmean(acc, 0)))]),
            float(np.quantile(stars, .05)), float(np.quantile(stars, .95)))


def collapse_threshold(X, y, n, r):
    """The exact eps_c past which W = 0 minimises the D_r-penalised objective, r in {1, inf}.

    The objective is convex in W, so W = 0 is optimal iff the cross-entropy gradient
    at 0 lies in eps times the penalty's subdifferential there:
        G = eps * Xb' Gamma / N,   each row of Gamma sum-zero with dual norm <= 1
    (the L2 term has zero gradient at 0).  eps_c is the gauge of G in that set: a
    linear programme at r in {1, inf}, where the dual norm is polyhedral.
    """
    N = len(X)
    Xb = np.hstack([X, np.ones((N, 1))]); Dd = Xb.shape[1]
    Y = np.eye(n)[y]
    G = -(Xb.T @ (np.ones((N, n)) / n - Y)) / N
    m = N * n
    A1 = lil_matrix((Dd * n, m))
    for j in range(Dd):
        for k in range(n):
            A1[j * n + k, np.arange(N) * n + k] = Xb[:, j] / N
    A2 = lil_matrix((N, m))
    for i in range(N):
        A2[i, i * n:(i + 1) * n] = 1
    beq = np.concatenate([G.ravel(), np.zeros(N)])
    if r == 1:                                    # dual norm l_inf: |gamma_ik| <= s
        Aeq = spvstack([sphstack([A1, csr_matrix((Dd * n, 1))]),
                        sphstack([A2, csr_matrix((N, 1))])]).tocsr()
        I = identity(m, format='csr'); one = csr_matrix(np.ones((m, 1)))
        Aub = spvstack([sphstack([I, -one]), sphstack([-I, -one])]).tocsr()
        c = np.zeros(m + 1); c[-1] = 1
        res = linprog(c, A_ub=Aub, b_ub=np.zeros(2 * m), A_eq=Aeq, b_eq=beq,
                      bounds=[(None, None)] * m + [(0, None)], method='highs')
    elif np.isinf(r):                             # dual norm l_1: gamma = a - b, sum(a + b) <= s
        Aeq = spvstack([sphstack([A1, -A1, csr_matrix((Dd * n, 1))]),
                        sphstack([A2, -A2, csr_matrix((N, 1))])]).tocsr()
        R = lil_matrix((N, m))
        for i in range(N):
            R[i, i * n:(i + 1) * n] = 1
        Aub = sphstack([R, R, -csr_matrix(np.ones((N, 1)))]).tocsr()
        c = np.zeros(2 * m + 1); c[-1] = 1
        res = linprog(c, A_ub=Aub, b_ub=np.zeros(N), A_eq=Aeq, b_eq=beq,
                      bounds=[(0, None)] * (2 * m + 1), method='highs')
    else:
        raise ValueError("exact threshold only at r in {1, inf}")
    if not res.success:
        raise RuntimeError(f"collapse LP failed: {res.message}")
    return float(res.fun)


def collapse_threshold_2(X, y, n):
    """The exact eps_c at r = 2, where the dual norm is not polyhedral.

    With G the cross-entropy descent direction at W = 0 (as in collapse_threshold), W = 0 is
    optimal iff <G, W> <= eps * mean_i D_2((Xb W)_i) for every W, so
        eps_c = sup_W <G, W> / mean_i D_2((Xb W)_i) = 1 / min { mean_i D_2((Xb W)_i) : <G, W> = 1 },
    a convex programme (SLSQP; D_2 is smooth wherever no row of Xb W is constant)."""
    N = len(X)
    Xb = np.hstack([X, np.ones((N, 1))]); Dd = Xb.shape[1]
    G = -(Xb.T @ (np.ones((N, n)) / n - np.eye(n)[y])) / N
    g = G.ravel() / np.dot(G.ravel(), G.ravel())

    def f(w):
        Z = Xb @ w.reshape(Dd, n)
        return float(D_rows(Z, 2).mean()), (Xb.T @ subgrad_rows(Z, 2)).ravel() / N
    res = minimize(f, g, jac=True, method='SLSQP',
                   constraints=[dict(type='eq', fun=lambda w: float(G.ravel() @ w) - 1.0,
                                     jac=lambda w: G.ravel())],
                   options=dict(maxiter=2000, ftol=1e-14))
    if not res.success:
        raise RuntimeError(f"collapse programme failed: {res.message}")
    return 1.0 / float(res.fun)


def exact_threshold(X, y, n, r):
    return collapse_threshold_2(X, y, n) if r == 2 else collapse_threshold(X, y, n, r)


def load_sweep(path=NOISE_SWEEP_JSON):
    """The frozen sweep output shipped with the notebooks (written by main())."""
    with open(path) as fh:
        return json.load(fh)


# Strength grids, per arm.  The D_r grid contains every point of the original 14-point
# grid (geomspace(0.02, 4, 14)) and one point between each pair.  The label-smoothing weight
# must stay below 1; the confidence-penalty grid is extended past 4 until its best value is
# inside the range.
BASE = np.geomspace(0.02, 4.0, 27)
GRIDS = {'D_1': BASE, 'D_2': BASE, 'D_inf': BASE,
         'ls': np.concatenate([BASE[BASE < 1.0], [0.9, 0.95, 0.99]]),
         'cp': np.concatenate([BASE, [6.0, 9.0, 13.5, 20.0, 30.0, 45.0]]),
         'prototype': np.array([])}


def _cell(job):
    """One (nominal, rho) cell: every arm, every seed, every strength."""
    nominal, rho, a = job
    n, d, seeds = a['n'], a['d'], a['seeds']
    lam = n * rho / (n - 1)
    out = {}
    for name, arm in ARMS:
        grid = np.concatenate([[0.0], GRIDS[name]])
        out[name] = dict(grid=grid, te=np.full((seeds, len(grid)), np.nan),
                         va=np.full((seeds, len(grid)), np.nan),
                         wmax=np.full((seeds, len(grid)), np.nan),
                         collapsed=np.zeros((seeds, len(grid)), bool), eps_c=[None] * seeds)
    for si in range(seeds):
        rng = np.random.default_rng(1000 + si)
        M = rng.normal(size=(n, d)) * a['sep']
        Xtr, ytr, yc = make_data(n, d, a['ntr'], rho, rng, M, a['sep'])
        Xte, yte, _ = make_data(n, d, a['nte'], 0.0, rng, M, a['sep'])
        Xva, _, yva = make_data(n, d, a['nva'], rho, rng, M, a['sep'])   # noisy, like training
        for name, arm in ARMS:
            o = out[name]
            ec = None
            if isinstance(arm, (int, float)) and nominal == 'onehot':
                ec = o['eps_c'][si] = exact_threshold(Xtr, yc, n, arm)
            W = None
            for gi, s in enumerate(o['grid']):
                if name == 'prototype' and gi > 0:
                    break
                if ec is not None and s >= ec:          # exact: the minimiser is W = 0
                    o['collapsed'][si, gi:] = True
                    o['wmax'][si, gi:] = 0.0
                    break
                W = fit(Xtr, yc, n, arm, float(s), nominal=nominal, lam=lam, l2=a['l2'], w0=W)
                o['te'][si, gi] = evaluate(W, Xte, yte)
                o['va'][si, gi] = evaluate(W, Xva, yva)
                o['wmax'][si, gi] = float(np.abs(W).max())
                o['collapsed'][si, gi] = ec is None and o['wmax'][si, gi] < COLLAPSE_TOL
    return nominal, rho, out


def summarise(o):
    """Mean curves over the live strengths (no seed collapsed), the validation choice of
    strength, and the test accuracy there, per seed."""
    grid, te, va, wmax = o['grid'], o['te'], o['va'], o['wmax']
    collapsed = o['collapsed']
    dead = collapsed.any(0)
    live_te = te.copy(); live_te[:, dead] = np.nan
    live_va = va.copy(); live_va[:, dead] = np.nan
    m_te, m_va = np.nanmean(live_te, 0), np.nanmean(live_va, 0)
    gv = int(np.nanargmax(m_va)); gt = int(np.nanargmax(m_te))
    first = next((float(grid[g]) for g in range(len(grid)) if collapsed[:, g].any()), None)
    return dict(grid=grid.tolist(), curve=m_te.tolist(), curve_val=m_va.tolist(),
                star=float(grid[gt]), best=float(m_te[gt]), base=float(m_te[0]),
                val_star=float(grid[gv]), val_star_index=gv,
                test_at_val=live_te[:, gv].tolist(), val_star_at_edge=bool(gv == np.flatnonzero(~dead)[-1]),
                first_collapse=first, collapsed_frac=collapsed.mean(0).tolist(),
                eps_c=o['eps_c'], per_seed=te.tolist(), per_seed_val=va.tolist(),
                per_seed_wmax=wmax.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--d', type=int, default=20)
    ap.add_argument('--ntr', type=int, default=400)
    ap.add_argument('--nte', type=int, default=10000)
    ap.add_argument('--nva', type=int, default=2000)
    ap.add_argument('--sep', type=float, default=0.55)
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default=str(NOISE_SWEEP_JSON))
    ap.add_argument('--seeds', type=int, default=None,
                    help='override; --quick sets 4, default 16')
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else (4 if a.quick else 16)
    cfg = dict(n=a.n, d=a.d, ntr=a.ntr, nte=a.nte, nva=a.nva, sep=a.sep, l2=a.l2, seeds=seeds)
    rhos = [0.0, 0.1, 0.2, 0.4]
    print(f"# {cfg}")
    jobs = [(nominal, rho, cfg) for nominal in ['onehot', 'soft'] for rho in rhos]
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        cells = pool.map(_cell, jobs)
    out = dict(cfg, grids={k: v.tolist() for k, v in GRIDS.items()}, res={})
    for nominal, rho, o in cells:
        out['res'][f"{nominal}|{rho}"] = {name: summarise(o[name]) if name != 'prototype' else
                                          dict(best=float(np.mean(o[name]['te'][:, 0])),
                                               per_seed=o[name]['te'][:, 0].tolist())
                                          for name, _ in ARMS}
        R = out['res'][f"{nominal}|{rho}"]
        print(f"{nominal} rho={rho}: " + '  '.join(
            f"{k} {v['best']:.3f}" + (f"@{v['star']:.3g} val@{v['val_star']:.3g}" if 'star' in v else '')
            for k, v in R.items()))
    with open(a.out, 'w') as fh:
        json.dump(out, fh)
    print(f"wrote {a.out}")


if __name__ == '__main__':
    sys.exit(main())
