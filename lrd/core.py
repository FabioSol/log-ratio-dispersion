"""Core primitives for log-ratio dispersion.

    D_r(v)       = min_c ||v - c*1||_r                  (lem:keystone, eq:Ddef)
    centre(v, r) = argmin_c ||v - c*1||_r               (cor:centre, the r-centre)
    subgrad      = a canonical element of dD_r(v)       (cor:subgrad)
    G            = {g : ||g||_{r*} <= 1, 1'g = 0}       (eq:Gset)

Every function is degeneracy-safe by construction.  The one input class where a
naive implementation fails -- v in or near span{1} -- is exactly where cor:subgrad
says dD_r(v) = G and where prop:saturation says every solution path ends.  The
guard is therefore the theory's boundary case, not a numerical patch, and it is
applied uniformly at every r.  Do not simplify the guards: each one is pinned by
a regression test in tests/test_core.py.

DEGEN_TOL is relative to the scale of v.  An absolute tolerance is wrong here:
zeta coordinates are logits and are not O(1).

The row-wise forms at the bottom (centre_rows, D_rows, subgrad_rows) serve the
training sweep.  They must equal the scalar forms row by row; check VB.2 and
tests/test_core.py assert it, so the two cannot drift apart.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import brentq, linprog

DEGEN_TOL = 1e-12


def dual(r):
    """r* with 1/r + 1/r* = 1."""
    if r == 1:
        return np.inf
    if np.isinf(r):
        return 1.0
    return r / (r - 1.0)


def nrm(v, p):
    v = np.asarray(v, float)
    return np.max(np.abs(v)) if np.isinf(p) else np.sum(np.abs(v) ** p) ** (1.0 / p)


def _scale(v):
    """Magnitude against which ptp(v) is judged small."""
    return max(1.0, float(np.max(np.abs(v))))


def is_degenerate(v, tol=DEGEN_TOL):
    """True iff v is in or numerically near span{1}, where dD_r(v) = G."""
    v = np.asarray(v, float)
    return bool(np.ptp(v) <= tol * _scale(v))


def centre(v, r):
    """argmin_c ||v - c*1||_r.  At r = 1 and even n this returns the midpoint of
    the median interval (numpy's convention); D_1 and dD_1 do not depend on the
    choice (cor:singleton), and subgrad below does not assume any particular one."""
    v = np.asarray(v, float)
    if is_degenerate(v):
        return float(v.mean())
    if r == 1:
        return float(np.median(v))
    if r == 2:
        return float(v.mean())
    if np.isinf(r):
        return 0.5 * (float(v.max()) + float(v.min()))
    # The centre equation (cor:centre).  f is continuous and strictly decreasing, but
    # for 1 < r < 2 its derivative is INFINITE wherever c meets a coordinate, and on
    # tied data the root very often sits exactly at a coordinate.  Handing that to
    # brentq gives |f(c*)| ~ 1e-8 (sqrt of machine epsilon, the signature of a root at
    # a vertical tangent) and hence sum(g) ~ 1e-8.  So test the coordinates as
    # candidate roots first, and only bracket between them.
    f = lambda c: np.sum(np.abs(v - c) ** (r - 1) * np.sign(v - c))
    nodes = np.unique(v)
    fn = np.array([f(c) for c in nodes])
    mag = np.array([np.sum(np.abs(v - c) ** (r - 1)) for c in nodes])
    hit = np.abs(fn) <= 1e-11 * np.maximum(mag, 1e-300)
    if hit.any():
        return float(nodes[int(np.argmax(hit))])
    j = int(np.flatnonzero(fn > 0)[-1])              # fn decreasing; root in (j, j+1)
    c = float(brentq(f, nodes[j], nodes[j + 1], xtol=1e-15, rtol=8.9e-16))
    # For r close to 1 the root can land within an ulp of a node while |f| there is
    # still ~1e-4, because f has a vertical tangent.  Snap; subgrad recovers the
    # near-node entry from the sum-zero condition instead of from |w_i|^{r-1}.
    i0 = int(np.argmin(np.abs(nodes - c)))
    return float(nodes[i0]) if abs(nodes[i0] - c) <= 1e-9 * _scale(v) else c


def D(v, r):
    """D_r(v) = min_c ||v - c*1||_r."""
    v = np.asarray(v, float)
    if not np.all(np.isfinite(v)):
        # log of a zero coordinate.  lem:barrier says D_r -> inf there; returning nan
        # would make it read as a failed comparison instead.
        raise ValueError("D_r received a non-finite v (log of a zero coordinate?)")
    return float(nrm(v - centre(v, r), r))


def subgrad(v, r, tol=DEGEN_TOL):
    """A canonical g in dD_r(v).  Guarantees (cor:subgrad), on EVERY input:
         sum(g) = 0,   ||g||_{r*} <= 1,   <g, v> = D_r(v),
       with ||g||_{r*} = 1 whenever v is not in span{1}.

    Where dD_r(v) is set-valued this returns one element (a vertex at
    r in {1, inf}).  prop:tightness quantifies over the whole face, so use
    face_feasible() for the exact test; this function gives the O(n) sufficient one.
    """
    v = np.asarray(v, float)
    n = len(v)
    if is_degenerate(v, tol):        # dD_r(v) = G, and 0 in G
        return np.zeros(n)

    if np.isinf(r):
        g = np.zeros(n)
        i, j = int(np.argmax(v)), int(np.argmin(v))
        g[i] += 0.5
        g[j] -= 0.5                  # += / -= , so i == j would cancel, not clobber
        return g

    c = centre(v, r)
    w = v - c
    Z = np.flatnonzero(w == 0.0)                 # coordinates sitting on the centre

    # Unnormalised subgradient.  Off Z it is |w|^{r-1} sgn(w) (which is sgn(w) at r=1).
    # On Z the formula gives 0, and 0 is wrong unless the rest already sums to zero:
    #   r = 1       -- dD_1 is set-valued on Z; sum-zero pins the free entries.
    #   1 < r < inf -- dD_r is a singleton (cor:singleton); the true root is within an
    #                  ulp of the node and the true entry is exactly the deficit.
    #                  Recovering it from sum-zero is stable; recovering it from
    #                  |w_i|^{r-1} is not (|sum g| up to 1e-4 at r = 1.2).
    # Either way the deficit splits equally over Z, by symmetry of the coordinates in Z.
    h = np.abs(w) ** (r - 1) * np.sign(w)
    if len(Z):
        h[Z] = 0.0
        h[Z] = -h.sum() / len(Z)

    # ||h||_{r*} = ||w||_r^{r-1} in exact arithmetic, and is the better-conditioned
    # of the two expressions.  Normalising by it makes ||g||_{r*} = 1 to machine eps.
    return h / nrm(h, dual(r))


def face(v, r, tol=1e-12):
    """Vertices whose convex hull is dD_r(v), at r in {1, inf}.

    Used by face_feasible for prop:tightness's exact (existential) test.  Only
    r in {1, inf} can be set-valued away from span{1} (cor:singleton).
    """
    v = np.asarray(v, float)
    n = len(v)
    if is_degenerate(v):
        raise ValueError("dD_r(v) = G; the face is all of G, handle separately")
    if np.isinf(r):
        hi = np.flatnonzero(v >= v.max() - tol * _scale(v))
        lo = np.flatnonzero(v <= v.min() + tol * _scale(v))
        return [0.5 * (np.eye(n)[i] - np.eye(n)[j]) for i in hi for j in lo]
    if r == 1:
        c = centre(v, r)
        w = v - c
        Z = np.flatnonzero(np.abs(w) <= tol * _scale(v))
        s = np.sign(w)
        s[Z] = 0.0
        if len(Z) == 0:
            return [s]
        k, target = len(Z), -s.sum()
        verts = []
        # vertices of {z in [-1,1]^k : sum z = target}: every entry +-1 except at most one (j),
        # which carries the fractional remainder.  The parity-matched barycentre fallback was wrong
        # for odd n with ties, where the true vertices have one fractional entry (face bug fix).
        for j in [None] + list(range(k)):
            idx = [i for i in range(k) if i != j]
            for m in range(len(idx) + 1):          # m of idx at +1, the rest at -1
                rest = target - (2 * m - len(idx)) # value entry j must carry
                if j is None:
                    if abs(rest) > 1e-9:
                        continue
                elif not (-1 - 1e-12 <= rest <= 1 + 1e-12):
                    continue
                for pick in combinations(idx, m):
                    z = -np.ones(k)
                    z[list(pick)] = 1.0
                    if j is not None:
                        z[j] = rest
                    g = s.copy()
                    g[Z] = z
                    verts.append(g)
        return verts
    raise ValueError("dD_r is a singleton for 1 < r < inf away from span{1}")


def face_feasible(ph, v, eps, r):
    """prop:tightness, exactly: does some g in dD_r(v) have ph + eps*g >= 0 ?

    The criterion is existential over the face.  subgrad() returns one vertex, so
    `(ph + eps*subgrad(v,r)).min() >= 0` is sufficient but not necessary -- that is
    the tied-regime counterexample of check V5.2.  Without exact ties the two agree.
    """
    ph = np.asarray(ph, float)
    v = np.asarray(v, float)
    if is_degenerate(v):
        return True                  # 0 in dD_r(v) = G, and ph >= 0
    if 1 < r < np.inf:
        return bool((ph + eps * subgrad(v, r)).min() >= -1e-12)
    verts = face(v, r)
    if any((ph + eps * g).min() >= -1e-12 for g in verts):
        return True
    # convex combinations can be feasible where no vertex is: solve the LP
    V = np.array(verts)                                   # (K, n)
    K = len(V)
    res = linprog(np.zeros(K),
                  A_ub=-(eps * V.T), b_ub=ph,
                  A_eq=np.ones((1, K)), b_eq=[1.0],
                  bounds=[(0, None)] * K, method='highs')
    return bool(res.success)


# ---------------------------------------------------------------------------
# Row-wise vectorised forms, for the training sweep (noise_sweep.py).  They carry
# every guard of the scalar forms; VB.2 asserts they agree row by row.
# ---------------------------------------------------------------------------

def centre_rows(V, r):
    V = np.atleast_2d(np.asarray(V, float))
    if r == 1:
        return np.median(V, axis=-1, keepdims=True)
    if r == 2:
        return V.mean(-1, keepdims=True)
    if np.isinf(r):
        return 0.5 * (V.max(-1, keepdims=True) + V.min(-1, keepdims=True))
    return np.array([[centre(v, r)] for v in V])


def _degenerate_rows(V, tol=DEGEN_TOL):
    scale = np.maximum(1.0, np.max(np.abs(V), axis=-1))
    return np.ptp(V, axis=-1) <= tol * scale


def D_rows(V, r):
    V = np.atleast_2d(np.asarray(V, float))
    W = V - centre_rows(V, r)
    out = (np.max(np.abs(W), -1) if np.isinf(r)
           else np.sum(np.abs(W) ** r, -1) ** (1.0 / r))
    return np.where(_degenerate_rows(V), 0.0, out)


def subgrad_rows(V, r, tol=DEGEN_TOL):
    """Row-wise canonical subgradient, carrying every guard in subgrad()."""
    V = np.atleast_2d(np.asarray(V, float))
    m, n = V.shape
    live = ~_degenerate_rows(V, tol)
    G = np.zeros((m, n))
    if not live.any():
        return G
    Vl = V[live]
    k = Vl.shape[0]
    if np.isinf(r):
        g = np.zeros((k, n))
        rows = np.arange(k)
        g[rows, np.argmax(Vl, -1)] += 0.5
        g[rows, np.argmin(Vl, -1)] -= 0.5   # += / -= so i == j cancels, not clobbers
        G[live] = g
        return G
    W = Vl - centre_rows(Vl, r)
    H = np.abs(W) ** (r - 1) * np.sign(W)
    Z = (W == 0.0)
    if Z.any():
        H = np.where(Z, 0.0, H)
        nz = Z.sum(-1)
        fill = np.zeros(k)
        has = nz > 0
        fill[has] = -H.sum(-1)[has] / nz[has]
        H = np.where(Z, fill[:, None], H)
    p = dual(r)
    den = (np.max(np.abs(H), -1, keepdims=True) if np.isinf(p)
           else np.sum(np.abs(H) ** p, -1, keepdims=True) ** (1.0 / p))
    G[live] = H / np.maximum(den, 1e-300)
    return G
