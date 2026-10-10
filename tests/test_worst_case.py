"""Tests for lrd.worst_case: exact ground truth for the adversary's problem."""
import numpy as np
import pytest
from scipy.optimize import minimize

from lrd import worst_case as wc
from lrd.core import D, dual
from lrd.worst_case import LPFailure, inner_p2, proj_simplex, worst_case, worst_case_LP


def slsqp_worst_case(ph, f, eps, p):
    """An independent (inexact) solve of max <q,f> over the ball in the simplex, or
    None when SLSQP reports failure (it can then end slightly outside the ball)."""
    n = len(ph)
    res = minimize(lambda q: -(q @ f), ph.copy(), method='SLSQP',
                   constraints=[dict(type='eq', fun=lambda q: q.sum() - 1),
                                dict(type='ineq', fun=lambda q: eps - np.linalg.norm(q - ph, p))],
                   bounds=[(0, 1)] * n, options=dict(ftol=1e-14, maxiter=500))
    return -res.fun if res.success else None


def test_proj_simplex_lands_on_the_simplex(rng):
    for _ in range(50):
        q = proj_simplex(rng.normal(size=6) * 3)
        assert q.min() >= 0 and q.sum() == pytest.approx(1.0)


def test_proj_simplex_fixes_simplex_points(rng):
    x = rng.dirichlet(np.ones(5))
    np.testing.assert_allclose(proj_simplex(x), x, atol=1e-15)


def test_inner_p2_matches_an_independent_solver(rng):
    for _ in range(40):
        n = int(rng.integers(3, 7)); ph = rng.dirichlet(np.ones(n))
        f = rng.normal(size=n) * 2; eps = rng.uniform(.01, .6)
        val, q, cert = inner_p2(ph, f, eps)
        assert cert['ok']
        assert val == pytest.approx(q @ f)
        ref = slsqp_worst_case(ph, f, eps, 2)
        assert ref is None or val >= ref - 1e-7


def test_inner_p2_special_cases():
    ph = np.array([.5, .3, .2])
    val, q, cert = inner_p2(ph, np.array([3., 1., 0.]), 2.0)          # ball holds a vertex
    assert cert['case'] == 'vertex' and val == 3.0
    val, q, cert = inner_p2(ph, np.full(3, 1.5), 0.1)                 # constant f
    assert cert['case'] == 'constant f' and val == pytest.approx(1.5)


@pytest.mark.parametrize('p', [1, np.inf])
def test_LP_matches_an_independent_solver(p, rng):
    for _ in range(20):
        n = int(rng.integers(3, 6)); ph = rng.dirichlet(np.ones(n))
        f = rng.normal(size=n); eps = rng.uniform(.01, .4)
        ref = slsqp_worst_case(ph, f, eps, p)
        assert ref is None or worst_case_LP(ph, f, eps, p) >= ref - 1e-6


@pytest.mark.parametrize('p', [1, 2, np.inf])
def test_closed_form_bound_is_valid(p, rng):
    """lem:upperbound: <ph, f> + eps * D_{p*}(f) >= W(f)."""
    for _ in range(50):
        n = int(rng.integers(3, 8)); ph = rng.dirichlet(np.ones(n))
        f = rng.normal(size=n) * 2; eps = rng.uniform(.005, .5)
        assert ph @ f + eps * D(f, dual(p)) >= worst_case(ph, f, eps, p) - 1e-8


def test_worst_case_dispatches(rng):
    ph = rng.dirichlet(np.ones(4)); f = rng.normal(size=4)
    assert worst_case(ph, f, .2, 1) == worst_case_LP(ph, f, .2, 1)
    assert worst_case(ph, f, .2, np.inf) == worst_case_LP(ph, f, .2, np.inf)
    assert worst_case(ph, f, .2, 2) == inner_p2(ph, f, .2)[0]


def test_LP_at_p2_is_refused():
    with pytest.raises(ValueError):
        worst_case_LP(np.ones(3) / 3, np.arange(3.), .1, 2)


def test_failed_LP_raises_instead_of_returning_nan(monkeypatch):
    """A nan here would make |nan - bound| < tol False: a failed solve recorded as a
    genuine non-tight case."""
    class Failed:
        success = False
        message = 'forced failure'
    monkeypatch.setattr(wc, 'linprog', lambda *a, **k: Failed())
    with pytest.raises(LPFailure):
        worst_case_LP(np.ones(3) / 3, np.arange(3.), .1, 1)


def test_failed_certificate_raises(monkeypatch):
    monkeypatch.setattr(wc, 'inner_p2', lambda *a, **k: (0.0, None, dict(ok=False)))
    with pytest.raises(LPFailure):
        worst_case(np.ones(3) / 3, np.arange(3.), .1, 2)
