"""Unit and regression tests for lrd.core.

The regression tests pin inputs on which an earlier implementation failed; each
one names the failure it guards against.  Do not loosen them to make a change pass.
"""
import numpy as np
import pytest

from lrd.core import (DEGEN_TOL, D, D_rows, centre, centre_rows, dual, face,
                      face_feasible, is_degenerate, nrm, subgrad, subgrad_rows)
from lrd.samplers import degenerate_row, degenerate_sampler

from conftest import R_ALL, R_FAMILY


def assert_subgrad_identities(v, r, tol=1e-9):
    """cor:subgrad: sum(g) = 0, ||g||_{r*} <= 1, <g, v> = D_r(v)."""
    g = subgrad(v, r)
    Dv = D(v, r)
    scale = max(1.0, abs(Dv), float(np.max(np.abs(v))))
    assert np.all(np.isfinite(g))
    assert abs(g.sum()) <= tol
    assert nrm(g, dual(r)) <= 1 + tol
    assert abs(g @ v - Dv) <= 1e-8 * scale


# --- dual, nrm ---------------------------------------------------------------

@pytest.mark.parametrize('r, rs', [(1, np.inf), (np.inf, 1.0), (2, 2.0), (3, 1.5), (1.5, 3.0)])
def test_dual(r, rs):
    assert dual(r) == rs


@pytest.mark.parametrize('r', [1.2, 1.5, 2, 3, 6])
def test_dual_is_an_involution(r):
    assert dual(dual(r)) == pytest.approx(r)


def test_nrm_matches_numpy(rng):
    v = rng.normal(size=7)
    for p in [1, 2, 3, np.inf]:
        assert nrm(v, p) == pytest.approx(np.linalg.norm(v, p))


# --- D_r and the centre --------------------------------------------------------

@pytest.mark.parametrize('r', R_FAMILY)
def test_D_is_invariant_along_ones(r, rng):
    v = rng.normal(size=6)
    assert D(v + 3.7, r) == pytest.approx(D(v, r), rel=1e-12)


@pytest.mark.parametrize('r', R_FAMILY)
def test_D_is_absolutely_homogeneous(r, rng):
    v = rng.normal(size=6)
    assert D(-2.5 * v, r) == pytest.approx(2.5 * D(v, r), rel=1e-12)


@pytest.mark.parametrize('r', R_FAMILY)
def test_D_is_zero_on_span_one(r):
    assert D(np.full(5, 4.2), r) == 0.0


def test_landmark_centres():
    v = np.array([0., 1., 5., 2., 7.])
    assert centre(v, 1) == 2.0                      # median
    assert centre(v, 2) == pytest.approx(3.0)       # mean
    assert centre(v, np.inf) == 3.5                 # midrange


@pytest.mark.parametrize('r', [1.2, 1.5, 3, 6])
def test_centre_solves_the_centre_equation(r, rng):
    """cor:centre: sum |v_i - c|^{r-1} sgn(v_i - c) = 0 at c = centre(v, r)."""
    v = rng.normal(size=8)
    c = centre(v, r)
    f = np.sum(np.abs(v - c) ** (r - 1) * np.sign(v - c))
    assert abs(f) <= 1e-9 * np.sum(np.abs(v - c) ** (r - 1))


def test_D_rejects_non_finite_input():
    """lem:barrier: log of a zero coordinate. nan would read as a failed comparison."""
    with pytest.raises(ValueError):
        D(np.array([-0.7, -0.7, -np.inf]), 2)


# --- degeneracy ------------------------------------------------------------

def test_degeneracy_tolerance_is_relative():
    """zeta coordinates are logits, not O(1): the tolerance must scale with |v|."""
    assert is_degenerate(np.full(4, 1e12) + np.array([0, 0.5, 0, 0]))
    assert not is_degenerate(np.array([0.0, 1e-6, 0.0, 0.0]))
    assert is_degenerate(np.array([1.0, 1.0 + 0.5 * DEGEN_TOL, 1.0]))


# --- subgradient: property test on the boundary-reaching sampler ------------------

@pytest.mark.parametrize('r', R_ALL)
def test_subgrad_identities_on_boundary_sampler(r):
    rng = np.random.default_rng(int(1000 * (r if np.isfinite(r) else 99)))
    for _ in range(300):
        assert_subgrad_identities(degenerate_sampler(rng), r)


@pytest.mark.parametrize('r', R_ALL)
def test_subgrad_is_unit_norm_off_span_one(r, rng):
    v = rng.normal(size=6)
    assert nrm(subgrad(v, r), dual(r)) == pytest.approx(1.0, abs=1e-12)


# --- subgradient: regressions -------------------------------------------------

@pytest.mark.parametrize('r', R_ALL)
def test_regression_constant_v(r):
    """Constant v: r = inf once self-overwrote to sum(g) = -1/2; 1 < r < inf gave 0/0 = nan."""
    g = subgrad(np.full(5, 2.0), r)
    assert np.all(g == 0.0)


@pytest.mark.parametrize('r', [1.2, 1.5, 2, 3, 6])
def test_regression_near_constant_v_cancellation(r, rng):
    """Near-constant v (ptp ~ 1e-14): silent catastrophic cancellation once gave
    |sum g| = 1.4e-1 with no nan to warn."""
    v = np.full(6, 0.3) + 1e-14 * rng.normal(size=6)
    assert_subgrad_identities(v, r)


@pytest.mark.parametrize('v, Dv', [(np.array([0., 0., 0., 1.]), 1.0),
                                   (np.array([5., 5., 5., 1., 1.]), 8.0)])
def test_regression_r1_tied_median(v, Dv):
    """r = 1 on tied data: np.sign(0) = 0 on the median gave sum(g) = 1 on the first
    input and <g, v> = -2 against D_1 = 8 on the second."""
    assert D(v, 1) == Dv
    assert_subgrad_identities(v, 1)


@pytest.mark.parametrize('r', [1.05, 1.2, 1.5])
def test_regression_root_at_vertical_tangent(r):
    """1 < r < 2 on integer data: the centre sits on a coordinate where the centre
    equation has infinite slope; brentq alone left |sum g| ~ 1e-8 (137/6000 draws)."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        v = rng.integers(0, 3, int(rng.integers(3, 9))).astype(float)
        assert abs(subgrad(v, r).sum()) <= 1e-10


# --- the face and the exact tightness test --------------------------------------

def test_face_feasible_counterexample_where_canonical_vertex_fails():
    """prop:tightness is existential: here no canonical vertex is feasible but the face is."""
    ph = np.array([0., .5, .3, .2]); f = np.array([0., 0., 1., 1.]); eps = .4
    assert (ph + eps * subgrad(f, np.inf)).min() < 0
    assert face_feasible(ph, f, eps, np.inf)


@pytest.mark.parametrize('r', [1, np.inf])
def test_face_vertices_are_subgradients(r):
    v = np.array([0., 0., 1., 1., 2.])
    for g in face(v, r):
        assert abs(g.sum()) <= 1e-12
        assert nrm(g, dual(r)) <= 1 + 1e-12
        assert g @ v == pytest.approx(D(v, r))


def test_face_rejects_singleton_and_degenerate_cases():
    with pytest.raises(ValueError):
        face(np.array([0., 1., 3.]), 2)
    with pytest.raises(ValueError):
        face(np.full(3, 1.0), 1)


def test_face_feasible_is_true_on_span_one():
    assert face_feasible(np.array([.2, .3, .5]), np.full(3, 7.0), 0.4, 1)


# --- row-wise forms -------------------------------------------------------------

@pytest.mark.parametrize('r', [1, 1.5, 2, 3, np.inf])
def test_rows_equal_scalar_forms(r):
    rng = np.random.default_rng(80)
    V = np.array([degenerate_row(rng, 6) for _ in range(200)])
    np.testing.assert_allclose(centre_rows(V, r)[:, 0],
                               [centre(v, r) for v in V], rtol=0, atol=1e-9)
    np.testing.assert_allclose(D_rows(V, r), [D(v, r) for v in V], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(subgrad_rows(V, r), [subgrad(v, r) for v in V], atol=1e-12)


def test_face_r1_vertices_with_ties():
    """face(v, 1) must return the full vertex set, including vertices with one fractional entry.
    Regression for the odd-n-with-ties bug where the barycentre was returned instead."""
    V = [np.round(g, 9) for g in face(np.array([0., 0., 1., 1., 2.]), 1)]
    for want in ([-1, -1, 1, 0, 1], [-1, -1, 0, 1, 1]):
        assert any(np.allclose(g, want) for g in V), want


def test_face_feasible_r1_matches_LP_on_ties():
    """At r = 1 with integer f (ties) and a zero in ph, face_feasible must equal LP tightness,
    including the odd-n fractional-vertex regime."""
    from lrd.worst_case import worst_case_LP
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.choice([3, 5, 7]))
        f = rng.integers(0, 3, size=n).astype(float)
        ph = rng.random(n); ph[rng.integers(n)] = 0.0; ph = ph / ph.sum()
        eps = float(rng.uniform(0.05, 0.6))
        tight = abs(worst_case_LP(ph, f, eps, np.inf) - (ph @ f + eps * D(f, 1))) < 1e-7
        assert face_feasible(ph, f, eps, 1) == tight
