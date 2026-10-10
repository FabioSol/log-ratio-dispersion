"""Tests for lrd.solver: the penalised solution x*(eps)."""
import numpy as np
import pytest

from lrd.core import dual, nrm, subgrad
from lrd.samplers import PH5
from lrd.solver import ones_perp_basis, saturation_radius, solve

from conftest import R_FAMILY


@pytest.mark.parametrize('n', [2, 3, 5, 10])
def test_ones_perp_basis_is_orthonormal_and_sums_to_zero(n):
    B = ones_perp_basis(n)
    assert B.shape == (n, n - 1)
    np.testing.assert_allclose(B.T @ B, np.eye(n - 1), atol=1e-12)
    np.testing.assert_allclose(B.sum(0), 0, atol=1e-12)


@pytest.mark.parametrize('r', R_FAMILY)
def test_saturation_returns_uniform_exactly(r):
    """prop:saturation: at and past eps_sat the solution is u, in closed form."""
    sat = saturation_radius(PH5, r)
    for fr in [1.0, 1.01, 3.0]:
        x, info = solve(PH5, fr * sat, r, full=True)
        assert info['case'] == 'saturated'
        assert np.all(x == 0.2)


@pytest.mark.parametrize('r', R_FAMILY)
def test_unit_speed_below_saturation(r):
    """prop:distance: ||x*(eps) - ph||_{r*} = eps below eps_sat."""
    sat = saturation_radius(PH5, r)
    for fr in [.2, .6, .95]:
        x = solve(PH5, fr * sat, r)
        assert nrm(x - PH5, dual(r)) == pytest.approx(fr * sat, abs=1e-7)


@pytest.mark.parametrize('r', R_FAMILY)
def test_solution_is_on_the_simplex(r):
    x = solve(PH5, 0.5 * saturation_radius(PH5, r), r)
    assert x.min() > 0
    assert x.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('r', [1.5, 2, 3])
def test_path_relation_at_interior_r(r):
    """prop:pathrelation: x* = ph + eps * g with g in dD_r(-log x*)."""
    eps = 0.4 * saturation_radius(PH5, r)
    x, info = solve(PH5, eps, r, full=True)
    assert info['converged'], info['message']
    assert info['b2_residual'] < 1e-7
    np.testing.assert_allclose(x, PH5 + eps * subgrad(-np.log(x), r), atol=1e-7)


@pytest.mark.parametrize('r', R_FAMILY)
def test_zero_budget_returns_the_nominal(r):
    np.testing.assert_allclose(solve(PH5, 0.0, r), PH5, atol=1e-7)


@pytest.mark.parametrize('r', [1.5, 2, np.inf])
def test_unit_tilt_equals_no_tilt(r):
    eps = 0.3 * saturation_radius(PH5, r)
    np.testing.assert_allclose(solve(PH5, eps, r, tilt=np.ones(5)), solve(PH5, eps, r), atol=1e-7)


def test_sat_slack_moves_the_closed_form_shortcut():
    sat = saturation_radius(PH5, 2)
    _, info = solve(PH5, 0.999 * sat, 2, full=True)
    assert info['case'] == 'interior'
    _, info = solve(PH5, 0.999 * sat, 2, sat_slack=0.01, full=True)
    assert info['case'] == 'saturated'
