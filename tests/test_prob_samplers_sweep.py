"""Tests for the small modules: lrd.prob, lrd.samplers, lrd.noise_sweep."""
import numpy as np
import pytest

from lrd import noise_sweep
from lrd.core import dual, is_degenerate, nrm
from lrd.prob import entropy, kl, logsumexp, softmax
from lrd.samplers import NOMS, PH5, degenerate_row, degenerate_sampler


# --- prob ------------------------------------------------------------------------

def test_softmax_vector_and_rows(rng):
    Z = rng.normal(size=(4, 6)) * 50
    P = softmax(Z)
    np.testing.assert_allclose(P.sum(-1), 1.0)
    np.testing.assert_allclose(P[2], softmax(Z[2]))


def test_logsumexp_is_stable():
    assert logsumexp(np.array([1000., 1000.])) == pytest.approx(1000 + np.log(2))


def test_kl():
    p = np.array([.5, .5, 0.])
    assert kl(p, p) == 0.0
    assert kl(p, np.array([1., 0., 0.])) == np.inf
    assert kl(p, np.array([.25, .25, .5])) == pytest.approx(np.log(2))


def test_entropy():
    assert entropy(np.ones(4) / 4) == pytest.approx(np.log(4))
    assert entropy(np.array([1., 0., 0.])) == pytest.approx(0.0, abs=1e-12)


# --- samplers --------------------------------------------------------------------

def test_fixed_nominals_are_sorted_compositions():
    for ph in [PH5] + NOMS:
        assert ph.sum() == pytest.approx(1.0)
        assert np.all(np.diff(ph) <= 0)


def test_sampler_reaches_the_boundary():
    """The point of the sampler: constant and near-constant inputs must occur."""
    rng = np.random.default_rng(0)
    draws = [degenerate_sampler(rng) for _ in range(400)]
    assert sum(is_degenerate(v) for v in draws) > 50
    assert {len(v) for v in draws} == {3, 4, 5, 6, 7, 8}
    assert all(len(degenerate_row(rng, 9)) == 9 for _ in range(50))


# --- noise_sweep -----------------------------------------------------------------

def _small_problem(rho, seed=0):
    rng = np.random.default_rng(seed)
    n, d = 4, 5
    M = rng.normal(size=(n, d)) * 1.5
    return n, noise_sweep.make_data(n, d, 400, rho, rng, M, 1.5)


def test_make_data_flips_to_a_wrong_class():
    n, (X, y, yc) = _small_problem(0.3)
    flipped = y != yc
    assert 0.2 < flipped.mean() < 0.4
    assert np.all(yc[flipped] != y[flipped]) and yc.max() < n


def test_fit_learns_on_clean_data():
    n, (X, y, yc) = _small_problem(0.0)
    W = noise_sweep.fit(X, yc, n, 2, 0.0)
    assert noise_sweep.evaluate(W, X, y) > 0.8


@pytest.mark.parametrize('arm', [1, 2, np.inf])
def test_strong_penalty_collapses_the_weights(arm):
    """obs:collapse: past a threshold the D_r arms drive W to zero."""
    n, (X, y, yc) = _small_problem(0.2)
    assert np.abs(noise_sweep.fit(X, yc, n, arm, 50.0)).max() < 1e-8


def test_predicted_radius_matches_the_label_noise_identity():
    n, rho = 10, 0.2
    lam = n * rho / (n - 1)
    e = np.zeros(n); e[0] = 1
    for r in [1, 2, np.inf]:
        assert noise_sweep.predicted_radius(n, rho, r) == pytest.approx(
            lam * nrm(np.ones(n) / n - e, dual(r)))


def test_r2_threshold_brackets_the_collapse():
    """collapse_threshold_2: below eps_c a nonzero W beats W = 0 (warm start from the
    unpenalised fit); at 1.05 eps_c the fit from W = 0 stays at 0."""
    n, (X, y, yc) = _small_problem(0.2)
    ec = noise_sweep.collapse_threshold_2(X, yc, n)
    W0 = noise_sweep.fit(X, yc, n, 2, 0.0)
    assert np.abs(noise_sweep.fit(X, yc, n, 2, 0.9 * ec, w0=W0)).max() > 1e-4
    assert np.abs(noise_sweep.fit(X, yc, n, 2, 1.05 * ec)).max() < 1e-8


def test_frozen_sweep_has_every_cell():
    sw = noise_sweep.load_sweep()
    for nominal in ['onehot', 'soft']:
        for rho in [0.0, 0.1, 0.2, 0.4]:
            cell = sw['res'][f'{nominal}|{rho}']
            assert set(cell) == {name for name, _ in noise_sweep.ARMS}
            for name, grid in noise_sweep.GRIDS.items():
                if name != 'prototype':
                    assert len(cell[name]['curve']) == len(grid) + 1
                    assert len(cell[name]['per_seed']) == sw['seeds']
    for name in ['D_1', 'D_2', 'D_inf']:
        assert all(e is not None for e in sw['res']['onehot|0.0'][name]['eps_c'])
