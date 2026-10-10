# log-ratio-dispersion

[![CI](https://github.com/FabioSol/log-ratio-dispersion/actions/workflows/ci.yml/badge.svg)](https://github.com/FabioSol/log-ratio-dispersion/actions/workflows/ci.yml)

Reproducibility code for the MSc thesis on **distributionally robust log-loss over norm
balls on the label simplex**, which reduces to cross-entropy plus a dispersion penalty on the
logits, controlled by a single radius `eps`. The package `lrd` (log-ratio dispersion) implements
the construction, and every quantitative claim in the thesis is reproduced by a check in
`lrd/checks/`, listed in the thesis's verification index.

The index `r` selects the penalty `D_r(v) = min_c ||v - c*1||_r`, the dispersion of the logits about
a generalised mean: the median at `r = 1` (a box), the geometric mean at `r = 2` (a Euclidean ball),
the geometric midrange at `r = inf` (an l1 ball).

## Layout

```
lrd/                 the package
  core.py            D_r, the r-centre, the canonical subgradient, the exact face test
  solver.py          the penalised solution x*(eps) at index r
  worst_case.py      exact ground truth for the adversary: LP at r* in {1, inf}, KKT at r* = 2
  general_g.py       the construction for a general sum-zero set G (beyond norm balls)
  prob.py            softmax, logsumexp, KL, entropy
  samplers.py        fixed nominals and the degeneracy-aware sampler
  noise_sweep.py     the label-noise training sweep (writes noise_sweep.json) and the exact
                     collapse threshold  (CLI: python -m lrd.noise_sweep)
  paths.py           figure and data locations
  checks/*.py        the checks by topic, one function each, with explicit pass criteria
tests/               pytest: unit and regression tests, plus every check
noise_sweep.json     frozen output of the 16-seed sweep (about 15 min to regenerate)
```

Each check returns a `Result` that carries its pass/fail, a summary, and the numbers behind it:

```python
from lrd.checks import path
res = path.v4_8()
res.passed, res.metrics, res.data
```

## Running

With Python 3.12 and `pip install -r requirements.txt`:

```
pytest                              # unit tests + the fast checks (about 25 s)
pytest --runslow                    # + the minutes-long checks: every number in the thesis (hours)
pytest tests/test_checks.py -k V4.5 # one check
python -m lrd.noise_sweep           # regenerate noise_sweep.json
```

Continuous integration runs the fast suite on every push; the slow checks (`--runslow`) are run
locally for the full verification.

## Requirements

`numpy`, `scipy`, `matplotlib`, `pytest`. `requirements.txt` pins the versions the quoted results
were produced with (Python 3.12); `requirements-lock.txt` is the full lockfile.

## Notebooks

One notebook per topic runs its checks and draws the thesis figures. They are added separately,
once a clean full re-run has refreshed their outputs.

## Citation

This is the code behind the thesis; cite the thesis (title pending the programme's approval) and
this repository's tagged release.
