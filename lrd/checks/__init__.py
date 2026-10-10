"""The thesis's verification checks (Appendix B, app:index), one definition each.

Every quantitative claim in the thesis is produced by one check here.  A check is
a zero-argument function (fixed seeds, so its numbers are reproducible) that
returns a Result: an explicit pass/fail, a one-line summary matching Appendix B,
the printed details, the metrics, and any arrays a figure needs.

    from lrd.checks import path
    res = path.v4_8()          # a Result; displays as its report in a notebook
    res.passed, res.metrics, res.data

The notebooks call these and draw the figures; tests/test_checks.py runs every
registered check and asserts it passes.  Checks registered with slow=True take
minutes and only run under `pytest --runslow`.
"""
from dataclasses import dataclass, field, replace

# Tolerances.  Each pass criterion uses one of these or states its own reason.
#
# SOLVER_TOL: any comparison against the reference solver (lrd.solver).  Its observed
#   floor is 1e-9 to 6e-9 (V3.6, V4.4, V4.5, V3.7, V7.2); 1e-7 leaves one to two orders
#   of margin, and a genuine failure of any claim checked this way sits far above it.
# EXACT_TOL: identities that hold in exact arithmetic and involve no optimiser.
# TIE_MARGIN: a coordinate gap below this could not be told from a tie at solver
#   precision (V5.5).
SOLVER_TOL = 1e-7
EXACT_TOL = 1e-12
TIE_MARGIN = 10 * SOLVER_TOL


@dataclass(repr=False)
class Result:
    passed: bool
    summary: str
    details: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    id: str = ''
    title: str = ''
    ref: str = ''

    def report(self):
        ref = f'  [{self.ref}]' if self.ref else ''
        lines = [f"{'PASS' if self.passed else 'FAIL'}  {self.id}: {self.title}{ref}"]
        lines += ['    ' + line for line in self.details]
        lines.append(f'  => {self.summary}')
        return '\n'.join(lines)

    __str__ = __repr__ = report


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    ref: str
    fn: object
    slow: bool

    def __call__(self, **kwargs):
        res = self.fn(**kwargs)
        return replace(res, id=self.id, title=self.title, ref=self.ref)


REGISTRY = {}


def check(id, title, ref='', slow=False):
    """Register a check under its Appendix B id (e.g. 'V3.2')."""
    def deco(fn):
        c = Check(id, title, ref, fn, slow)
        REGISTRY[id] = c
        return c
    return deco


def run(id, **kwargs):
    return REGISTRY[id](**kwargs)


from . import keystone, computation, path, validity, calibration, labels  # noqa: E402  (populate REGISTRY)
