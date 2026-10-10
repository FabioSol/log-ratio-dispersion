"""Every check in Appendix B passes.  This is the one command that confirms every
number quoted in the thesis still reproduces:

    pytest tests/test_checks.py --runslow
"""
import pytest

from lrd.checks import REGISTRY

CASES = [pytest.param(c, id=cid, marks=[pytest.mark.slow] if c.slow else [])
         for cid, c in REGISTRY.items()]


@pytest.mark.parametrize('chk', CASES)
def test_check_passes(chk):
    res = chk()
    assert res.passed, '\n' + res.report()


def test_registry_covers_appendix_b():
    expected = ({f'V3.{i}' for i in range(1, 10)} | {f'V4.{i}' for i in range(1, 10)}
                | {f'V5.{i}' for i in range(1, 8)} | {f'V6.{i}' for i in range(1, 15)}
                | {f'V7.{i}' for i in range(1, 10)} | {'VA.1', 'VB.1', 'VB.2'})
    assert set(REGISTRY) == expected


def test_result_report_names_the_check():
    res = REGISTRY['V3.3']()
    assert res.id == 'V3.3' and res.report().startswith('PASS  V3.3')
