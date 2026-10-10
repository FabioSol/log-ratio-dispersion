import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False,
                     help='also run the slow thesis checks (several minutes)')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip = pytest.mark.skip(reason='slow check: use --runslow')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


# The indices exercised everywhere: both endpoints, the landmark r = 2, and
# interior members on either side of it (1 < r < 2 is where centre() is fragile).
R_ALL = [1, 1.05, 1.2, 1.5, 2, 3, 6, 20, np.inf]
R_FAMILY = [1, 1.5, 2, 3, np.inf]
