import os
import sys

import pytest

# Make the package importable without installation (CI installs pytest only).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Deterministic timestamp for report rendering tests.
FIXED_TS = 1750000000


@pytest.fixture
def tmp_report(tmp_path):
    return str(tmp_path / "report.html")
