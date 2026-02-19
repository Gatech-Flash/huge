"""Runtime error surface tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pyhuge import PyHugeError, huge


@pytest.mark.skipif(
    importlib.util.find_spec("rpy2") is not None,
    reason="Only applicable when rpy2 is not installed",
)
def test_missing_rpy2_error_message():
    with pytest.raises(PyHugeError, match="rpy2 is required"):
        huge(np.random.randn(8, 4), verbose=False)
