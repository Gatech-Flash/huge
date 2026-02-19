"""Public API smoke tests."""

from __future__ import annotations

import pytest

import pyhuge


def test_test_returns_status_when_rpy2_missing(monkeypatch):
    monkeypatch.setattr(pyhuge, "_importlib_util", type("X", (), {"find_spec": staticmethod(lambda _: None)}))

    status = pyhuge.test()

    assert status["python_import"] is True
    assert status["rpy2"] is False
    assert status["runtime"] is False


def test_test_raises_when_runtime_required_and_rpy2_missing(monkeypatch):
    monkeypatch.setattr(pyhuge, "_importlib_util", type("X", (), {"find_spec": staticmethod(lambda _: None)}))

    with pytest.raises(pyhuge.PyHugeError, match="rpy2 is required"):
        pyhuge.test(require_runtime=True)


def test_test_reports_runtime_ready(monkeypatch):
    monkeypatch.setattr(
        pyhuge,
        "_importlib_util",
        type("X", (), {"find_spec": staticmethod(lambda _: object())}),
    )
    monkeypatch.setattr(pyhuge._core, "_r_env", lambda: {"ok": True})

    status = pyhuge.test()

    assert status["rpy2"] is True
    assert status["runtime"] is True


def test_test_raises_when_runtime_required_and_runtime_fails(monkeypatch):
    monkeypatch.setattr(
        pyhuge,
        "_importlib_util",
        type("X", (), {"find_spec": staticmethod(lambda _: object())}),
    )

    def _fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(pyhuge._core, "_r_env", _fail)

    with pytest.raises(pyhuge.PyHugeError, match="R runtime for pyhuge is unavailable"):
        pyhuge.test(require_runtime=True)
