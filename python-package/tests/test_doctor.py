"""Unit tests for doctor diagnostics helpers."""

from __future__ import annotations

import importlib as py_importlib
import json

import pytest

from pyhuge import PyHugeError, doctor, format_doctor_report
from pyhuge.doctor import main as doctor_main


def test_doctor_returns_expected_top_level_keys():
    report = doctor()
    assert "python" in report
    assert "rpy2" in report
    assert "r" in report
    assert "r_packages" in report
    assert "runtime" in report
    assert "suggestions" in report


def test_format_doctor_report_contains_runtime_line():
    report = {
        "python": {"version": "3.11.0", "arch": "arm64"},
        "rpy2": {"importable": False, "error": "missing"},
        "r": {"in_path": False, "path": None, "home": None, "arch": None, "error": "R not found"},
        "r_packages": {"huge": False},
        "runtime": {"ready": False, "error": "bridge down"},
        "suggestions": ["Install runtime bridge"],
    }
    text = format_doctor_report(report)
    assert "Runtime ready: FAIL" in text
    assert "Install runtime bridge" in text


def test_doctor_main_json_output(monkeypatch, capsys):
    doctor_module = py_importlib.import_module("pyhuge.doctor")
    fake = {
        "python": {"version": "3.11.0", "arch": "arm64"},
        "rpy2": {"importable": True, "error": None},
        "r": {"in_path": True, "path": "/usr/bin/R", "home": "/usr/lib/R", "arch": "arm64", "error": None},
        "r_packages": {"huge": True},
        "runtime": {"ready": True, "error": None},
        "suggestions": [],
    }
    monkeypatch.setattr(doctor_module, "doctor", lambda require_runtime=False: fake)
    code = doctor_main(["--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert code == 0
    assert parsed["runtime"]["ready"] is True


def test_doctor_require_runtime_raises_when_not_ready(monkeypatch):
    doctor_module = py_importlib.import_module("pyhuge.doctor")
    monkeypatch.setattr(doctor_module.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(PyHugeError):
        doctor(require_runtime=True)
