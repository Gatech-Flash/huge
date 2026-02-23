"""Runtime diagnostics for pyhuge."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from typing import Any, Optional

from . import core as _core
from .core import PyHugeError


_R_INSTALL_HUGE_CMD = (
    "R -q -e 'install.packages(c(\"huge\", \"Rcpp\", \"RcppEigen\", \"igraph\"), "
    "repos=\"https://cloud.r-project.org\")'"
)


def _run_command(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return False, str(exc)

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    merged = out if out else err
    if out and err:
        merged = f"{out}\n{err}"
    return proc.returncode == 0, merged


def _r_value(expr: str) -> tuple[bool, str]:
    return _run_command(["R", "--slave", "-e", expr])


def _first_line(text: str) -> str:
    if not text:
        return ""
    return text.splitlines()[0].strip()


def doctor(require_runtime: bool = False) -> dict[str, Any]:
    """Collect and return a runtime readiness report."""

    py_arch = platform.machine().lower()
    report: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "arch": py_arch,
        },
        "rpy2": {
            "importable": False,
            "error": None,
        },
        "r": {
            "in_path": False,
            "path": None,
            "home": None,
            "arch": None,
            "error": None,
        },
        "r_packages": {
            "huge": False,
        },
        "runtime": {
            "ready": False,
            "error": None,
        },
        "suggestions": [],
    }

    rpy2_spec = importlib.util.find_spec("rpy2")
    if rpy2_spec is None:
        report["rpy2"]["error"] = "rpy2 is not installed."
    else:
        try:
            import rpy2  # noqa: F401

            report["rpy2"]["importable"] = True
        except Exception as exc:  # pragma: no cover - environment specific
            report["rpy2"]["error"] = str(exc)

    r_path = shutil.which("R")
    if r_path is None:
        report["r"]["error"] = "R executable not found in PATH."
    else:
        report["r"]["in_path"] = True
        report["r"]["path"] = r_path

        ok_home, out_home = _r_value("cat(R.home())")
        if ok_home:
            report["r"]["home"] = _first_line(out_home)
        else:
            report["r"]["error"] = _first_line(out_home)

        ok_arch, out_arch = _r_value("cat(R.version$arch)")
        if ok_arch:
            report["r"]["arch"] = _first_line(out_arch).lower()
        elif report["r"]["error"] is None:
            report["r"]["error"] = _first_line(out_arch)

        ok_huge, out_huge = _r_value(
            'if (requireNamespace("huge", quietly=TRUE)) cat("yes") else cat("no")'
        )
        if ok_huge:
            report["r_packages"]["huge"] = _first_line(out_huge) == "yes"

    if report["rpy2"]["importable"]:
        try:
            _core._r_env()
            report["runtime"]["ready"] = True
        except Exception as exc:  # pragma: no cover - runtime depends on env
            report["runtime"]["error"] = str(exc)
    else:
        report["runtime"]["error"] = "rpy2 bridge is unavailable."

    suggestions: list[str] = []

    if not report["rpy2"]["importable"]:
        suggestions.append('Install runtime bridge: pip install "pyhuge[runtime]"')

    if not report["r"]["in_path"]:
        suggestions.append(
            "Install R and make sure `R` is available in PATH (https://cran.r-project.org/)."
        )

    if report["r"]["in_path"] and not report["r_packages"]["huge"]:
        suggestions.append(f"Install R package dependencies: {_R_INSTALL_HUGE_CMD}")

    r_arch: Optional[str] = report["r"]["arch"]
    if isinstance(r_arch, str) and r_arch and py_arch and r_arch != py_arch:
        suggestions.append(
            f"Architecture mismatch detected (Python={py_arch}, R={r_arch}). "
            "Use matching architectures."
        )

    if report["r"]["in_path"] and report["r_packages"]["huge"] and not report["runtime"]["ready"]:
        suggestions.append(
            "Runtime still not ready. Verify R library visibility with "
            "`R -q -e 'print(.libPaths()); packageVersion(\"huge\")'` and set R_LIBS_USER if needed."
        )

    report["suggestions"] = suggestions

    if require_runtime and not report["runtime"]["ready"]:
        raise PyHugeError(format_doctor_report(report))

    return report


def format_doctor_report(report: dict[str, Any]) -> str:
    """Render a human-readable doctor report."""

    def flag(ok: bool) -> str:
        return "OK" if ok else "FAIL"

    python_info = report.get("python", {})
    rpy2_info = report.get("rpy2", {})
    r_info = report.get("r", {})
    pkg_info = report.get("r_packages", {})
    runtime_info = report.get("runtime", {})
    suggestions = report.get("suggestions", [])

    lines = [
        "pyhuge doctor report",
        f"- Python: {flag(True)} ({python_info.get('version', 'unknown')}, "
        f"{python_info.get('arch', 'unknown')})",
        f"- rpy2 import: {flag(bool(rpy2_info.get('importable')))}",
        f"- R in PATH: {flag(bool(r_info.get('in_path')))}",
        f"- R package huge: {flag(bool(pkg_info.get('huge')))}",
        f"- Runtime ready: {flag(bool(runtime_info.get('ready')))}",
    ]

    if r_info.get("path"):
        lines.append(f"- R path: {r_info['path']}")
    if r_info.get("home"):
        lines.append(f"- R home: {r_info['home']}")
    if r_info.get("arch"):
        lines.append(f"- R arch: {r_info['arch']}")

    error_notes = [
        ("rpy2 error", rpy2_info.get("error")),
        ("R error", r_info.get("error")),
        ("runtime error", runtime_info.get("error")),
    ]
    for label, value in error_notes:
        if value:
            lines.append(f"- {label}: {value}")

    if suggestions:
        lines.append("- Suggested actions:")
        lines.extend([f"  * {item}" for item in suggestions])

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for pyhuge-doctor."""

    parser = argparse.ArgumentParser(description="Check pyhuge runtime readiness.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report.",
    )
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Exit non-zero when runtime is not fully ready.",
    )
    args = parser.parse_args(argv)

    report = doctor(require_runtime=False)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_doctor_report(report))

    if args.require_runtime and not report["runtime"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
