from __future__ import annotations

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class _OptionalBuildExt(build_ext):
    """Build C++ extension opportunistically; fall back to pure Python on error."""

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:
            self._warn(exc)
            self.extensions = []

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._warn(exc)

    @staticmethod
    def _warn(exc: Exception) -> None:
        print(
            f"warning: failed to build optional pyhuge C++ extension ({exc}); "
            "falling back to pure Python runtime.",
            file=sys.stderr,
        )

try:
    import numpy
    import pybind11

    ext_modules = [
        Extension(
            "pyhuge._native_core",
            ["cpp/native_core.cpp"],
            include_dirs=[pybind11.get_include(), numpy.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ]
except Exception:
    # Build without C++ acceleration if build deps are unavailable.
    ext_modules = []

setup(ext_modules=ext_modules, cmdclass={"build_ext": _OptionalBuildExt})
