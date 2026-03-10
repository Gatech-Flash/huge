from __future__ import annotations

import os
from setuptools import Extension, setup

# Relative path from python-package/ to repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.relpath(os.path.dirname(_HERE), _HERE)

try:
    import numpy
    import pybind11

    ext_modules = [
        Extension(
            "pyhuge._native_core",
            [
                "cpp/native_core_bindings.cpp",
                os.path.join(_REPO, "src", "huge_core.cpp"),
            ],
            include_dirs=[
                pybind11.get_include(),
                numpy.get_include(),
                os.path.join(_REPO, "include"),
            ],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ]
except Exception:
    # Build without C++ acceleration if build deps are unavailable.
    ext_modules = []

setup(ext_modules=ext_modules)
