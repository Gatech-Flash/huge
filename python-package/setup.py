from __future__ import annotations

import os
from setuptools import Extension, setup

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import numpy
    import pybind11

    ext_modules = [
        Extension(
            "pyhuge._native_core",
            [
                os.path.join("cpp", "native_core_bindings.cpp"),
                os.path.join("cpp", "huge_core.cpp"),
            ],
            include_dirs=[
                pybind11.get_include(),
                numpy.get_include(),
                os.path.join(_HERE, "cpp", "include"),
            ],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ]
except Exception:
    # Build without C++ acceleration if build deps are unavailable.
    ext_modules = []

setup(ext_modules=ext_modules)
