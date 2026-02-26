from __future__ import annotations

from setuptools import Extension, setup

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

setup(ext_modules=ext_modules)
