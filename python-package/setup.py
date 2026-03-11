from __future__ import annotations

import os
import sys
import platform
from setuptools import Extension, setup

_HERE = os.path.dirname(os.path.abspath(__file__))


def _get_blas_args():
    """Return (libraries, library_dirs, extra_link_args) for BLAS."""
    if sys.platform == "darwin":
        # macOS: use Accelerate framework (always available)
        return [], [], ["-framework", "Accelerate"]
    else:
        # Linux: link against OpenBLAS (or generic BLAS/CBLAS)
        return ["openblas"], [], []


try:
    import numpy
    import pybind11

    blas_libs, blas_lib_dirs, blas_link_args = _get_blas_args()

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
            libraries=blas_libs,
            library_dirs=blas_lib_dirs,
            extra_compile_args=["-O3", "-std=c++17"],
            extra_link_args=blas_link_args,
            language="c++",
        )
    ]
except ImportError:
    # pybind11 or numpy not available at build time — skip C++ extension.
    ext_modules = []

setup(ext_modules=ext_modules)
