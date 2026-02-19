# Design and Roadmap

## Why this design

`huge` currently exposes its core through R interfaces (`Rcpp`/`.Call`), not a
stable C ABI suitable for direct `ctypes` loading from Python.

To ship a usable Python interface quickly and safely, this version uses:

- `rpy2` for Python<->R bridge
- existing R package APIs as the execution backend
- Python-side conversion of outputs to NumPy/SciPy objects

## Data flow

1. Python input matrix (`numpy.ndarray`)
2. Converted to R object by `rpy2`
3. R `huge` package computes model path / selection
4. Returned R list is parsed into Python dataclasses
5. Matrix outputs converted to sparse/dense SciPy/NumPy formats

## Compatibility notes

- Runtime requires local R + installed R package `huge`.
- Python and R environments must be mutually visible in current shell/session.

## Roadmap to a native Python backend

If the goal is picasso-style native wrapping (no R runtime), next steps are:

1. Extract stable C/C++ API from `huge` core kernels
2. Build shared library target (`libhuge`)
3. Provide Python bindings (`pybind11` or `cffi`/`ctypes`)
4. Keep R wrapper as one frontend over the same core library

This is a larger refactor and should be done as a dedicated phase.
