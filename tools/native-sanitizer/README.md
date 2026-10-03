# Native sanitizer checks

From the repository root, run:

```sh
sh tools/check_native_sanitizers.sh
HUGE_NATIVE_SANITIZERS=thread sh tools/check_native_sanitizers.sh
```

The first command instruments both the C++ core and the standalone harness with
AddressSanitizer and UndefinedBehaviorSanitizer. The second uses ThreadSanitizer
to check concurrent calls from host threads. CMake, Clang/GCC and an LP64 BLAS
are required; R and Python are not needed. Temporary builds are removed on exit.

The harness checks known MB solutions across active-set cache boundaries, MB KKT
conditions on rank-deficient data, glasso and TIGER analytical solutions, extreme
scales, path truncation, invalid arguments, RIC against a scalar reference,
generator invariants, and concurrent calls sharing read-only inputs.

The glasso checks include an independent Markov-chain solution at dimension 512
and concurrent host calls. A separate executable includes the core directly to
compare private sparse residual certification against the original full BLAS
product at both decision thresholds, dimension/density boundaries, nonsymmetric
inputs and extreme scales. No testing interface is added to the shipped core.

To require actual OpenMP execution:

```sh
HUGE_NATIVE_OPENMP=ON HUGE_NATIVE_EXPECT_OPENMP=ON \
  sh tools/check_native_sanitizers.sh
```

Extra script arguments are forwarded to CMake. Apple Clang with a separately
installed `libomp` may need `OpenMP_CXX_FLAGS`, `OpenMP_CXX_INCLUDE_DIR`,
`OpenMP_CXX_LIB_NAMES`, and `OpenMP_omp_LIBRARY`. On macOS, the library's install
name must resolve at runtime; an absolute `/usr/local/lib/libomp.dylib` install
name is not fixed by adding an unrelated rpath. Use a correctly installed runtime
or a temporary copy with a matching `@rpath` install name and valid signature.

With OpenMP enabled, the harness verifies a multiple-worker team and concurrent
host calls that each request two OpenMP workers. These checks validate observable
results and sanitizer findings; they do not instrument prebuilt BLAS/OpenMP
libraries or prove the absence of every race. Leak detection depends on the
platform's sanitizer runtime.

## Python binding ownership and lifetime checks

The standalone commands above do not instrument the Rcpp or pybind11 adapters.
Ordinary wrapper tests also do not provide sanitizer coverage. To instrument
both `native_core_bindings.cpp` and `huge_core.cpp` in an isolated Python build:

```sh
PYTHON=/path/to/venv/bin/python sh tools/check_python_binding_sanitizers.sh
```

Install Clang, NumPy, SciPy, pybind11, setuptools and pytest first; Linux also
requires OpenBLAS development libraries. The script supports Linux x86_64 and
aarch64, and macOS with a non-system Python that permits sanitizer runtime
preloading. Unsupported compilers/platforms, missing runtimes, failed imports
and skipped tests fail the check. It verifies instrumentation in both object
files, checks the loaded extension location, and requires temporary heap-overflow
and signed-overflow probes to trigger ASan and UBSan respectively.

The selected existing tests cover capsule owners, Fortran-contiguous matrices,
matrix lists, retained views, exception cleanup, shared read-only inputs and
concurrent calls that release the GIL. OpenMP is disabled in this check. Python,
NumPy and BLAS remain prebuilt; Rcpp, ThreadSanitizer and native leak detection
are outside its scope. ASan leak detection is disabled for the Python process.
Set `HUGE_PYTHON_SANITIZER_OUTPUT` to an absent or empty directory to retain build
logs, object-symbol evidence, probe diagnostics and JUnit results; otherwise
temporary files are removed. With an output directory set, additionally set
`HUGE_PYTHON_SANITIZER_KEEP_BUILD=1` to retain the copied source, build and probes.
Setuptools may retain Python's `-fwrapv` compiler option, so the signed-overflow
probe verifies UBSan runtime activation rather than coverage of every signed
overflow in the extension.
