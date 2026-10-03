#!/bin/sh
# Instrument the Python binding and core in an isolated, serial source build.
set -eu
if [ "$#" -ne 0 ]; then
    echo "Usage: PYTHON=/path/to/python sh tools/check_python_binding_sanitizers.sh" >&2
    exit 2
fi
repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export HUGE_PYTHON_SANITIZER_REPOSITORY="$repository_root"
ulimit -c 0
"${PYTHON:-python3}" - <<'PY'
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def main():
    require(sys.platform in ("linux", "darwin"), "Only Linux and macOS are supported.")
    require(sys.version_info >= (3, 9), "Python 3.9 or newer is required.")
    for module in ("setuptools", "numpy", "scipy", "pybind11", "pytest"):
        importlib.import_module(module)
    cxx = shutil.which(os.environ.get("CXX", "clang++"))
    cc = shutil.which(os.environ.get("CC", "clang"))
    require(cxx and cc, "Clang/clang++ must be installed (CC and CXX may select their paths).")
    require(all("clang" in command_output([compiler, "--version"]).lower()
                for compiler in (cc, cxx)), "This check requires Clang; GCC is not supported.")
    require(shutil.which("nm"), "nm is required to verify instrumentation in both object files.")
    resource = Path(command_output([cxx, "-print-resource-dir"]))
    if sys.platform == "darwin":
        runtime = resource / "lib/darwin/libclang_rt.asan_osx_dynamic.dylib"
        shared_flags = ["-dynamiclib"]
        link_flags = []
    else:
        arch = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine())
        require(arch, "Linux sanitizer checks support x86_64 and aarch64 only.")
        runtime = resource / f"lib/linux/libclang_rt.asan-{arch}.so"
        shared_flags = ["-shared", "-fPIC"]
        link_flags = ["-shared-libasan"]
    require(runtime.is_file(), f"Clang's matching shared ASan runtime is missing: {runtime}")

    repository = Path(os.environ["HUGE_PYTHON_SANITIZER_REPOSITORY"])
    requested_output = os.environ.get("HUGE_PYTHON_SANITIZER_OUTPUT")
    keep_build = os.environ.get("HUGE_PYTHON_SANITIZER_KEEP_BUILD") == "1"
    require(not keep_build or requested_output,
            "HUGE_PYTHON_SANITIZER_KEEP_BUILD=1 requires HUGE_PYTHON_SANITIZER_OUTPUT.")
    with tempfile.TemporaryDirectory(prefix="huge-python-sanitizer-") as temporary:
        scratch = Path(temporary)
        output = Path(requested_output).resolve() if requested_output else scratch / "logs"
        require(not output.exists() or not any(output.iterdir()),
                f"The output directory must be absent or empty: {output}")
        require(repository / "python-package" not in output.parents,
                "The output directory must be outside python-package to avoid recursive copying.")
        output.mkdir(parents=True, exist_ok=True)
        if keep_build:
            scratch = output / "work"
            scratch.mkdir()
        package = scratch / "package"
        # The copied conftest inserts this isolated package, never the repository.
        shutil.copytree(repository / "python-package", package,
                        ignore=shutil.ignore_patterns("build", "*.egg-info", "*.so", "*.dylib",
                                                     "*.pyd", "*.pyc", "__pycache__", ".mplconfig",
                                                     "site", "dist", ".venv", ".pytest_cache"))
        sanitizer_flags = ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        build_env = os.environ.copy()
        build_env.update(CC=cc, CXX=cxx, PYHUGE_NO_OPENMP="1",
                         CFLAGS=" ".join(sanitizer_flags), CXXFLAGS=" ".join(sanitizer_flags),
                         CPPFLAGS="", LDFLAGS=" ".join(sanitizer_flags + link_flags))

        def run(command, name, env, expected_failure=None, timeout=300):
            log = output / f"{name}.log"
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=package, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            contents = log.read_text(errors="replace")
            if expected_failure:
                require(result.returncode != 0 and expected_failure in contents,
                        f"{name} did not produce the required sanitizer failure; see {log}\n{contents[-6000:]}")
            else:
                require(result.returncode == 0,
                        f"{name} failed with status {result.returncode}; see {log}\n{contents[-6000:]}")
            return contents

        print("Building both Python binding and core with ASan/UBSan (OpenMP disabled).", flush=True)
        run([sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
            "build", build_env, timeout=600)
        objects = {}
        for name in ("huge_core", "native_core_bindings"):
            matches = list((package / "build").rglob(f"{name}.o"))
            require(len(matches) == 1, f"Expected exactly one freshly built {name}.o, found {matches}")
            symbols = command_output(["nm", "-u", str(matches[0])])
            (output / f"{name}-undefined-symbols.txt").write_text(symbols + "\n")
            require("__asan_report" in symbols and "__ubsan_handle" in symbols,
                    f"Missing ASan or UBSan instrumentation in {name}.o")
            source = package / "cpp" / f"{name}.cpp"
            objects[name] = {"asan": True, "ubsan": True,
                             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

        runtime_env = build_env.copy()
        runtime_env.update(ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1",
                           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
                           PYTHONMALLOC="malloc", PYTHONPATH=str(package),
                           PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        if sys.platform == "darwin":
            runtime_env["DYLD_INSERT_LIBRARIES"] = str(runtime)
        else:
            runtime_env["LD_PRELOAD"] = str(runtime)
            runtime_env["LD_LIBRARY_PATH"] = str(runtime.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        preflight = """
import ctypes, json, pathlib
runtime = ctypes.CDLL(None)
getattr(runtime, '__asan_init')
getattr(runtime, '__ubsan_handle_add_overflow')
from pyhuge import _native_core
path = pathlib.Path(_native_core.__file__).resolve()
assert path.parent == pathlib.Path('pyhuge').resolve(), path
assert _native_core.omp_max_threads() == 1
print(json.dumps({'extension': str(path), 'asan_runtime_loaded': True, 'ubsan_runtime_loaded': True}))
"""
        try:
            loaded = json.loads(run([sys.executable, "-c", preflight], "runtime-preflight", runtime_env, timeout=30))
        except RuntimeError as error:
            raise RuntimeError(f"Sanitizer runtime/import preflight failed. On macOS use a non-system Python "
                               f"whose executable permits DYLD_INSERT_LIBRARIES (for example a uv Python).\n{error}") from error

        probe = scratch / "runtime_probe.cpp"
        probe.write_text("""
#include <climits>
extern "C" void heap_oob() {
    volatile int* values = new int[1];
    values[1] = 7;
    delete[] values;
}
extern "C" void signed_overflow() {
    volatile int value = INT_MAX;
    value += 1;
}
""")
        probe_library = scratch / ("runtime_probe.dylib" if sys.platform == "darwin" else "runtime_probe.so")
        run([cxx, "-std=c++17", "-O0", *sanitizer_flags, *shared_flags, *link_flags,
             str(probe), "-o", str(probe_library)], "probe-build", build_env, timeout=60)
        for function, diagnostic in (("heap_oob", "AddressSanitizer: heap-buffer-overflow"),
                                     ("signed_overflow", "runtime error: signed integer overflow")):
            code = f"import ctypes; ctypes.CDLL({str(probe_library)!r}).{function}()"
            run([sys.executable, "-c", code], function, runtime_env, diagnostic, timeout=30)

        tests = ["test_matrix_ownership.py", "test_tiger_materialization.py",
                 "test_solver_contracts.py", "test_native_symbols.py"]
        junit = output / "pytest.xml"
        contents = run([sys.executable, "-m", "pytest", "-ra", f"--junitxml={junit}",
                        *(f"tests/{name}" for name in tests)], "pytest", runtime_env)
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        counts = {key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        require(counts["tests"] > 0 and all(counts[key] == 0 for key in ("failures", "errors", "skipped")),
                f"Sanitizer checks must execute tests with no skips or failures: {counts}")
        result = {"python": sys.version, "compiler": command_output([cxx, "--version"]),
                  "platform": platform.platform(), "runtime": str(runtime),
                  "dependencies": {name: importlib.import_module(name).__version__
                                   for name in ("numpy", "scipy", "pybind11", "pytest", "setuptools")},
                  "instrumented_objects": objects, "extension": loaded,
                  "extension_sha256": hashlib.sha256(Path(loaded["extension"]).read_bytes()).hexdigest(),
                  "asan_negative_probe": "detected", "ubsan_negative_probe": "detected",
                  "tests": tests, "pytest_counts": counts, "openmp": False, "leak_detection": False}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(contents, end="")
        print(json.dumps(result, indent=2))
        if requested_output:
            print(f"Sanitizer logs retained in {output}")


try:
    main()
except (RuntimeError, subprocess.SubprocessError, ImportError, OSError) as error:
    print(f"Python binding sanitizer check failed: {error}", file=sys.stderr)
    sys.exit(1)
PY
