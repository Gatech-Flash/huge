#!/bin/sh
# Run the standalone core without Python/R wrappers under ASan and UBSan.
# Additional arguments are passed to CMake (for example, OpenMP library paths).
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sanitizer_root=$(mktemp -d "${TMPDIR:-/tmp}/huge-native-sanitizer.XXXXXX")
trap 'rm -rf -- "$sanitizer_root"' EXIT HUP INT TERM

cmake -S "$repository_root/tools/native-sanitizer" -B "$sanitizer_root" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DHUGE_OPENMP="${HUGE_NATIVE_OPENMP:-OFF}" \
    -DHUGE_NATIVE_EXPECT_OPENMP="${HUGE_NATIVE_EXPECT_OPENMP:-OFF}" \
    -DHUGE_NATIVE_SANITIZERS="${HUGE_NATIVE_SANITIZERS:-address,undefined}" "$@"
cmake --build "$sanitizer_root" --config RelWithDebInfo
for executable in huge_native_sanitizer huge_residual_boundary; do
    ASAN_OPTIONS="${ASAN_OPTIONS:-halt_on_error=1}" \
    UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}" \
        "$sanitizer_root/$executable"
done
