cmake_minimum_required(VERSION 3.18)
project(huge_native_sanitizers LANGUAGES CXX)

set(HUGE_NATIVE_SANITIZERS "address,undefined" CACHE STRING "Clang/GCC sanitizers")
option(HUGE_NATIVE_EXPECT_OPENMP "Require the harness to exercise OpenMP" OFF)
if(HUGE_NATIVE_EXPECT_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
set(HUGE_BUILD_SHARED OFF CACHE BOOL "Build the instrumented core statically" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." huge-core)
find_package(Threads REQUIRED)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    message(FATAL_ERROR "The native sanitizer harness requires Clang or GCC")
endif()
if(HUGE_NATIVE_EXPECT_OPENMP AND NOT TARGET OpenMP::OpenMP_CXX)
    message(FATAL_ERROR "OpenMP was requested but its C++ runtime was not found")
endif()

add_executable(huge_native_sanitizer core_checks.cpp)
target_link_libraries(huge_native_sanitizer PRIVATE huge_core Threads::Threads)
target_compile_definitions(huge_native_sanitizer PRIVATE
    "$<$<BOOL:${HUGE_NATIVE_EXPECT_OPENMP}>:HUGE_EXPECT_OPENMP>")

# This independent test TU includes the core to exercise private numerical
# helpers. Do not also link huge_core, which would define the core twice.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    set(BLA_SIZEOF_INTEGER 4)
endif()
find_package(BLAS REQUIRED)
add_executable(huge_residual_boundary residual_boundary_checks.cpp)
target_include_directories(huge_residual_boundary PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../src")
target_compile_features(huge_residual_boundary PRIVATE cxx_std_17)
target_link_libraries(huge_residual_boundary PRIVATE BLAS::BLAS Threads::Threads)
if(HUGE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(huge_residual_boundary PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

foreach(target huge_core huge_native_sanitizer huge_residual_boundary)
    target_compile_options(${target} PRIVATE
        "-fsanitize=${HUGE_NATIVE_SANITIZERS}" -fno-omit-frame-pointer)
endforeach()
foreach(target huge_native_sanitizer huge_residual_boundary)
    target_link_options(${target} PRIVATE
        "-fsanitize=${HUGE_NATIVE_SANITIZERS}" -fno-omit-frame-pointer)
endforeach()
