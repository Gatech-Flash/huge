cmake_minimum_required(VERSION 3.14)
project(huge VERSION 1.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Options ----------------------------------------------------------
option(HUGE_BUILD_SHARED "Build shared library instead of static" OFF)
option(HUGE_OPENMP       "Enable OpenMP parallelism"             ON)

# ---------- OpenMP -----------------------------------------------------------
if(HUGE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
endif()

# ---------- Core library -----------------------------------------------------
set(HUGE_LIB_TYPE STATIC)
if(HUGE_BUILD_SHARED)
    set(HUGE_LIB_TYPE SHARED)
endif()

add_library(huge_core ${HUGE_LIB_TYPE} src/huge_core.cpp)
target_include_directories(huge_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(huge_core PRIVATE -O2 -Wall)

if(OpenMP_CXX_FOUND)
    target_link_libraries(huge_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------- Install rules ----------------------------------------------------
include(GNUInstallDirs)

install(TARGETS huge_core
    EXPORT hugeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/huge DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT hugeTargets
    FILE hugeTargets.cmake
    NAMESPACE huge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/huge
)
