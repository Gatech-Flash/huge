## Submission preparation: huge 2.0.2

This is a preparation record, not a completed submission or a declaration
that all release checks have passed. No publication date is assigned.
The version is increased from published 2.0.1 to 2.0.2; the public API is
retained. Implementation and memory changes are described in NEWS.md.

## Completed local checks on 2.0.2

Platform: macOS arm64, R 4.6.1, Apple Clang 21, R's reference BLAS,
normal C++17 -O2 compilation without OpenMP.

The BLAS label is corrected from the earlier preparation record: the R
installation and both compared R packages resolve libRblas to the reference
library, not Accelerate. The recorded test results are unchanged; Python and
standalone macOS builds use Accelerate separately.

* A fresh source export was built with vignette rebuilding enabled.
  A private R library initially contained no huge installation. A complete
  cold R CMD check --as-cran of huge_2.0.2.tar.gz exited 0 without timeout:
  0 ERROR, 1 WARNING, 2 NOTEs. All seven C++ units were freshly compiled;
  installed object-symbol metadata was generated. Compiled code, incoming
  feasibility and timestamp checks were OK.
* The WARNING identified missing local checkbashisms and Pandoc tools.
  After providing official tools and modern HTML Tidy in a temporary
  directory, a second --as-cran check reused that successful cold installation
  and its actual install log/symbol metadata. It exited 0 without timeout:
  0 ERROR, 0 WARNING, 2 NOTEs. Shell, README/NEWS and HTML validation passed.
  This second run was not another cold compilation; the first log is retained.
* Both checks reported 3746 testthat PASS, with no FAIL, WARN or SKIP.
  Legacy tests, examples, vignette rebuilding and the PDF manual passed.
  Default manual fonts were supplied through a temporary user TeX tree.
  Local FLIBS= bypassed nonexistent host gfortran paths for this pure C++
  package without changing optimization flags or system configuration.
* Python 2.0.2 sdist and its wheel were rebuilt. Fresh wheel and uncached
  sdist installations each passed 189 runtime tests and 7 comparisons with
  the new R 2.0.2 library, without failures, skips or warnings. Metadata,
  licenses and installation paths passed. These are local CPython 3.12
  serial builds. Standalone CMake 2.0.2 static/shared OpenMP installations
  and their actual consumers also passed on macOS.

## Remaining diagnostics and submission checks

The two local NOTEs indicate that Ghostscript is unavailable for PDF
size-reduction checks and the R V8 package is unavailable for math-rendering
checks. Those checks were not disabled or declared accepted by CRAN.

Before submission, obtain actual Linux R-release/R-devel, Windows R and
manylinux wheel results, and verify remaining diagnostics in a fully equipped
check environment. Workflow syntax and synthetic controls are not remote
runner results. R-devel submission checks remain unverified locally.

Check current reverse dependencies, prioritizing heterocop, NetGreg,
netgwas, nutriNetwork and SparseTSCGM; the official package page also lists
nine reverse suggests. The old six-of-eight results do not establish current
coverage. Historical 2.0.1 ATLAS fixes and benchmark reports are preserved
as history, not substituted for 2.0.2 validation.

Detailed commands, logs, hashes and limitations are recorded in
benchmark/release-2.0.2-readiness.md and its evidence archive. No package has
been submitted or published by this preparation.
