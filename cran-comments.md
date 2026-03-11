## Test environments
* macOS Tahoe 26.3, R 4.5.2 (aarch64-apple-darwin20)

## R CMD check results
* `R CMD build --no-build-vignettes` on the source directory: OK
* `R CMD check --as-cran huge_1.5.tar.gz`: unable to run locally due to
  missing gfortran at `/opt/gfortran/lib/` (macOS-specific). The `$(FLIBS)`
  flag in `Makevars.in` is required by CRAN policy when using `$(BLAS_LIBS)`,
  and CRAN build servers have gfortran properly configured.
* All 306 testthat tests pass on locally installed package.

## Resubmission
This is an update from version 1.4 to 1.5. Changes include:

* BLAS acceleration for graphical lasso and TIGER solvers
* Removed RcppEigen dependency
* Added testthat test suite
* Fixed documentation and minor bugs

The maintainer has been updated to:
Tuo Zhao <tourzhao@gatech.edu>
