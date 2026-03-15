## Test environments
* macOS Tahoe 26.3, R 4.5.2 (aarch64-apple-darwin20)

## R CMD check results
* `R CMD check --as-cran huge_1.5.tar.gz`: Status OK
  (1 local-only WARNING for missing `checkbashisms`, 1 local-only NOTE
  for `unable to verify current time` — neither appears on CRAN servers)

## Resubmission
This is an update from version 1.4 to 1.5. Changes include:

* BLAS acceleration for graphical lasso and TIGER solvers
* Removed RcppEigen dependency
* Added testthat test suite
* Fixed documentation and minor bugs
* Fixed vignette PDF rebuild (renamed included PDF to avoid filename
  collision with pdflatex output)
* Compacted vignette PDF to pass CRAN size check

The maintainer has been updated to:
Tuo Zhao <tourzhao@gatech.edu>
