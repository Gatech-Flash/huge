args <- commandArgs(TRUE)
phase <- args[1]
shared <- normalizePath(args[2], mustWork = TRUE)
baseline <- args[3]
.libPaths(c(shared, if (nzchar(baseline)) baseline, .libPaths()))
# The R CMD INSTALL subprocess must see the same read-only huge installation.
Sys.setenv(R_LIBS_USER = paste(.libPaths(), collapse = .Platform$path.sep))
options(repos = c(CRAN = "https://cran.r-project.org"))
Sys.setenv(PKG_SYSREQS = "true", PKG_SYSREQS_SUDO = "true",
           MAKEFLAGS = "-j1", OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")
stopifnot(requireNamespace("pak", quietly = TRUE))
print(sessionInfo())
if (phase == "core") {
    refs <- readLines(args[4], warn = FALSE)
    stopifnot(!"huge" %in% refs)
    pak::pkg_install(refs, lib = shared, dependencies = NA, ask = FALSE,
                     upgrade = TRUE)
} else if (phase == "plan") {
    refs <- readLines(args[4], warn = FALSE)
    stopifnot(!"huge" %in% refs)
    plan <- pak::pkg_deps(refs, dependencies = NA)
    saveRDS(plan, args[5])
    fields <- intersect(c("package", "version", "ref", "type", "status", "priority", "sha256", "md5sum", "mirror"), names(plan))
    write.csv(plan[, fields, drop = FALSE], args[6], row.names = FALSE)
    stopifnot(all(plan$status == "OK"))
} else if (phase == "install") {
    plan <- readRDS(args[4])
    keep <- !plan$package %in% c("huge", "R")
    if ("priority" %in% names(plan)) {
        keep <- keep & (is.na(plan$priority) | plan$priority != "base")
    }
    # Keep prior-stage versions. A required upgrade is a recorded setup failure,
    # never a silent change to an already-tested environment.
    prior <- installed.packages(lib.loc = shared)
    present <- match(plan$package, prior[, "Package"])
    existing <- keep & !is.na(present)
    stopifnot(all(plan$version[existing] == prior[present[existing], "Version"]))
    system <- installed.packages(lib.loc = .Library)
    system_present <- match(plan$package, system[, "Package"])
    system_satisfies <- !is.na(system_present) &
        plan$version == system[system_present, "Version"]
    keep <- keep & is.na(present) & !system_satisfies
    pending <- which(keep)
    # Install in the recorded hard-dependency order without allowing the
    # solver to add huge to shared. Optional-dependency cycles do not order builds.
    while (length(pending)) {
        ready <- pending[vapply(pending, function(i) {
            deps <- plan$deps[[i]]
            hard <- deps$package[deps$type %in% c("depends", "imports", "linkingto")]
            !any(hard %in% plan$package[pending])
        }, logical(1))]
        if (!length(ready)) stop("Recorded hard-dependency plan contains a cycle")
        pak::pkg_install(unique(plan$ref[ready]), lib = shared,
                         dependencies = FALSE, ask = FALSE, upgrade = FALSE)
        stopifnot(!dir.exists(file.path(shared, "huge")))
        pending <- setdiff(pending, ready)
    }
    installed <- installed.packages(lib.loc = c(shared, .Library))
    wanted <- !plan$package %in% c("huge", "R")
    present <- match(plan$package[wanted], installed[, "Package"])
    stopifnot(!anyNA(present),
              all(plan$version[wanted] == installed[present, "Version"]))
} else {
    stop("Unknown dependency phase")
}
stopifnot(!dir.exists(file.path(shared, "huge")))
installed <- installed.packages(lib.loc = shared)
write.csv(installed[, c("Package", "Version", "LibPath"), drop = FALSE],
          args[length(args)], row.names = FALSE)
