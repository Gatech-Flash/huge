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
# Explicit installed references avoid resolving R-devel Recommended rows in
# the global source index. Shared/core versions are fixed within this epoch.
installed_refs <- function(pkgs) {
    paste0("installed::", file.path(pkgs[, "LibPath"], pkgs[, "Package"]))
}
resolver_roots <- function(refs, shared, baseline) {
    sys <- installed.packages(lib.loc = .Library)
    base_names <- sys[sys[, "Priority"] %in% "base", "Package"]
    rec <- sys[sys[, "Priority"] %in% "recommended", , drop = FALSE]
    fields <- c("Package", "Version", "LibPath")
    fixed <- rbind(installed.packages(lib.loc = shared)[, fields, drop = FALSE],
                   rec[, fields, drop = FALSE])
    fixed <- fixed[!duplicated(fixed[, "Package"]), , drop = FALSE]
    if (nzchar(baseline)) {
        huge <- installed.packages(lib.loc = baseline)
        huge <- huge[huge[, "Package"] == "huge", , drop = FALSE]
        fixed <- rbind(fixed, huge[, fields, drop = FALSE])
    }
    # Base packages are supplied by R; pak cannot resolve them as installed roots.
    roots <- refs[!refs %in% c(fixed[, "Package"], base_names)]
    unique(c(roots, installed_refs(fixed)))
}
if (phase == "core") {
    refs <- readLines(args[4], warn = FALSE)
    stopifnot(!"huge" %in% refs)
    pak::pkg_install(resolver_roots(refs, shared, baseline), lib = shared,
                     dependencies = NA, ask = FALSE, upgrade = FALSE)
} else if (phase == "plan") {
    refs <- readLines(args[4], warn = FALSE)
    stopifnot(!"huge" %in% refs)
    roots <- resolver_roots(refs, shared, baseline)
    libs <- c(shared, if (nzchar(baseline)) normalizePath(baseline, mustWork = TRUE))
    fixed <- roots[startsWith(roots, "installed::")]
    rows <- lapply(fixed, function(ref) {
        path <- substring(ref, nchar("installed::") + 1L)
        d <- read.dcf(file.path(path, "DESCRIPTION"))
        data.frame(Package = d[1, "Package"], Version = d[1, "Version"],
                   Ref = ref, stringsAsFactors = FALSE)
    })
    write.csv(do.call(rbind, rows), paste0(args[5], ".expected.csv"), row.names = FALSE)
    writeLines(roots, paste0(args[5], ".roots.txt"))
    writeLines(c(paste0("R=", getRversion()), paste0("pak=", packageVersion("pak")),
                 paste0("configured_library=", libs), paste0("R_lib_path=", .libPaths()),
                 paste0("implicit_Recommended_library=", .Library)),
               paste0(args[5], ".visibility.txt"))
    # pkg_deps() always solves against an empty temporary library. The public
    # lockfile API accepts explicit libraries and keeps the complete solved plan.
    pak::lockfile_create(fixed, lockfile = paste0(args[5], ".visibility.lock.json"),
                         lib = libs, upgrade = FALSE, dependencies = NA)
    pak::lockfile_create(roots, lockfile = args[5], lib = libs,
                         upgrade = FALSE, dependencies = NA)
} else if (phase == "install") {
    plan <- read.csv(args[4], colClasses = "character", check.names = FALSE)
    dep_table <- read.csv(paste0(args[4], ".deps.csv"), colClasses = "character", check.names = FALSE)
    plan$deps <- lapply(plan$package, function(package) {
        dep_table[dep_table$owner == package, c("ref", "type", "package", "op", "version"), drop = FALSE]
    })
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
            hard <- deps$package[tolower(deps$type) %in% c("depends", "imports", "linkingto")]
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
