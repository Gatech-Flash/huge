"""Bounded real Linux CRAN reverse checks; selftest does no R/network/model work."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

CRAN = "https://cran.r-project.org"
PRIORITY = ["heterocop", "NetGreg", "netgwas", "nutriNetwork", "SparseTSCGM"]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def dcf(text):
    records, current, field = [], {}, None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if current:
                records.append(current)
            current, field = {}, None
        elif line[0].isspace():
            if field is None:
                raise ValueError("DCF continuation without a field")
            current[field] += " " + line.strip()
        else:
            field, value = line.split(":", 1)
            current[field] = value.strip()
    return records


def dependencies(value):
    return [re.match(r"([A-Za-z][A-Za-z0-9.]*)", x.strip()).group(1)
            for x in value.split(",") if x.strip()]


def discover(records):
    relations = {key: sorted(r["Package"] for r in records
                            if "huge" in dependencies(r.get(key, "")))
                 for key in ("Imports", "Suggests", "Depends", "LinkingTo")}
    names = set().union(*relations.values()) - {"huge"}
    ordered = [x for x in PRIORITY if x in names] + sorted(names - set(PRIORITY))
    return relations, ordered


def tar_description(path):
    with tarfile.open(path) as opened:
        entries = [m for m in opened.getmembers()
                   if len(Path(m.name).parts) == 2 and Path(m.name).name == "DESCRIPTION"]
        if len(entries) != 1 or entries[0].size > 1048576:
            raise ValueError("Source archive has no unique top-level DESCRIPTION")
        return dcf(opened.extractfile(entries[0]).read().decode())[0]


def test_output_counters(check_directory):
    """Retain every tests Rout identity; count only actual structured summaries."""
    files = {}
    directory = Path(check_directory) / "tests"
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or not re.search(r"\.Rout(?:\.fail)?$", path.name):
                continue
            raw = path.read_text(errors="replace")
            plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw)
            summaries = re.findall(r"\[\s*FAIL\s+(\d+)\s*\|\s*WARN\s+(\d+)\s*\|\s*SKIP\s+(\d+)\s*\|\s*PASS\s+(\d+)\s*\]", plain)
            counts = dict(zip(("FAIL", "WARN", "SKIP", "PASS"), map(int, summaries[-1]))) if summaries else None
            key = str(path.relative_to(directory)).removesuffix(".fail")
            testthat = bool(counts is not None or "testthat" in path.name.lower() or
                            re.search(r"testthat::|test_check\s*\(|library\s*\(\s*['\"]?testthat", raw))
            if key in files:
                raise ValueError("Duplicate test output identity: " + key)
            files[key] = {"sha256": sha(path), "counts": counts, "structured_testthat": testthat,
                          "count_status": "available" if counts is not None else "unavailable",
                          "count_unavailable_reason": None if counts is not None else "No structured testthat summary; raw output retained"}
    return {"files": files, "scope": "Structured testthat counters only; traditional R scripts also retain raw identity and actual R CMD check test status"}


def compare_test_counters(baseline, candidate):
    left, right = baseline["files"], candidate["files"]
    regressions, required_unavailable, unavailable = [], [], []
    compared = 0
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is not None and b is None:
            regressions.append("missing_candidate_test_output:" + name)
            continue
        if a is None:
            required_unavailable.append("missing_baseline_test_output:" + name)
            continue
        x, y = a["counts"], b["counts"]
        if x is None or y is None:
            unavailable.append(name)
            if a["structured_testthat"] or b["structured_testthat"]:
                required_unavailable.append("missing_structured_testthat_counters:" + name)
            continue
        compared += 1
        for kind in ("FAIL", "WARN", "SKIP"):
            if y[kind] > x[kind]:
                regressions.append("increased_testthat_" + kind + ":" + name)
        if y["PASS"] < x["PASS"]:
            regressions.append("reduced_testthat_PASS:" + name)
    status = "fail" if regressions else "unavailable" if required_unavailable else "pass" if compared else "not_applicable"
    return {"status": status, "no_candidate_counter_regression": not regressions and not required_unavailable,
            "structured_count_pass": True if status == "pass" else False if status in ("fail", "unavailable") else None,
            "compared_structured_files": compared, "regressions": regressions,
            "required_unavailable": required_unavailable, "unavailable_count_files": unavailable,
            "all_test_output_counts_available": bool(left or right) and not unavailable and not required_unavailable,
            "scope": "No missing candidate Rout; FAIL/WARN/SKIP must not increase and PASS must not decrease for actual structured testthat counters. Non-testthat script counts remain unavailable; their actual R CMD check status is still checked."}


def parse_check(log, roots=()):
    text = Path(log).read_text(errors="replace")
    statuses = re.findall(r"^Status:\s*(.*)$", text, flags=re.M)
    if not statuses:
        raise ValueError("Missing actual R CMD check Status")
    status = statuses[-1].strip()
    counts = {x: 0 for x in ("ERROR", "WARNING", "NOTE")}
    for n, kind in re.findall(r"(\d+)\s+(ERROR|WARNING|NOTE)s?", status, re.I):
        counts[kind.upper()] = int(n)
    if status != "OK" and not any(counts.values()):
        raise ValueError("Unrecognized check status: " + status)
    blocks = {}
    for block in re.split(r"(?=^\* checking )", text, flags=re.M):
        if not block.startswith("* checking "):
            continue
        block = re.split(r"^\* DONE", block, maxsplit=1, flags=re.M)[0]
        header = block.splitlines()[0].split(" ...", 1)[0]
        first_line = block.splitlines()[0]
        # Multiline check results (especially tests) put the result on a
        # following line. Match a result line, never a quoted diagnostic word.
        result_text = first_line.split(" ...", 1)[1] if " ..." in first_line else ""
        if not result_text.strip():
            result_text = next((line.strip() for line in block.splitlines()[1:]
                                if re.fullmatch(r"\s*(?:OK|ERROR|WARNING|NOTE|SKIPPED)\s*", line)), "")
        severity = next((kind for kind in ("ERROR", "WARNING", "NOTE", "SKIPPED")
                         if re.search(r"\b" + kind + r"\b", result_text)), "OK")
        normalized = block
        for root in sorted(map(str, roots), key=len, reverse=True):
            normalized = normalized.replace(root, "<audit-path>")
        blocks[header] = {"severity": severity, "normalized_diagnostic": normalized.strip()}
    return {"status": status, "counts": counts, "blocks": blocks, "log_sha256": sha(log), "test_output_counters": test_output_counters(Path(log).parent)}


def compare(baseline, candidate):
    changed = []
    for kind in ("ERROR", "WARNING"):
        if candidate["counts"][kind] > baseline["counts"][kind]:
            changed.append("increased_" + kind)
    for name, block in candidate["blocks"].items():
        if block["severity"] in ("ERROR", "WARNING") and block != baseline["blocks"].get(name):
            changed.append(name)
    # Tests/examples may regress to missing or skipped even without a top-level error.
    for name, block in baseline["blocks"].items():
        if re.match(r"\* checking (tests|examples)$", name) and block["severity"] == "OK":
            other = candidate["blocks"].get(name)
            if other is None or other["severity"] != "OK" or re.search(r"\bSKIPPED\b", other["normalized_diagnostic"]):
                changed.append("tests_or_examples_not_completed:" + name)
    counter_gate = compare_test_counters(baseline["test_output_counters"], candidate["test_output_counters"])
    if not counter_gate["no_candidate_counter_regression"]:
        changed.append("test_output_counter_gate:" + counter_gate["status"])
    return {"no_candidate_regression": not changed, "test_output_counter_gate": counter_gate, "new_or_changed_candidate_issues": changed,
            "baseline_existing_issues": {k: v for k, v in baseline["blocks"].items() if v["severity"] in ("ERROR", "WARNING", "NOTE")},
            "both_clean": baseline["status"] == candidate["status"] == "OK"}


class Audit:
    def __init__(self, output, seconds):
        self.output, self.deadline = output, time.monotonic() + seconds
        self.evidence = output / "evidence"
        self.work = output / "work"
        self.evidence.mkdir(parents=True)
        self.work.mkdir()
        self.env = os.environ.copy()
        self.env.update(MAKEFLAGS="-j1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="1", NOT_CRAN="true", _R_CHECK_FORCE_SUGGESTS_="true",
                        R_ENVIRON_USER="/dev/null", R_PROFILE_USER="/dev/null",
                        R_LIBS="", R_LIBS_SITE=str(self.work / "absent-site-library"))

    def remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Overall audit deadline reached")
        return left

    def command(self, args, stem, cwd=None, env=None, timeout=1200, allow_nonzero=False):
        limit = min(timeout, self.remaining())
        log = self.evidence / (stem + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        row = {"command": args, "cwd": str(cwd or self.work), "timeout_seconds": limit,
               "started_utc": datetime.now(timezone.utc).isoformat(), "timed_out": False}
        with log.open("w") as stream:
            process = subprocess.Popen(args, cwd=cwd or self.work, env=env or self.env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            row["owned_process_group"] = process.pid
            try:
                row["returncode"] = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                row["timed_out"] = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # Kill the owned group even if its leader exited first.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                row["returncode"] = process.wait(timeout=5)
        row.update(ended_utc=datetime.now(timezone.utc).isoformat(), log_sha256=sha(log))
        write(self.evidence / (stem + ".command.json"), row)
        if not allow_nonzero and (row["timed_out"] or row["returncode"]):
            raise RuntimeError("Command failed or timed out: " + stem)
        return row

    def download(self, urls, path):
        attempts = []
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=min(60, self.remaining())) as response:
                    data = response.read(100 * 1048576 + 1)
                    if len(data) > 100 * 1048576:
                        raise ValueError("Source download exceeds 100MiB cap")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    row = {"url": url, "response_url": response.url, "headers": dict(response.headers),
                           "sha256": sha(path), "bytes": len(data), "attempts": attempts,
                           "utc": datetime.now(timezone.utc).isoformat()}
                    write(path.with_suffix(path.suffix + ".download.json"), row)
                    return row
            except urllib.error.HTTPError as exc:
                attempts.append({"url": url, "status": exc.code})
                if exc.code != 404:
                    raise
        raise RuntimeError("No pinned official source available: " + str(path))


def inventory(lib):
    return {str(p.relative_to(lib)): sha(p) for p in sorted(lib.rglob("*"))
            if p.is_file() and (p.name == "DESCRIPTION" or p.suffix in (".so", ".dll"))}


def compatible_archive(original, destination):
    """One explicit archived-dependency ABI variant; preserve the official tar."""
    original, destination = Path(original), Path(destination)
    if original.resolve() == destination.resolve():
        raise ValueError("Compatibility archive must not overwrite official archive")
    if sha(original) != "8757054042503a5fa1760d6a70a130b29d591233b7509c9c68e0ff30c456a7c1":
        raise ValueError("Compatibility input must be the pinned official networkTomography0.3")
    source = "networkTomography/src/ipfp.c"
    checksum = "networkTomography/MD5"
    before = b"&REAL(xx)[0], &incx, &beta, errVec, &incx);"
    after = b"&REAL(xx)[0], &incx, &beta, errVec, &incx FCONE);"
    with tarfile.open(original) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archived member")
        contents = {m.name: archive.extractfile(m).read() for m in members if m.isfile()}
    if hashlib.sha256(contents[source]).hexdigest() != "5b696c6abb9950e2c3898c787b0ce51ffd97253d909e16836de762ef7e4fc91e" or contents[source].count(before) != 1:
        raise ValueError("Compatibility call-site identity mismatch")
    changed = dict(contents)
    changed[source] = contents[source].replace(before, after)
    if hashlib.sha256(changed[source]).hexdigest() != "b057ff72be635f8183d53a1d8a0ac8040eea22d42b1a5eaf0c6b612493539636":
        raise ValueError("Compatibility patched source identity mismatch")
    old_md5 = b"7ecbb322265b8d78a975874160170093 *src/ipfp.c"
    new_md5 = b"44e93a85d66cd2aacd366eb6e6d63a9e *src/ipfp.c"
    if contents[checksum].count(old_md5) != 1:
        raise ValueError("Compatibility checksum identity mismatch")
    changed[checksum] = contents[checksum].replace(old_md5, new_md5)
    with destination.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for member in members:
                    if member.isfile():
                        member.size = len(changed[member.name])
                        archive.addfile(member, io.BytesIO(changed[member.name]))
                    else:
                        archive.addfile(member)
    with tarfile.open(destination) as archive:
        actual = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    if actual != changed or set(actual) != set(contents):
        raise ValueError("Compatibility archive changed unexpected contents")
    changed_names = sorted(name for name in contents if contents[name] != actual[name])
    if changed_names != sorted([source, checksum]):
        raise ValueError("Compatibility archive must change only ABI argument and its MD5 entry")
    return {"scope": "Explicit common dependency variant for both huge versions; original strict archived-dependency environment failed and is retained separately",
            "package": "networkTomography", "version": "0.3", "change": "Append FCONE to one dgemv character argument; update only its packaging MD5 entry",
            "original_tar": str(original), "original_tar_sha256": sha(original),
            "patched_tar": str(destination), "patched_tar_sha256": sha(destination),
            "changed_members": {name: {"original_sha256": hashlib.sha256(contents[name]).hexdigest(), "patched_sha256": hashlib.sha256(actual[name]).hexdigest()} for name in changed_names},
            "all_patched_file_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(actual.items())},
            "unchanged_content_members": len(contents) - 2,
            "original_strict_environment_passed": False,
            "same_shared_dependency_for_both_arms": True}


def run(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Actual reverse audit requires Linux; use selftest locally")
    out = Path(args.output).resolve()
    out.mkdir(exist_ok=False)
    audit = Audit(out, args.total_minutes * 60)
    result = {"completed": False, "no_candidate_regressions": False, "packages": [],
              "scope": "Actual current CRAN direct reverse dependencies, paired full tests/examples, no manuals/vignettes",
              "baseline_version": "2.0.1", "candidate_version": "2.0.2",
              "as_cran": False, "NOT_CRAN": "true", "dependency_epochs": []}
    try:
        tools_lib = audit.work / "tool-library"
        tools_lib.mkdir()
        audit.env["R_LIBS_USER"] = str(tools_lib)
        bootstrap = ('args<-commandArgs(TRUE); '
                     'install.packages("pak",lib=args[1],repos="https://cran.r-project.org",dependencies=NA); '
                     'stopifnot(normalizePath(find.package("pak",lib.loc=args[1]))==normalizePath(file.path(args[1],"pak"))); '
                     'write.csv(installed.packages(lib.loc=args[1])[,c("Package","Version","LibPath"),drop=FALSE],args[2],row.names=FALSE); '
                     'print(sessionInfo())')
        audit.command(["Rscript", "--vanilla", "-e", bootstrap, str(tools_lib), str(audit.evidence / "tool-inventory.csv")],
                      "bootstrap-pak", timeout=1200)
        write(audit.evidence / "tool-library-pins.json", inventory(tools_lib))
        index = audit.evidence / "downloads" / "PACKAGES.gz"
        audit.download([CRAN + "/src/contrib/PACKAGES.gz"], index)
        records = dcf(gzip.decompress(index.read_bytes()).decode())
        by_name = {r["Package"]: r for r in records}
        relations, names = discover(records)
        result["full_current_direct_reverse_count"] = len(names)
        result["priority_imports_not_rechecked"] = [name for name in names if name in PRIORITY]
        names = [name for name in names if name not in PRIORITY]
        result["scope"] = "Actual remaining direct reverse dependencies in a fresh common dependency epoch; earlier five Imports checks are separate evidence"
        result["earlier_imports_run"] = "https://github.com/Gatech-Flash/huge/actions/runs/37152328135/job/111289894465"
        if not names:
            raise RuntimeError("CRAN index produced no reverse dependencies")
        result.update(index_sha256=sha(index), relations=relations, priority_order=names,
                      actual_reverse_count=len(names), dependency_cap=args.max_dependencies)
        write(audit.evidence / "discovered.json", result)
        targets = {}
        for name in names + ["huge"]:
            version = "2.0.1" if name == "huge" else by_name[name]["Version"]
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", name) or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]*", version):
                raise ValueError("Unsafe package/version")
            path = audit.evidence / "downloads" / f"{name}_{version}.tar.gz"
            urls = [CRAN + f"/src/contrib/{path.name}", CRAN + f"/src/contrib/Archive/{name}/{path.name}"]
            if name == "huge" and by_name.get("huge", {}).get("Version") != version:
                urls = urls[1:]
            audit.download(urls, path)
            description = tar_description(path)
            if description["Package"] != name or description["Version"] != version:
                raise ValueError("Downloaded DESCRIPTION mismatch")
            targets[name] = {"version": version, "path": str(path), "sha256": sha(path),
                             "description": description}
        write(audit.evidence / "target-sources.json", targets)
        archive_pins = json.loads((HERE / "archived-optional-pins.json").read_text())
        archive_refs = {}
        for name, pin in archive_pins.items():
            path = audit.evidence / "downloads" / (name + "_" + pin["version"] + ".tar.gz")
            audit.download([pin["url"]], path)
            if sha(path) != pin["sha256"]:
                raise ValueError("Archived optional source SHA256 mismatch: " + name)
            archive_desc = tar_description(path)
            if (archive_desc["Package"], archive_desc["Version"]) != (name, pin["version"]):
                raise ValueError("Archived optional DESCRIPTION mismatch: " + name)
            if name == "networkTomography":
                patched = audit.evidence / "downloads" / "networkTomography_0.3.compat-fcone.tar.gz"
                compatibility = compatible_archive(path, patched)
                write(audit.evidence / "archived-dependency-compatibility.json", compatibility)
                result["archived_dependency_compatibility"] = compatibility
                result["scope"] += "; explicit common networkTomography FCONE compatibility variant, not the original strict dependency environment"
                path = patched
            archive_refs[name] = name + "=local::" + str(path)
        write(audit.evidence / "archived-optional-sources.json", archive_pins)
        source = Path(args.source).resolve()
        desc = dcf((source / "DESCRIPTION").read_text())[0]
        if (desc["Package"], desc["Version"]) != ("huge", "2.0.2"):
            raise ValueError("Candidate must be current huge2.0.2 source")
        current = audit.work / "source-current"
        current.mkdir()
        for path in source.iterdir():
            if path.is_dir() and path.name in ("R", "src", "man", "data", "inst", "vignettes", "tests"):
                shutil.copytree(path, current / path.name,
                                ignore=shutil.ignore_patterns("*.o", "*.so", "*.dll", "*.dylib", "__pycache__"))
            elif path.is_file() and (path.name in ("DESCRIPTION", "NAMESPACE", ".Rbuildignore", "NEWS.md", "configure", "configure.ac", "cleanup", "aclocal.m4", "config.guess", "config.sub", "install-sh") or path.name.startswith("LICENSE")):
                shutil.copy2(path, current / path.name)
        write(audit.evidence / "candidate-source-pins.json", {str(p.relative_to(current)): sha(p) for p in current.rglob("*") if p.is_file()})
        shared = audit.work / "shared-library"
        shared.mkdir()
        libs = {arm: audit.work / (arm + "-library") for arm in ("baseline", "candidate")}
        for lib in libs.values():
            lib.mkdir()
        core_names = sorted(set().union(*(set(dependencies(d.get(k, ""))) for d in
            (desc, tar_description(Path(targets["huge"]["path"]))) for k in ("Depends", "Imports", "LinkingTo"))) - {"R", "huge"})
        core_refs = audit.evidence / "core-refs.txt"
        core_refs.write_text("\n".join(core_names) + "\n")
        helper = str(HERE / "dependencies.R")
        audit.command(["Rscript", "--vanilla", helper, "core", str(shared), "", str(core_refs),
                       str(audit.evidence / "core-inventory.csv")], "install-core-dependencies", timeout=1800)
        build_env = audit.env.copy()
        build_env["R_LIBS_USER"] = str(shared)
        audit.command(["R", "CMD", "build", "--no-build-vignettes", str(current)], "build-current", env=build_env, timeout=1200)
        candidate_tar = audit.work / "huge_2.0.2.tar.gz"
        shutil.copy2(candidate_tar, audit.evidence / candidate_tar.name)
        huge_tars = {"baseline": Path(targets["huge"]["path"]), "candidate": candidate_tar}
        for arm, lib in libs.items():
            env = audit.env.copy()
            env["R_LIBS_USER"] = os.pathsep.join(map(str, (lib, shared)))
            audit.command(["R", "CMD", "INSTALL", "--library=" + str(lib), str(huge_tars[arm])],
                          "install-huge-" + arm, env=env)
        guards = {}
        preflight = audit.work / "preflight.R"
        preflight.write_text('args <- commandArgs(TRUE)\nlibrary(huge)\np <- normalizePath(find.package("huge"))\nstopifnot(p == normalizePath(file.path(args[1], "huge")), as.character(packageVersion("huge")) == args[2])\nd <- getLoadedDLLs()[["huge"]]\nstopifnot(!d[["dynamicLookup"]], startsWith(normalizePath(d[["path"]]), paste0(p, "/libs/")))\nwriteLines(c(p, as.character(packageVersion("huge")), normalizePath(d[["path"]])), args[3])\n')
        for arm, lib in libs.items():
            env = audit.env.copy()
            env["R_LIBS_USER"] = os.pathsep.join(map(str, (lib, shared)))
            version = "2.0.1" if arm == "baseline" else "2.0.2"
            guard_file = audit.evidence / ("huge-" + arm + "-guard.txt")
            audit.command(["Rscript", "--vanilla", str(preflight), str(lib), version, str(guard_file)], "preflight-" + arm, env=env)
            fields = guard_file.read_text().splitlines()
            binary = Path(fields[2])
            guards[arm] = {"package_path": fields[0], "version": fields[1], "DLL": fields[2], "DLL_sha256": sha(binary),
                           "source_tar_sha256": sha(huge_tars[arm]), "library_pins": inventory(lib)}
            retain = audit.evidence / "binaries" / arm / binary.name
            retain.parent.mkdir(parents=True)
            shutil.copy2(binary, retain)
        write(audit.evidence / "huge-guards.json", guards)
        groups = [("priority-imports", [name for name in names if name in PRIORITY]),
                  ("remaining-reverse", [name for name in names if name not in PRIORITY])]
        accumulated = set(core_names)
        known_dependencies = inventory(shared)
        group_index = 0
        for i, name in enumerate(names):
            while group_index < len(groups) and not groups[group_index][1]:
                group_index += 1
            epoch, group_names = groups[group_index]
            if name == group_names[0]:
                for target in group_names:
                    for key in ("Depends", "Imports", "LinkingTo", "Suggests"):
                        accumulated.update(dependencies(targets[target]["description"].get(key, "")))
                accumulated -= {"huge", "R"}
                refs = audit.evidence / (epoch + "-roots.txt")
                refs.write_text("\n".join(archive_refs.get(name, name) for name in sorted(accumulated)) + "\n")
                plan_lock = audit.evidence / (epoch + "-dependency-plan.lock.json")
                plan_csv = audit.evidence / (epoch + "-dependency-plan.csv")
                audit.command(["Rscript", "--vanilla", helper, "plan", str(shared), str(libs["baseline"]),
                               str(refs), str(plan_lock), str(plan_csv), str(audit.evidence / (epoch + "-pre-inventory.csv"))],
                              "plan-" + epoch, timeout=600)
                plan = write_lockfile_plan(plan_lock, plan_csv)
                closure = {row["package"] for row in plan if row["package"] not in ("huge", "R") and row.get("priority") != "base"}
                size = {"epoch": epoch, "actual_unique_dependency_closure": len(closure), "cap": args.max_dependencies,
                        "direct_roots": sorted(accumulated), "recursive_types": ["Depends", "Imports", "LinkingTo"],
                        "target_direct_Suggests_included": group_names,
                        "planning_api": "pak::lockfile_create with explicit shared/baseline libraries",
                        "raw_lockfile": str(plan_lock.relative_to(audit.evidence)),
                        "normalized_fields": ["package", "version", "ref", "type", "deps"],
                        "status_scope": "successful public solve required; status/priority/md5sum/mirror not synthesized"}
                write(audit.evidence / (epoch + "-dependency-size.json"), size)
                if len(closure) > args.max_dependencies:
                    raise RuntimeError("Dependency closure exceeds explicit cap; nothing trimmed")
                for target in names:
                    versions = {row["version"] for row in plan if row["package"] == target}
                    if versions and versions != {targets[target]["version"]}:
                        raise RuntimeError("pak target version differs from frozen CRAN index")
                audit.command(["Rscript", "--vanilla", helper, "install", str(shared), str(libs["baseline"]),
                               str(plan_csv), str(audit.evidence / (epoch + "-inventory.csv"))],
                              "install-" + epoch, timeout=2400)
                if (shared / "huge").exists():
                    raise RuntimeError("Shared library must not contain huge")
                dep_hashes = inventory(shared)
                if any(dep_hashes.get(key) != value for key, value in known_dependencies.items()):
                    raise RuntimeError("Later dependency preparation changed earlier dependency versions/native identities")
                if any(inventory(libs[arm]) != guards[arm]["library_pins"] for arm in libs):
                    raise RuntimeError("Dependency preparation changed an isolated huge installation")
                known_dependencies = dep_hashes
                write(audit.evidence / (epoch + "-shared-library-pins.json"), dep_hashes)
                result["dependency_epochs"].append({**size, "installed_identity_scope": "DESCRIPTION versions/hash and native .so/.dll hashes; not all dependency source bytes",
                                                     "pins_sha256": sha(audit.evidence / (epoch + "-shared-library-pins.json"))})
                write(audit.evidence / "results.json", result)
            item = {"package": name, "version": targets[name]["version"], "source_sha256": targets[name]["sha256"], "arms": {}}
            item["dependency_epoch"] = epoch
            result["packages"].append(item)
            for arm in (("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")):
                directory = audit.evidence / "checks" / name / arm
                directory.mkdir(parents=True)
                env = audit.env.copy()
                env["R_LIBS_USER"] = os.pathsep.join(map(str, (libs[arm], shared)))
                row = audit.command(["R", "CMD", "check", "--no-manual", "--no-build-vignettes", "--no-vignettes", targets[name]["path"]],
                    "checks/" + name + "/" + arm + "-process", cwd=directory, env=env, timeout=1200, allow_nonzero=True)
                log = directory / (name + ".Rcheck") / "00check.log"
                row["check"] = None
                if not row["timed_out"] and row["returncode"] in (0, 1):
                    try:
                        row["check"] = parse_check(log, (audit.work, libs[arm], directory))
                    except (OSError, ValueError) as exc:
                        row["check_parse_failure"] = str(exc)
                item["arms"][arm] = row
                if inventory(shared) != dep_hashes or inventory(libs[arm]) != guards[arm]["library_pins"]:
                    raise RuntimeError("A check changed shared dependencies or huge installation")
                write(audit.evidence / "results.json", result)
            if all(v["check"] is not None for v in item["arms"].values()):
                item["comparison"] = compare(item["arms"]["baseline"]["check"], item["arms"]["candidate"]["check"])
            else:
                item["comparison"] = {"no_candidate_regression": False, "both_clean": False,
                                       "unavailable": "An arm timed out, crashed or lacked an actual check Status"}
            write(audit.evidence / "results.json", result)
            if name == group_names[-1]:
                group_index += 1
        result.update(completed=all(p["arms"][a]["check"] is not None for p in result["packages"] for a in ("baseline", "candidate")),
                      no_candidate_regressions=all(p["comparison"]["no_candidate_regression"] for p in result["packages"]),
                      all_pairs_clean=all(p["comparison"]["both_clean"] for p in result["packages"]))
    except Exception as exc:
        result["setup_or_execution_failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        done = {p["package"] for p in result["packages"]}
        for name in result.get("priority_order", []):
            if name not in done:
                result["packages"].append({"package": name, "arms": {}, "not_run_reason": "Setup/deadline interruption; see setup_or_execution_failure"})
        result["pairs_with_actual_status"] = sum(len(p["arms"]) == 2 and all(a.get("check") is not None for a in p["arms"].values()) for p in result["packages"])
        result["packages_not_run"] = sum(not p["arms"] for p in result["packages"])
        result["packages_incomplete"] = len(result["packages"]) - result["pairs_with_actual_status"]
        write(audit.evidence / "results.json", result)
    print(json.dumps({k: result.get(k) for k in ("completed", "actual_reverse_count", "no_candidate_regressions", "all_pairs_clean", "setup_or_execution_failure")}))
    return 0 if result["completed"] and result["no_candidate_regressions"] else 1


def write_lockfile_plan(lock_path: Path, csv_path: Path) -> list[dict]:
    """Normalize only the installer's fields; retain the complete raw public lockfile."""
    raw = json.loads(lock_path.read_text())
    if raw.get("lockfile_version") != 1 or not isinstance(raw.get("packages"), list):
        raise ValueError("Unsupported public pak lockfile schema")
    plan = raw["packages"]
    fields = ("package", "version", "ref", "type")
    seen = set()
    deps = []
    for row in plan:
        if not all(isinstance(row.get(k), str) and row[k] for k in fields):
            raise ValueError("Lockfile lacks required package/ref/version/type fields")
        if row["package"] in seen:
            raise ValueError("Lockfile selected a package more than once")
        seen.add(row["package"])
        if not isinstance(row.get("deps"), list):
            raise ValueError("Lockfile lacks complete dependency records")
        for dep in row["deps"]:
            if not all(isinstance(dep.get(k), str) for k in ("ref", "type", "package", "op", "version")):
                raise ValueError("Lockfile has malformed dependency constraint fields")
            deps.append({"owner": row["package"], **{k: dep[k] for k in ("ref", "type", "package", "op", "version")}})
    expected = list(csv.DictReader(Path(str(lock_path) + ".expected.csv").open()))
    visible = json.loads(Path(str(lock_path) + ".visibility.lock.json").read_text())["packages"]
    for name, data in (("visibility", visible), ("full plan", plan)):
        chosen = {row["package"]: row for row in data}
        for row in expected:
            found = chosen.get(row["Package"])
            if found is None or found["version"] != row["Version"] or found["ref"] != row["Ref"]:
                raise ValueError(f"Explicit installed package not preserved by {name}: {row['Package']}")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in plan)
    with Path(str(csv_path) + ".deps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("owner", "ref", "type", "package", "op", "version"))
        writer.writeheader()
        writer.writerows(deps)
    return plan


def selftest(args):
    import tempfile
    rows = dcf("Package: first\nVersion: 1.0\nImports: stats,\n huge (>= 2.0.1)\nSuggests: utils\n\nPackage: second\nVersion: 1.0\nSuggests: huge\n")
    relations, names = discover(rows)
    assert relations["Imports"] == ["first"] and relations["Suggests"] == ["second"] and names == ["first", "second"]
    with tempfile.TemporaryDirectory() as name:
        p = Path(name) / "00check.log"
        p.write_text("* checking examples ... OK\n* checking tests ... OK\n* DONE\nStatus: OK\n")
        good = parse_check(p)
        assert good["counts"] == {"ERROR": 0, "WARNING": 0, "NOTE": 0}
        p.write_text("* checking examples ... ERROR\nchanged expectation\n* checking tests ... OK\n* DONE\nStatus: 1 ERROR\n")
        bad = parse_check(p)
        assert not compare(good, bad)["no_candidate_regression"]
        assert compare(bad, bad)["no_candidate_regression"] and not compare(bad, bad)["both_clean"]
        p.write_text("* checking examples ... ERROR\ndifferent failure\n* checking tests ... OK\n* DONE\nStatus: 1 ERROR\n")
        assert not compare(bad, parse_check(p))["no_candidate_regression"]
        p.write_text("* checking examples ... WARNING\nnew warning\n* checking tests ...\n Running 'testthat.R'\n OK\n* DONE\nStatus: 1 WARNING\n")
        warn = parse_check(p)
        assert warn["blocks"]["* checking tests"]["severity"] == "OK"
        assert not compare(good, warn)["no_candidate_regression"]
        p.write_text("* checking examples ... OK\n* checking tests ... SKIPPED\n* DONE\nStatus: OK\n")
        assert not compare(good, parse_check(p))["no_candidate_regression"]
        p.write_text("* checking examples ... OK\n* DONE\nStatus: OK\n")
        assert not compare(good, parse_check(p))["no_candidate_regression"]
        p.write_text("* checking examples ... OK\n* checking tests ...\n Running 'testthat.R'\n ERROR\nfailed checks\n* DONE\nStatus: 1 ERROR, 1 NOTE\n")
        multiline = parse_check(p)
        assert multiline["blocks"]["* checking tests"]["severity"] == "ERROR"
        assert multiline["counts"] == {"ERROR": 1, "WARNING": 0, "NOTE": 1}
    print(json.dumps({"passed": True, "scope": "synthetic DCF/check-status/parser gates only; no network/R/model/build", "controls": 9}))
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("run")
    q.add_argument("--source", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--total-minutes", type=int, default=90)
    q.add_argument("--max-dependencies", type=int, default=350)
    sub.add_parser("selftest")
    args = p.parse_args()
    if args.command == "run" and not (1 <= args.total_minutes <= 90 and 1 <= args.max_dependencies <= 350):
        p.error("Audit caps must be positive and at most 90 minutes / 350 dependencies")
    return run(args) if args.command == "run" else selftest(args)


if __name__ == "__main__":
    raise SystemExit(main())
