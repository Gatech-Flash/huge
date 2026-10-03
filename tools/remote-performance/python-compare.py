#!/usr/bin/env python3
"""Frozen, fresh-process Python GitHub/current workflow comparison.

No fixture generation, compilation or source edits occur in this harness.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
import warnings

THREAD_ENV = {
    "OMP_NUM_THREADS": os.environ.get("HUGE_AUDIT_OMP_THREADS", "1"), "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
}
OPERATIONS = {"fit", "fit_select", "npn", "roc", "inference", "screen_idx"}
API_SEED = 314159
ORDER_SEED = 20261005
MAX_BATCH = 4096
MAX_ESTIMATED_BATCH_SECONDS = 60.0


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stamp():
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "monotonic_ns": time.perf_counter_ns(),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}


def warning_rows(records):
    return [{"category": w.category.__name__, "message": str(w.message),
             "filename": str(w.filename), "lineno": w.lineno} for w in records]


def rng_state(np):
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"algorithm": algorithm, "keys": keys.tolist(), "position": position,
            "has_gauss": has_gauss, "cached_gaussian_hex": float(cached_gaussian).hex()}


def package_files(package):
    package = Path(package).resolve()
    rows = []
    for relative in ("pyhuge/__init__.py", "pyhuge/core.py", "pyhuge/parity.py",
                     "cpp/huge_core.cpp", "cpp/native_core_bindings.cpp",
                     "cpp/include/huge/huge_core.h", "cpp/include/huge/blas_config.h",
                     "pyproject.toml", "setup.py"):
        path = package / relative
        if path.is_file():
            rows.append({"path": str(path), "relative_path": relative,
                         "sha256": sha(path)})
    for path in sorted((package / "pyhuge").glob("_native_core*")):
        if path.is_file() and path.suffix in {".so", ".pyd", ".dylib"}:
            rows.append({"path": str(path), "relative_path": str(path.relative_to(package)),
                         "sha256": sha(path)})
    return rows


def input_snapshot(values, np, sparse):
    rows = {}
    for name, obj in sorted(values.items()):
        arrays = {"": obj} if isinstance(obj, np.ndarray) else {}
        if sparse.issparse(obj):
            rows[name] = {"format": obj.format, "shape": list(obj.shape)}
            for member in ("data", "indices", "indptr", "row", "col"):
                if hasattr(obj, member):
                    arrays["." + member] = getattr(obj, member)
        for suffix, array in arrays.items():
            rows[name + suffix] = {"shape": list(array.shape), "strides": list(array.strides),
                "dtype": array.dtype.str, "writeable": bool(array.flags.writeable),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    return rows


def load_workflow(spec, pyhuge, core, np, sparse):
    fixture = Path(spec["fixture"])
    loaded = np.load(fixture, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        values = {"x": loaded}
    else:
        with loaded:
            values = {name: loaded[name] for name in loaded.files}
    for name, option in spec.get("array_options", {}).items():
        if option.get("order") == "F":
            values[name] = np.asfortranarray(values[name])
        elif option.get("order") == "C":
            values[name] = np.ascontiguousarray(values[name])
        if option.get("readonly"):
            values[name].flags.writeable = False
    for name, payload in spec.get("sparse_payload", {}).items():
        shape = tuple(payload["shape"])
        fmt = payload.get("format", "coo")
        data = values[payload["data"]]
        if fmt == "coo":
            values[name] = sparse.coo_matrix((data, (values[payload["row"]],
                                                    values[payload["col"]])), shape=shape)
        elif fmt in {"csr", "csc"}:
            cls = sparse.csr_matrix if fmt == "csr" else sparse.csc_matrix
            values[name] = cls((data, values[payload["indices"]],
                               values[payload["indptr"]]), shape=shape)
        else:
            raise ValueError("Unsupported sparse fixture format: " + fmt)
    operation = spec["operation"]
    kwargs = dict(spec.get("kwargs", {}))
    if operation in {"fit", "fit_select", "npn", "roc"}:
        kwargs.setdefault("verbose", False)
    x_key = spec.get("fixture_fields", {}).get("x", "x")
    if operation == "fit":
        workflow = lambda: pyhuge.huge(values[x_key], **kwargs)
    elif operation == "fit_select":
        select_kwargs = {"n_jobs": 1, "verbose": False, **spec.get("select_kwargs", {})}
        if select_kwargs["n_jobs"] != 1:
            raise ValueError("This serial protocol requires selection n_jobs=1")
        def workflow():
            fitted = pyhuge.huge(values[x_key], **kwargs)
            return fitted, pyhuge.huge_select(fitted, **select_kwargs)
    elif operation == "npn":
        workflow = lambda: pyhuge.huge_npn(values[x_key], **kwargs)
    elif operation == "inference":
        fields = spec.get("fixture_fields", {})
        workflow = lambda: pyhuge.huge_inference(values[x_key], values[fields.get("t", "t")],
                                                  values[fields.get("adj", "adj")], **kwargs)
    elif operation == "roc":
        kwargs.setdefault("plot", False)
        keys = spec.get("path_keys", sorted(k for k in values if k.startswith("path_")
                    and k[5:].isdigit()))
        theta_key = spec.get("fixture_fields", {}).get("theta", "theta")
        storage = spec.get("storage", "dense")
        def convert(name, index=None):
            if sparse.issparse(values[name]):
                return values[name]
            fmt = storage.get(name, "dense") if isinstance(storage, dict) else storage
            if fmt == "mixed":
                fmt = "csc" if index is None or index % 2 else "dense"
            if name == theta_key:
                fmt = spec.get("theta_storage", fmt)
            if fmt == "dense":
                return values[name]
            if fmt not in {"csc", "csr", "coo"}:
                raise ValueError("Unsupported ROC storage: " + str(fmt))
            return getattr(sparse, fmt + "_matrix")(values[name])
        paths = [convert(key, index) for index, key in enumerate(keys)]
        theta = convert(theta_key)
        for index, value in enumerate(paths):
            values["api_path_" + str(index)] = value
        values["api_theta"] = theta
        workflow = lambda: pyhuge.huge_roc(paths, theta, **kwargs)
    elif operation == "screen_idx":
        count = kwargs.get("scr_num", kwargs.get("n_scr", kwargs.get("count", spec.get("screen_count"))))
        if count is None:
            raise ValueError("screen_idx needs kwargs.scr_num")
        workflow = lambda: core._build_screen_idx(values[x_key], int(count))
    else:
        raise ValueError("Unsupported operation: " + operation)
    return workflow, values


def measured_batch(workflow, count, np):
    gc.collect()
    np.random.seed(API_SEED)
    start_rng = rng_state(np)
    before = [dict(row) for row in gc.get_stats()]
    events, active = [], {}
    def callback(phase, info):
        now = time.perf_counter_ns()
        generation = int(info["generation"])
        if phase == "start":
            active[generation] = now
        elif phase == "stop":
            beginning = active.pop(generation, None)
            events.append({"generation": generation, "start_ns": beginning, "stop_ns": now,
                           "elapsed_ns": None if beginning is None else now - beginning,
                           "collected": int(info.get("collected", 0)),
                           "uncollectable": int(info.get("uncollectable", 0))})
    gc.callbacks.append(callback)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            for _ in range(count):
                value = workflow()
            wall_end = time.perf_counter_ns()
            cpu_end = time.process_time_ns()
    finally:
        gc.callbacks.remove(callback)
    after = [dict(row) for row in gc.get_stats()]
    counts = [after[g]["collections"] - before[g]["collections"] for g in range(3)]
    callback_counts = [sum(e["generation"] == g for e in events) for g in range(3)]
    for event in events:
        event["inside_timed_window"] = (event["start_ns"] is not None
            and event["start_ns"] >= wall_start and event["stop_ns"] <= wall_end)
    row = {"batch": count, "wall_ns": wall_end - wall_start, "cpu_ns": cpu_end - cpu_start,
           "batch_seconds": (wall_end - wall_start) / 1e9,
           "call_seconds": (wall_end - wall_start) / 1e9 / count,
           "cpu_call_seconds": (cpu_end - cpu_start) / 1e9 / count,
           "warnings": warning_rows(seen), "rng_start": start_rng, "rng_end": rng_state(np),
           "gc": {"collections_by_generation": counts, "callback_counts": callback_counts,
                  "counts_match": counts == callback_counts, "events": events,
                  "instrumentation_span_elapsed_ns": sum(e["elapsed_ns"] or 0 for e in events),
                  "timed_window_elapsed_ns": sum(e["elapsed_ns"] or 0 for e in events if e["inside_timed_window"]),
                  "timed_window_collections_by_generation": [sum(e["generation"] == g and e["inside_timed_window"] for e in events) for g in range(3)],
                  "wall_start_ns": wall_start, "wall_end_ns": wall_end,
                  "span_note": "Stats and full callback span include setup/teardown; timed-window events separately classified",
                  "get_stats_before": before, "get_stats_after": after}}
    return value, row


def encode_output(value, payload, np, sparse):
    arrays = []
    with Path(payload).open("wb") as sink:
        def encode(obj, name):
            if isinstance(obj, np.ndarray):
                if obj.dtype.hasobject:
                    raise TypeError("Object output arrays are not lossless numeric payloads: " + name)
                data = obj.tobytes(order="C")
                entry = {"field": name, "dtype": obj.dtype.str, "dtype_descr": obj.dtype.descr,
                         "shape": list(obj.shape), "bytes": len(data), "offset": sink.tell(),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "layout": {"strides": list(obj.strides),
                            "c_contiguous": bool(obj.flags.c_contiguous),
                            "f_contiguous": bool(obj.flags.f_contiguous),
                            "writeable": bool(obj.flags.writeable),
                            "own_data": bool(obj.flags.owndata),
                            "base_type": None if obj.base is None else type(obj.base).__name__}}
                sink.write(data)
                arrays.append(entry)
                return {"array": entry}
            if isinstance(obj, np.generic):
                return {"numpy_scalar": encode(np.asarray(obj), name)}
            if sparse.issparse(obj):
                buffers = {}
                for member in ("data", "indices", "indptr", "row", "col"):
                    if hasattr(obj, member):
                        buffers[member] = encode(getattr(obj, member), name + "." + member)
                return {"sparse": obj.format, "shape": list(obj.shape), "buffers": buffers,
                        "class": type(obj).__name__}
            if dataclasses.is_dataclass(obj):
                return {"dataclass": type(obj).__name__,
                        "fields": {field.name: encode(getattr(obj, field.name), name + "." + field.name)
                                   for field in dataclasses.fields(obj)}}
            if isinstance(obj, dict):
                return {"dict": [[encode(key, name + ".<key>"), encode(item, name + "." + str(key))]
                                 for key, item in obj.items()]}
            if isinstance(obj, (list, tuple)):
                return {"sequence": type(obj).__name__,
                        "items": [encode(item, name + "[" + str(index) + "]")
                                  for index, item in enumerate(obj)]}
            if isinstance(obj, float):
                return {"float_hex": obj.hex()}
            if isinstance(obj, bytes):
                return {"bytes_hex": obj.hex()}
            if obj is None or isinstance(obj, (str, bool, int)):
                return obj
            raise TypeError("Unsupported output type at " + name + ": " + str(type(obj)))
        manifest = encode(value, "result")
    return manifest, arrays


def workload_summary(value, spec, inputs):
    """Inspect final work outside timers; never label a truncated path complete."""
    expected = spec.get("expected_work", {})
    operation = spec["operation"]
    observed = {"operation": operation, "expected": expected, "contract_errors": []}
    fields = spec.get("fixture_fields", {})
    x = inputs.get(fields.get("x", "x"))
    if x is not None:
        observed["input_shape"] = list(x.shape)
        if expected.get("n") is not None and x.shape[0] != expected["n"]:
            observed["contract_errors"].append("input_observations")
        if expected.get("d") is not None and x.shape[1] != expected["d"]:
            observed["contract_errors"].append("input_dimension")
    if operation in {"fit", "fit_select"}:
        fitted = value[0] if operation == "fit_select" else value
        observed["returned_path_length"] = len(fitted.path)
        observed["returned_lambda_hex"] = [float(v).hex() for v in fitted.lambda_path]
        requested = expected.get("requested_path_length")
        observed["requested_path_length"] = requested
        observed["all_paths_empty"] = all(path.nnz == 0 for path in fitted.path)
        if len(fitted.path) != len(fitted.lambda_path):
            observed["contract_errors"].append("path_lambda_lengths")
        if not fitted.path:
            observed["contract_errors"].append("empty_certified_path")
        if requested is not None:
            if len(fitted.path) > requested or (len(fitted.path) < requested and not expected.get("allow_truncation", False)):
                observed["contract_errors"].append("unexpected_path_length")
        if expected.get("expected_all_paths_empty") and not observed["all_paths_empty"]:
            observed["contract_errors"].append("expected_empty_graphs")
        if operation == "fit_select":
            selected = value[1]
            observed["selection_criterion"] = selected.criterion
            observed["refit_shape"] = list(selected.refit.shape)
            observed["variability_shape"] = None if selected.variability is None else list(selected.variability.shape)
            criterion = expected.get("resolved_criterion", spec.get("select_kwargs", {}).get("criterion"))
            if criterion is not None and selected.criterion != criterion:
                observed["contract_errors"].append("selection_criterion")
    elif operation == "roc":
        observed["returned_path_length"] = len(value.f1)
        requested = expected.get("path_length")
        observed["requested_path_length"] = requested
        if requested is not None and observed["returned_path_length"] != requested:
            observed["contract_errors"].append("ROC_path_length")
    observed["contract_ok"] = not observed["contract_errors"]
    return observed


def worker(args):
    os.environ.update(THREAD_ENV)
    result = {"schema": 1, "mode": args.mode, "start": stamp(), "status": "running"}
    progress = Path(args.output).with_suffix(".progress.json")
    def stage(name):
        result["stage"] = name
        write(progress, {"stage": name, "stamp": stamp()})
    try:
        stage("imports")
        package = Path(args.package).resolve()
        sys.path.insert(0, str(package))
        import numpy as np
        import scipy
        from scipy import sparse
        import pyhuge
        from pyhuge import core
        package_actual = Path(pyhuge.__file__).resolve().parent
        if package_actual != package / "pyhuge":
            raise RuntimeError("Package import escaped requested tree: " + str(package_actual))
        native_path = Path(core._CPP.__file__).resolve()
        if not native_path.is_relative_to(package_actual):
            raise RuntimeError("Native import escaped requested tree: " + str(native_path))
        native_threads = int(core._CPP.omp_max_threads())
        if native_threads != int(THREAD_ENV["OMP_NUM_THREADS"]):
            raise RuntimeError("Unexpected omp_max_threads; observed " + str(native_threads))
        # -I removes the script directory; add only this frozen audit utility.
        sys.path.insert(1, str(Path(__file__).resolve().parent))
        from blas_identity import identity
        actual_blas = identity(core._CPP.__file__)
        spec = json.loads(Path(args.spec).read_text())
        fixture = Path(spec["fixture"])
        if sha(fixture) != spec["fixture_sha256"]:
            raise RuntimeError("Frozen fixture fingerprint mismatch")
        result.update(case=spec.get("id", spec.get("name")), operation=spec["operation"],
            spec_sha256=sha(args.spec), metadata={"python": sys.version, "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__, "pyhuge": pyhuge.__version__,
            "platform": platform.platform(), "package": str(package), "package_file": str(pyhuge.__file__),
            "core_file": str(core.__file__), "core_sha256": sha(core.__file__),
            "native_file": str(native_path), "native_sha256": sha(native_path),
            "actual_blas": actual_blas,
            "blas_identity_sha256": sha(Path(__file__).with_name("blas_identity.py")),
            "native_omp_max_threads": native_threads, "fixture": str(fixture),
            "fixture_sha256": sha(fixture), "thread_environment": dict(THREAD_ENV),
            "package_files": package_files(package)})
        stage("load_fixture")
        workflow, inputs = load_workflow(spec, pyhuge, core, np, sparse)
        initial_inputs = input_snapshot(inputs, np, sparse)
        result["input_before"] = initial_inputs
        stage("warmups")
        np.random.seed(API_SEED)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            for _ in range(5):
                value = workflow()
        result["warmup_warnings"] = warning_rows(seen)
        del value
        if args.mode == "calibrate":
            result["calibration_rounds"] = []
            count = 1
            while True:
                stage("calibration_batch_" + str(count))
                value, timing = measured_batch(workflow, count, np)
                result["calibration_rounds"].append(timing)
                del value
                write(args.output, result)
                if timing["batch_seconds"] >= args.target_seconds or count == MAX_BATCH:
                    break
                count = min(MAX_BATCH, count * 2)
            result["suggested_batch"] = count
        else:
            stage("timed_batch")
            result["before_batch"] = stamp()
            value, timing = measured_batch(workflow, args.batch, np)
            result.update(timing)
            result["after_batch"] = stamp()
            result["target_met"] = result["batch_seconds"] >= args.target_seconds
            result["workload"] = workload_summary(value, spec, inputs)
            stage("input_verification")
            result["input_after"] = input_snapshot(inputs, np, sparse)
            result["input_unchanged"] = initial_inputs == result["input_after"]
            stage("serialize_final_output")
            payload = Path(args.output).with_suffix(".bin")
            result["manifest"], arrays = encode_output(value, payload, np, sparse)
            result["array_count"] = len(arrays)
            result["payload"] = {"filename": payload.name, "bytes": payload.stat().st_size,
                                 "sha256": sha(payload)}
            del value
        if "input_after" not in result:
            result["input_after"] = input_snapshot(inputs, np, sparse)
            result["input_unchanged"] = initial_inputs == result["input_after"]
        if not result["input_unchanged"]:
            raise RuntimeError("API mutated fixture inputs")
        result["status"] = "passed"
        stage("finished")
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        result["end"] = stamp()
        write(args.output, result)
        raise
    result["end"] = stamp()
    write(args.output, result)


def decode_array(node, path, np):
    entry = node["array"]
    with Path(path).open("rb") as stream:
        stream.seek(entry["offset"])
        data = stream.read(entry["bytes"])
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("Array payload fingerprint mismatch: " + entry["field"])
    dtype = np.dtype(entry["dtype"])
    if dtype.fields is not None:
        raise ValueError("Structured arrays need dtype-descriptor decoding")
    return np.frombuffer(data, dtype=dtype).reshape(entry["shape"])


def compare_outputs(first, second, first_path, second_path):
    import numpy as np
    from scipy import sparse
    records, differences = [], []
    def difference(field, kind, left=None, right=None):
        differences.append({"field": field, "kind": kind, "baseline": left, "candidate": right})
    def arrays(a_node, b_node, field, exact=False):
        a, b = decode_array(a_node, first_path, np), decode_array(b_node, second_path, np)
        for key in ("dtype", "layout"):
            if a_node["array"][key] != b_node["array"][key]:
                difference(field, key, a_node["array"][key], b_node["array"][key])
        same_shape = a.shape == b.shape
        numeric = a.dtype.kind in "fciub" and b.dtype.kind in "fciub"
        normalized_field = field.replace('"', '')
        graph_field = any(token in normalized_field for token in (".path[", ".refit", ".graph", ".adjacency"))
        exact = exact or graph_field or not numeric or a.dtype.kind in "iub" or b.dtype.kind in "iub"
        passed, maximum = False, None
        if same_shape:
            if exact:
                passed = bool(np.array_equal(a, b, equal_nan=True)) if numeric else bool(np.array_equal(a, b))
            else:
                passed = bool(np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True))
            if numeric:
                finite = np.isfinite(a) & np.isfinite(b)
                if np.any(finite):
                    maximum = float(np.max(np.abs(a[finite].astype(complex) - b[finite].astype(complex))))
        records.append({"field": field, "kind": "array", "passed": passed,
                        "shape_baseline": list(a.shape), "shape_candidate": list(b.shape),
                        "comparison": "exact" if exact else "atol=rtol=1e-10",
                        "maximum_absolute_difference": maximum,
                        "byte_exact": a_node["array"]["sha256"] == b_node["array"]["sha256"]})
        return a, b
    def sparse_decode(node, path):
        buffers = {key: decode_array(value, path, np) for key, value in node["buffers"].items()}
        fmt = node["sparse"]
        if fmt == "coo":
            value = sparse.coo_matrix((buffers["data"], (buffers["row"], buffers["col"])),
                                      shape=node["shape"])
        elif fmt in {"csr", "csc"}:
            value = getattr(sparse, fmt + "_matrix")((buffers["data"], buffers["indices"],
                                                     buffers["indptr"]), shape=node["shape"])
        else:
            raise ValueError("Unsupported sparse output: " + fmt)
        value = value.tocsc(copy=True)
        value.sum_duplicates()
        value.eliminate_zeros()
        value.sort_indices()
        return value
    def walk(a, b, field):
        if not isinstance(a, dict) or not isinstance(b, dict):
            records.append({"field": field, "kind": "scalar", "passed": type(a) is type(b) and a == b})
            return
        if "array" in a and "array" in b:
            arrays(a, b, field)
        elif "numpy_scalar" in a and "numpy_scalar" in b:
            walk(a["numpy_scalar"], b["numpy_scalar"], field)
        elif "sparse" in a and "sparse" in b:
            if a["sparse"] != b["sparse"] or a["class"] != b["class"]:
                difference(field, "sparse_format", a["sparse"], b["sparse"])
            if set(a["buffers"]) != set(b["buffers"]):
                difference(field, "sparse_buffer_names", sorted(a["buffers"]), sorted(b["buffers"]))
            for key in sorted(set(a["buffers"]) & set(b["buffers"])):
                if a["buffers"][key]["array"]["sha256"] != b["buffers"][key]["array"]["sha256"]:
                    difference(field + "." + key, "stored_sparse_buffer")
                for metadata in ("dtype", "layout"):
                    left_meta = a["buffers"][key]["array"][metadata]
                    right_meta = b["buffers"][key]["array"][metadata]
                    if left_meta != right_meta:
                        difference(field + "." + key, metadata, left_meta, right_meta)
            left, right = sparse_decode(a, first_path), sparse_decode(b, second_path)
            passed = (left.shape == right.shape and np.array_equal(left.indptr, right.indptr)
                      and np.array_equal(left.indices, right.indices)
                      and np.array_equal(left.data, right.data, equal_nan=True))
            records.append({"field": field, "kind": "sparse_graph", "passed": bool(passed),
                            "comparison": "exact canonical coordinates and values"})
        elif "float_hex" in a and "float_hex" in b:
            left, right = float.fromhex(a["float_hex"]), float.fromhex(b["float_hex"])
            passed = (math.isnan(left) and math.isnan(right)) or left == right or (
                math.isfinite(left) and math.isfinite(right)
                and abs(left - right) <= 1e-10 + 1e-10 * abs(left))
            records.append({"field": field, "kind": "float", "passed": passed,
                            "byte_exact": a == b})
        elif "dataclass" in a and "dataclass" in b:
            keys = set(a["fields"]) | set(b["fields"])
            records.append({"field": field, "kind": "dataclass", "passed": a["dataclass"] == b["dataclass"]
                            and set(a["fields"]) == set(b["fields"])})
            for key in sorted(keys & set(a["fields"]) & set(b["fields"])):
                walk(a["fields"][key], b["fields"][key], field + "." + key)
        elif "sequence" in a and "sequence" in b:
            records.append({"field": field, "kind": "sequence", "passed":
                a["sequence"] == b["sequence"] and len(a["items"]) == len(b["items"])})
            for index, (left, right) in enumerate(zip(a["items"], b["items"])):
                walk(left, right, field + "[" + str(index) + "]")
        elif "dict" in a and "dict" in b:
            def keymap(value):
                return {json.dumps(key, sort_keys=True): item for key, item in value["dict"]}
            left, right = keymap(a), keymap(b)
            records.append({"field": field, "kind": "dict", "passed": set(left) == set(right)})
            for key in sorted(set(left) & set(right)):
                walk(left[key], right[key], field + "." + key)
        else:
            records.append({"field": field, "kind": "type_or_metadata", "passed": a == b})
    walk(first, second, "result")
    return {"passed": all(row["passed"] for row in records), "fields": records,
            "metadata_differences": differences,
            "field_count": len(records), "failed_fields": sum(not row["passed"] for row in records)}


def inference_decision_gate(first, second, first_path, second_path, alpha):
    """Require finite tested edges, matching finite masks and edge decisions."""
    import numpy as np
    alpha = float(alpha)
    report = {"passed": False, "alpha": alpha, "alpha_hex": alpha.hex(),
              "scope": "off-diagonal p-values; diagonal NaNs are excluded",
              "comparison": "exact finite masks and p<alpha decisions; finite tested edges required"}
    try:
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Inference alpha must be finite and in (0,1]")
        a = decode_array(first["fields"]["p"], first_path, np)
        b = decode_array(second["fields"]["p"], second_path, np)
        if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("Inference p arrays must have matching square shapes")
        if a.dtype.kind != "f" or b.dtype.kind != "f":
            raise ValueError("Inference p arrays must have real floating dtype")
        offdiag = ~np.eye(a.shape[0], dtype=bool)
        finite = [np.isfinite(x)[offdiag] for x in (a, b)]
        decisions = [(x < alpha)[offdiag] for x in (a, b)]
        report.update(shape=list(a.shape), tested_directed_edges=int(np.count_nonzero(offdiag)),
                      finite_masks_exact=bool(np.array_equal(*finite)),
                      both_offdiag_finite=bool(all(x.all() for x in finite)),
                      decisions_exact=bool(np.array_equal(*decisions)),
                      decision_mismatch_count=int(np.count_nonzero(decisions[0] != decisions[1])))
        for arm, finite_mask, decision_mask in zip(("baseline", "candidate"), finite, decisions):
            report[arm] = {"finite_mask_sha256": hashlib.sha256(finite_mask.tobytes()).hexdigest(),
                           "decision_mask_sha256": hashlib.sha256(decision_mask.tobytes()).hexdigest(),
                           "finite_edge_count": int(np.count_nonzero(finite_mask)),
                           "significant_edge_count": int(np.count_nonzero(decision_mask))}
        report["passed"] = report["finite_masks_exact"] and report["both_offdiag_finite"] and report["decisions_exact"]
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return report


def schedule(cases, trials, seed):
    order_by_case = {}
    for name in cases:
        rng = random.Random(seed + int(hashlib.sha256(name.encode()).hexdigest()[:12], 16))
        orders = []
        while len(orders) + 4 <= trials:
            first = ["baseline", "candidate"] if rng.randrange(2) == 0 else ["candidate", "baseline"]
            reverse = list(reversed(first))
            orders.extend([first, reverse, reverse, first])
        while len(orders) < trials:
            residual = ["baseline", "candidate"]
            rng.shuffle(residual)
            orders.append(residual)
        order_by_case[name] = orders
    rng = random.Random(seed)
    result = []
    for trial in range(trials):
        shuffled = list(cases)
        rng.shuffle(shuffled)
        for name in shuffled:
            result.append({"case": name, "trial": trial + 1, "order": order_by_case[name][trial]})
    return result


def load_grid(grid_path):
    path = Path(grid_path).resolve()
    grid = json.loads(path.read_text())
    items = grid["cases"] if isinstance(grid, dict) else grid
    cases = []
    for item in items:
        source = path
        if isinstance(item, str):
            source = (path.parent / item).resolve()
            item = json.loads(source.read_text())
        spec = dict(item)
        spec["id"] = spec.get("id", spec.get("name"))
        if not spec["id"] or not all(c.isalnum() or c in "-_." for c in spec["id"]):
            raise ValueError("Case IDs must be safe filenames")
        if spec["operation"] not in OPERATIONS:
            raise ValueError("Unknown operation: " + spec["operation"])
        fixture = Path(spec["fixture"])
        if not fixture.is_absolute():
            fixture = source.parent / fixture
        spec["fixture"] = str(fixture.resolve())
        observed = sha(fixture)
        if observed != spec.get("fixture_sha256"):
            raise ValueError("Grid fixture must declare matching sha256: " + spec["id"])
        cases.append(spec)
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    return cases


def invoke(args, output, package, spec_path, stem, mode, batch=1):
    workers = output / "workers"
    target = workers / (stem + ".json")
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "worker",
               "--package", str(package), "--spec", str(spec_path), "--mode", mode,
               "--batch", str(batch), "--target-seconds", str(args.target_seconds),
               "--output", str(target)]
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env.pop("PYTHONPATH", None)
    started = stamp()
    timed_out = False
    try:
        completed = subprocess.run(command, cwd=output, env=env, capture_output=True,
                                   text=True, timeout=args.timeout)
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        returncode = None
    (workers / (stem + ".stdout.log")).write_text(stdout)
    (workers / (stem + ".stderr.log")).write_text(stderr)
    row = {"stem": stem, "command": command, "mode": mode, "batch": batch,
           "start": started, "end": stamp(), "returncode": returncode,
           "timed_out": timed_out, "worker_path": str(target.relative_to(output))}
    if target.exists():
        try:
            row["worker"] = json.loads(target.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            row["read_error"] = str(exc)
    progress = target.with_suffix(".progress.json")
    if progress.exists():
        row["progress"] = json.loads(progress.read_text())
    row["successful"] = not timed_out and returncode == 0 and row.get("worker", {}).get("status") == "passed"
    write(workers / (stem + ".process.json"), row)
    print(stem, "complete" if row["successful"] else "TIMEOUT" if timed_out else "FAILED", flush=True)
    return row


def provenance_ok(row, frozen_files, fixture_hash, spec_hash):
    metadata = row.get("metadata", {})
    expected = {entry["path"]: entry["sha256"] for entry in frozen_files}
    observed = {entry["path"]: entry["sha256"] for entry in metadata.get("package_files", [])}
    return (metadata.get("fixture_sha256") == fixture_hash
            and row.get("spec_sha256") == spec_hash
            and metadata.get("native_omp_max_threads") == int(THREAD_ENV["OMP_NUM_THREADS"])
            and expected.get(metadata.get("core_file")) == metadata.get("core_sha256")
            and expected.get(metadata.get("native_file")) == metadata.get("native_sha256")
            and expected == observed)


def work_classification(first, second):
    left, right = first.get("workload", {}), second.get("workload", {})
    if not left.get("contract_ok") or not right.get("contract_ok"):
        return "unexpected_work"
    if left != right:
        return "different_work"
    requested, returned = left.get("requested_path_length"), left.get("returned_path_length")
    if requested is not None and returned is not None and returned < requested:
        return "same_prefix"
    return "full"


def analyze(output, results, plan):
    if "timing_ended" not in results:
        raise RuntimeError("Comparison/statistics prohibited before timing_ended")
    import numpy as np
    comparisons = output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    summaries = []
    for case_index, row in enumerate(results["cases"]):
        frozen_spec = next(item for item in plan["cases"] if item["id"] == row["id"])
        ratios, raw_ratios, cpus, deltas = [], [], [], []
        groups = collections.defaultdict(lambda: {"ratios": [], "cpu_ratios": [], "differences_seconds": []})
        for pair in row["pairs"]:
            workers = pair["workers"]
            if not all(value["successful"] for value in workers.values()):
                pair["comparison"] = {"status": "unavailable_worker_failure"}
                continue
            baseline, candidate = (workers[arm]["worker"] for arm in ("baseline", "candidate"))
            payloads = [(output / workers[arm]["worker_path"]).with_suffix(".bin")
                        for arm in ("baseline", "candidate")]
            try:
                for value, payload in zip((baseline, candidate), payloads):
                    if sha(payload) != value["payload"]["sha256"]:
                        raise ValueError("Full payload fingerprint mismatch")
                qc = compare_outputs(baseline["manifest"], candidate["manifest"], *payloads)
            except BaseException as exc:
                qc = {"passed": False, "error": {"type": type(exc).__name__, "message": str(exc)},
                      "traceback": traceback.format_exc()}
            qc["provenance_ok"] = all(provenance_ok(value, plan["package_files"][arm],
                                                       row["spec"]["fixture_sha256"], frozen_spec["spec_sha256"])
                for arm, value in (("baseline", baseline), ("candidate", candidate)))
            qc["inputs_unchanged"] = baseline["input_unchanged"] and candidate["input_unchanged"]
            qc["gc_counts_match"] = baseline["gc"]["counts_match"] and candidate["gc"]["counts_match"]
            qc["rng_start_equal"] = baseline["rng_start"] == candidate["rng_start"]
            qc["rng_end_equal"] = baseline["rng_end"] == candidate["rng_end"]
            qc["workload_class"] = work_classification(baseline, candidate)
            normalized_warning = lambda items: [{"category": w["category"], "message": w["message"]} for w in items]
            qc["warnings_equal"] = normalized_warning(baseline["warnings"]) == normalized_warning(candidate["warnings"])
            qc["warmup_warnings_equal"] = normalized_warning(baseline["warmup_warnings"]) == normalized_warning(candidate["warmup_warnings"])
            allowed_dtype_fields = row["spec"].get("allowed_dtype_differences", [])
            qc["dtype_contract_ok"] = all(item["field"] in allowed_dtype_fields
                for item in qc.get("metadata_differences", []) if item["kind"] == "dtype")
            qc["numerical_equivalence"] = qc["passed"]
            qc["inference_decision_equivalence"] = True
            if row["spec"]["operation"] == "inference":
                qc["inference_decisions"] = inference_decision_gate(
                    baseline["manifest"], candidate["manifest"], *payloads,
                    row["spec"].get("kwargs", {}).get("alpha", 0.05))
                qc["inference_decision_equivalence"] = qc["inference_decisions"]["passed"]
            qc["value_dtype_warning_equivalence"] = qc["passed"] and qc["dtype_contract_ok"] and qc["warnings_equal"] and qc["warmup_warnings_equal"]
            qc["strict_api_equivalence"] = qc["value_dtype_warning_equivalence"] and qc["inference_decision_equivalence"] and not qc.get("metadata_differences", [])
            qc["eligible"] = (qc["value_dtype_warning_equivalence"] and qc["provenance_ok"]
                and qc["inputs_unchanged"] and qc["gc_counts_match"] and qc["rng_start_equal"]
                and qc["rng_end_equal"] and qc["inference_decision_equivalence"]
                and qc["workload_class"] in {"full", "same_prefix"})
            qc_path = comparisons / (row["id"] + "-pair" + str(pair["trial"]) + ".json")
            write(qc_path, qc)
            pair["comparison"] = {"status": "passed" if qc["eligible"] else "failed",
                                   "path": str(qc_path.relative_to(output)),
                                   "workload_class": qc["workload_class"],
                                   "numerical_equivalence": qc["numerical_equivalence"],
                                   "inference_decision_equivalence": qc["inference_decision_equivalence"],
                                   "value_dtype_warning_equivalence": qc["value_dtype_warning_equivalence"],
                                   "strict_api_equivalence": qc["strict_api_equivalence"],
                                   "warning_difference": not qc["warnings_equal"],
                                   "metadata_difference_count": len(qc.get("metadata_differences", []))}
            ratio = candidate["call_seconds"] / baseline["call_seconds"]
            pair["raw_candidate_over_baseline"] = ratio
            raw_ratios.append(ratio)
            pair["target_met_both"] = baseline["target_met"] and candidate["target_met"]
            if qc["eligible"]:
                ratios.append(ratio)
                cpus.append(candidate["cpu_call_seconds"] / baseline["cpu_call_seconds"])
                deltas.append(candidate["call_seconds"] - baseline["call_seconds"])
                group = groups[qc["workload_class"]]
                group["ratios"].append(ratio)
                group["cpu_ratios"].append(cpus[-1])
                group["differences_seconds"].append(deltas[-1])
        summary = {"id": row["id"], "operation": row["spec"]["operation"], "planned_pairs": plan["trials"],
                   "returned_pairs": len(raw_ratios), "eligible_pairs": len(ratios),
                   "raw_candidate_over_baseline": raw_ratios,
                   "eligible_candidate_over_baseline": ratios,
                   "target_miss_workers": sum(not value["worker"].get("target_met", False)
                        for pair in row["pairs"] for value in pair["workers"].values() if value["successful"]),
                   "calibration_failed": row.get("calibration_failed", False)}
        summary["by_workload_class"] = {}
        for group_index, (work_class, group) in enumerate(sorted(groups.items())):
            seed = ORDER_SEED + case_index * 10 + group_index
            rng = np.random.default_rng(seed)
            values = np.asarray(group["ratios"])
            medians = np.median(values[rng.integers(0, len(values), size=(10000, len(values)))], axis=1)
            detail = {"eligible_pairs": len(values), "median_ratio": float(np.median(values)),
                      "descriptive_bootstrap_95_interval": np.quantile(medians, [.025, .975]).tolist(),
                      "bootstrap_replicates": 10000, "bootstrap_seed": seed,
                      "median_cpu_ratio": statistics.median(group["cpu_ratios"]),
                      "median_call_difference_seconds": statistics.median(group["differences_seconds"]),
                      "candidate_slower_pairs": int(sum(value > 1 for value in values))}
            summary["by_workload_class"][work_class] = detail
        if len(groups) == 1:
            summary["workload_class"] = next(iter(groups))
            summary.update(summary["by_workload_class"][summary["workload_class"]])
        summaries.append(summary)
    results["analysis"] = {"ratio_direction": "current/GitHub; below1faster",
                           "bootstrap_scope": "Descriptive paired median intervals; no universal equivalence/no-regression claim",
                           "cases": summaries, "end": stamp()}
    write(output / "analysis-summary.json", results["analysis"])
    write(output / "results.json", results)


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "workers").mkdir()
    (output / "specs").mkdir()
    cases = load_grid(args.grid)
    packages = {"baseline": Path(args.baseline).resolve(), "candidate": Path(args.candidate).resolve()}
    files = {arm: package_files(package) for arm, package in packages.items()}
    if not all(any("_native_core" in item["relative_path"] for item in group) for group in files.values()):
        raise RuntimeError("Both package trees need already-built native extensions")
    specs = {}
    for spec in cases:
        path = output / "specs" / (spec["id"] + ".json")
        write(path, spec)
        specs[spec["id"]] = path
    case_names = [spec["id"] for spec in cases]
    calibrations_order = list(case_names)
    random.Random(ORDER_SEED - 1).shuffle(calibrations_order)
    plan = {"schema": 1, "frozen_before_any_worker": stamp(), "grid_file": str(Path(args.grid).resolve()),
            "grid_sha256": sha(args.grid), "runner_file": str(Path(__file__).resolve()),
            "runner_sha256": sha(__file__), "package_files": files,
            "cases": [{"id": spec["id"], "spec_sha256": sha(specs[spec["id"]]),
                       "fixture": spec["fixture"], "fixture_sha256": spec["fixture_sha256"]} for spec in cases],
            "trials": args.trials, "warmups": 5, "api_seed_once_per_batch": API_SEED,
            "target_seconds": args.target_seconds, "worker_timeout_seconds": args.timeout,
            "maximum_batch": MAX_BATCH, "maximum_estimated_batch_seconds": MAX_ESTIMATED_BATCH_SECONDS,
            "calibration": "Fresh workers both arms, doubling1..4096towardtarget; commonmaxcounts, boundedby60sestimate",
            "calibration_case_order": calibrations_order,
            "timed_schedule": schedule(case_names, args.trials, ORDER_SEED), "order_seed": ORDER_SEED,
            "comparison_deferred_until": "all timed workers ended",
            "output_policy": "Retain every worker output/log/progress/payload/failure; no outlier removal or selected reruns",
            "timing_scope": "Full API workflow with identical GC callback and always-warning capture instrumentation; imports/fixture conversion/warmup/fullGC/serialization excluded",
            "operation_scope": "fit_select includes initial fit plus selection and actual refit where applicable; fit, npn, roc and inference measure their complete respective API calls, not a universal fit-selection-evaluation chain",
            "instrumentation_limit": "Callbacks and captured warnings add overhead, especially tiny controls; GC fullspan and events strictly within timer are separately retained",
            "rng_limit": "Records legacy global NumPy RNG only; StARS/RIC use internal default_rng(0), so global states do not certify internal random work",
            "float_tolerance": {"atol": 1e-10, "rtol": 1e-10},
            "exact_scope": "Integer/index arrays and canonical sparse graph values/support",
            "inference_decision_gate": "Frozen kwargs.alpha (default0.05); exact finite off-diagonal p masks and p<alpha decisions; all tested edges finite; record alpha hex, mask SHA256 and counts",
            "metadata_policy": "Layout/sparse storage differences recorded separately; strict equivalence additionally requires same warnings and dtype unless explicitly allowed",
            "workload_policy": "Full requested paths and identical certified prefixes classified and summarized separately; unexpected/different work excluded from quality ratios"}
    write(output / "frozen-plan.json", plan)
    results = {"schema": 1, "start": stamp(), "frozen_plan_sha256": sha(output / "frozen-plan.json"),
               "cases": [{"id": spec["id"], "spec": spec, "pairs": [], "calibrations": {}} for spec in cases]}
    by_name = {row["id"]: row for row in results["cases"]}
    write(output / "results.json", results)
    for index, name in enumerate(calibrations_order):
        row = by_name[name]
        arms = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        for arm in arms:
            row["calibrations"][arm] = invoke(args, output, packages[arm], specs[name],
                                               name + "-" + arm + "-calibrate", "calibrate")
            process = row["calibrations"][arm]
            frozen_spec = next(item for item in plan["cases"] if item["id"] == name)
            process["provenance_ok"] = process["successful"] and provenance_ok(
                process["worker"], plan["package_files"][arm], row["spec"]["fixture_sha256"],
                frozen_spec["spec_sha256"])
            process["calibration_quality_ok"] = (process["provenance_ok"]
                and process["worker"].get("input_unchanged", False)
                and all(round_["gc"]["counts_match"] for round_ in process["worker"].get("calibration_rounds", [])))
            write(output / "results.json", results)
        if not all(value["calibration_quality_ok"] for value in row["calibrations"].values()):
            row["calibration_failed"] = True
            row["timed_workers_skipped_reason"] = "Cannot establish calibrated common workload"
            continue
        successful = [value["worker"] for value in row["calibrations"].values()]
        requested = max(value["suggested_batch"] for value in successful)
        slowest = max(value["calibration_rounds"][-1]["call_seconds"] for value in successful)
        budget_count = max(1, math.floor(MAX_ESTIMATED_BATCH_SECONDS / slowest))
        row["common_batch"] = min(requested, MAX_BATCH, budget_count)
        row["batch_decision"] = {"requested_common_batch": requested, "slowest_calibrated_call_seconds": slowest,
                                 "estimated_60second_budget_count": budget_count,
                                 "bounded_common_batch": row["common_batch"],
                                 "target_misses_retained": True}
        write(output / "results.json", results)
    results["calibration_ended"] = stamp()
    write(output / "results.json", results)
    for scheduled in plan["timed_schedule"]:
        name = scheduled["case"]
        row = by_name[name]
        if row.get("calibration_failed"):
            continue
        pair = {**scheduled, "workers": {}}
        row["pairs"].append(pair)
        for arm in scheduled["order"]:
            stem = name + "-" + arm + "-trial" + str(scheduled["trial"])
            pair["workers"][arm] = invoke(args, output, packages[arm], specs[name], stem, "timed", row["common_batch"])
            write(output / "results.json", results)
    results["timing_ended"] = stamp()
    write(output / "timing-ended.json", results["timing_ended"])
    write(output / "results.json", results)
    analyze(output, results, plan)
    results["final_package_files"] = {arm: package_files(package) for arm, package in packages.items()}
    results["source_files_unchanged"] = results["final_package_files"] == plan["package_files"]
    results["end"] = stamp()
    write(output / "results.json", results)
    print("FINISHED", len(cases), "cases; all payloads retained; source_unchanged=", results["source_files_unchanged"], flush=True)


def selftest(args):
    """Synthetic serializer/comparator/schedule test; never imports pyhuge."""
    import tempfile
    import numpy as np
    from scipy import sparse
    @dataclasses.dataclass
    class Synthetic:
        data: object
        graph: object
        values: object
    with tempfile.TemporaryDirectory(prefix="huge-github-harness-selftest-") as temporary:
        directory = Path(temporary)
        a = np.asarray([[1., -0.], [np.nan, np.inf]])
        first = Synthetic(a, sparse.csc_matrix([[0., 1.], [1., 0.]]),
                          {"tuple": (np.int32(2), None, b"ab", -0.0), "list": [True, "x"]})
        second = Synthetic(np.asfortranarray(a), first.graph.copy(), first.values)
        left, _ = encode_output(first, directory / "first.bin", np, sparse)
        right, _ = encode_output(second, directory / "second.bin", np, sparse)
        good = compare_outputs(left, right, directory / "first.bin", directory / "second.bin")
        assert good["passed"] and any(x["kind"] == "layout" for x in good["metadata_differences"])
        dense_left, _ = encode_output({"graph": np.asarray([[0., 1.], [1., 0.]])}, directory / "dense-first.bin", np, sparse)
        dense_right, _ = encode_output({"graph": np.asarray([[0., 1. + 1e-11], [1., 0.]])}, directory / "dense-second.bin", np, sparse)
        # Dict fields are quoted in the comparison pathname; exact graph
        # identity must also apply to raw native dictionaries.
        dense_check = compare_outputs(dense_left, dense_right, directory / "dense-first.bin", directory / "dense-second.bin")
        assert not dense_check["passed"]
        second.graph[0, 1] = 2
        right, _ = encode_output(second, directory / "second.bin", np, sparse)
        bad = compare_outputs(left, right, directory / "first.bin", directory / "second.bin")
        assert not bad["passed"]
        generated = schedule(["a", "b", "c"], 5, ORDER_SEED)
        assert len(generated) == 15 and len({(row["case"], row["trial"]) for row in generated}) == 15
        for name in ("a", "b", "c"):
            orders = [row["order"] for row in generated if row["case"] == name]
            assert orders[0] == orders[3] and orders[1] == orders[2] and orders[0] == list(reversed(orders[1]))
        value, measured = measured_batch(lambda: (gc.collect(), np.arange(3)), 4, np)
        assert measured["batch"] == 4 and measured["gc"]["counts_match"]
        assert measured["gc"]["timed_window_collections_by_generation"][2] == 4
        complete = {"workload": {"contract_ok": True, "requested_path_length": 10,
                                  "returned_path_length": 10, "returned_lambda_hex": ["0x1p+0"] * 10}}
        prefix = {"workload": {**complete["workload"], "returned_path_length": 4,
                               "returned_lambda_hex": ["0x1p+0"] * 4}}
        assert work_classification(complete, complete) == "full"
        assert work_classification(prefix, prefix) == "same_prefix"
        assert work_classification(complete, prefix) == "different_work"
        @dataclasses.dataclass
        class SyntheticInference:
            data: object
            p: object
            error: float
        alpha, epsilon = .2, 5e-12
        inference = SyntheticInference(np.ones((2, 2)),
                                       np.asarray([[np.nan, .1], [.3, np.nan]]), .25)
        perturbed = SyntheticInference(inference.data,
                                       np.asarray([[np.nan, .1 + epsilon], [.3 - epsilon, np.nan]]), .25)
        left_path, right_path = directory / "inference-left.bin", directory / "inference-right.bin"
        left_manifest, _ = encode_output(inference, left_path, np, sparse)
        right_manifest, _ = encode_output(perturbed, right_path, np, sparse)
        assert compare_outputs(left_manifest, right_manifest, left_path, right_path)["passed"]
        positive = inference_decision_gate(left_manifest, right_manifest, left_path, right_path, alpha)
        assert positive["passed"] and positive["baseline"]["decision_mask_sha256"] == positive["candidate"]["decision_mask_sha256"]
        inference.p = np.asarray([[np.nan, alpha - epsilon], [alpha + epsilon, np.nan]])
        perturbed.p = np.asarray([[np.nan, alpha + epsilon], [alpha - epsilon, np.nan]])
        left_manifest, _ = encode_output(inference, left_path, np, sparse)
        right_manifest, _ = encode_output(perturbed, right_path, np, sparse)
        assert compare_outputs(left_manifest, right_manifest, left_path, right_path)["passed"]
        negative = inference_decision_gate(left_manifest, right_manifest, left_path, right_path, alpha)
        assert not negative["passed"] and negative["decision_mismatch_count"] == 2
        assert negative["baseline"]["significant_edge_count"] == negative["candidate"]["significant_edge_count"] == 1
        perturbed.p[0, 1] = np.nan
        right_manifest, _ = encode_output(perturbed, right_path, np, sparse)
        nonfinite = inference_decision_gate(left_manifest, right_manifest, left_path, right_path, alpha)
        assert not nonfinite["passed"] and not nonfinite["finite_masks_exact"]
        print(json.dumps({"status": "passed", "scope": "synthetic only, no pyhuge imports/models",
                          "checks": ["lossless tree", "layout separated", "graph exactness", "ABBA schedule", "timer/GC callback",
                                     "inference decision positive", "threshold crossing despite same error count", "inference offdiag finite mask"],
                          "inference_controls": {"positive": positive, "negative_same_count": negative, "negative_nonfinite": nonfinite}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    for name in ("grid", "baseline", "candidate", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--target-seconds", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=150.0)
    p = sub.add_parser("worker")
    for name in ("package", "spec", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("calibrate", "timed"), required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--target-seconds", type=float, default=1.0)
    sub.add_parser("selftest")
    args = parser.parse_args()
    os.environ.update(THREAD_ENV)
    if args.command == "run":
        if args.trials < 1 or args.target_seconds <= 0 or not math.isfinite(args.target_seconds) or args.timeout <= 0 or not math.isfinite(args.timeout):
            parser.error("Trials, target and timeout must be positive finite values")
        run(args)
    elif args.command == "worker":
        if not 1 <= args.batch <= MAX_BATCH:
            parser.error("Worker batch must be1..4096")
        worker(args)
    else:
        selftest(args)


if __name__ == "__main__":
    main()
