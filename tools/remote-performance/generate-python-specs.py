#!/usr/bin/env python3
"""Generate pinned fixtures/specs without importing huge or running any model.

Run once with the comparison interpreter after reviewing this file. Existing
grid/fixture/spec outputs are rejected rather than overwritten. NumPy data
values are float64; sparse coordinate indices are int64. The artifact hashes
are established from the actual NPZ bytes, not a seed alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


UPSTREAM_COMMIT = "a2452e490934aae2212327faa2c013adc3c6383f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float64)


def ar_data(n: int, d: int, rho: float, seed: int) -> np.ndarray:
    values = normal(n, d, seed)
    scale = np.sqrt(1.0 - rho * rho)
    for column in range(1, d):
        values[:, column] = rho * values[:, column - 1] + scale * values[:, column]
    return values


def block_covariance(d: int, block_size: int, rho: float) -> np.ndarray:
    indices = np.arange(d)
    corr = rho ** np.abs(indices[:, None] - indices[None, :])
    corr[indices[:, None] // block_size != indices[None, :] // block_size] = 0.0
    sd = np.geomspace(0.5, 2.0, d)
    return (corr * sd[:, None]) * sd[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.out.resolve()
    fixtures = out / "python-fixtures"
    specs = out / "python-specs"
    output_paths = [fixtures, specs, out / "python-grid.json", out / "python-fixture-manifest.json"]
    present = [str(path) for path in output_paths if path.exists()]
    if present:
        raise SystemExit("Refusing to overwrite existing outputs: " + ", ".join(present))
    fixtures.mkdir(parents=True)
    specs.mkdir()
    cases = []
    identities = []

    def add(identifier, operation, arrays, kwargs, *, select_kwargs=None,
            seed=None, structure=None, expected_work=None, default_omissions=(),
            corner_flags=(), sparse_payload=None, storage=None, path_keys=None,
            correctness_notes=(), layout_notes=None):
        fixture = fixtures / (identifier + ".npz")
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        for key, value in arrays.items():
            if value.dtype.kind == "f" and value.dtype != np.dtype("float64"):
                raise ValueError("Floating fixture must be float64: " + key)
            if value.dtype.kind not in "fiu b".replace(" ", ""):
                raise ValueError("Fixture requires numeric values: " + key)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite fixture field: " + key)
        np.savez_compressed(fixture, **arrays)
        fixture_hash = sha256(fixture)
        spec = {
            "schema": 1, "id": identifier, "operation": operation,
            "fixture": str(fixture.relative_to(out)), "fixture_sha256": fixture_hash,
            "kwargs": kwargs, "fixture_seed": seed,
            "structure": structure,
            "default_omissions": list(default_omissions),
            "expected_work": expected_work or {},
            "corner_flags": list(corner_flags),
            "output_scope": "complete_public_result" if operation != "screen_idx" else "internal_helper_diagnostic",
            "correctness_notes": list(correctness_notes),
        }
        if select_kwargs is not None:
            spec["select_kwargs"] = select_kwargs
        if sparse_payload is not None:
            spec["sparse_payload"] = sparse_payload
        if storage is not None:
            spec["storage"] = storage
        if path_keys is not None:
            spec["path_keys"] = path_keys
        if layout_notes is not None:
            spec["expected_layout_difference"] = layout_notes
        spec_path = specs / (identifier + ".json")
        spec_path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
        cases.append(spec)
        identities.append({
            "id": identifier, "fixture": spec["fixture"], "fixture_sha256": fixture_hash,
            "fixture_bytes": fixture.stat().st_size,
            "spec": str(spec_path.relative_to(out)), "spec_sha256": sha256(spec_path),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                              "logical_c_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
                       for key, value in arrays.items()},
        })

    # Five CT comparisons retain both ordinary defaults and the two historical
    # unfavorable controls. Only parameters shown in kwargs are passed.
    ct128 = normal(100, 128, 20261011)
    ct512 = normal(100, 512, 20261012)
    add("py-ct-default-d128", "fit", {"x": ct128},
        {"method": "ct", "input_type": "data", "verbose": False},
        seed=20261011, structure="independent Gaussian observations",
        default_omissions=("lambda_", "nlambda", "lambda_min_ratio"),
        expected_work={"n": 100, "d": 128, "requested_path_length": 20,
                       "resolved_default_lambda_min_ratio": 0.05, "allow_truncation": False})
    add("py-ct-default-d512", "fit", {"x": ct512},
        {"method": "ct", "input_type": "data", "verbose": False},
        seed=20261012, structure="independent Gaussian observations",
        default_omissions=("lambda_", "nlambda", "lambda_min_ratio"),
        expected_work={"n": 100, "d": 512, "requested_path_length": 20,
                       "resolved_default_lambda_min_ratio": 0.05, "allow_truncation": False})
    add("py-ct-longpath-d64-l512", "fit", {"x": normal(100, 64, 20261013)},
        {"method": "ct", "input_type": "data", "nlambda": 512,
         "lambda_min_ratio": 1.0, "verbose": False},
        seed=20261013, structure="independent Gaussian observations",
        corner_flags=("many_order_statistics", "full_density_endpoint", "prior_minor_cost"),
        expected_work={"n": 100, "d": 64, "requested_path_length": 512,
                       "lambda_min_ratio": 1.0, "allow_truncation": False})
    add("py-ct-stars-default-d512-b5", "fit_select", {"x": ct512},
        {"method": "ct", "input_type": "data", "verbose": False},
        select_kwargs={"rep_num": 5, "n_jobs": 1, "verbose": False},
        seed=20261012, structure="same observations as py-ct-default-d512",
        default_omissions=("lambda_", "nlambda", "lambda_min_ratio", "criterion",
                           "stars_subsample_ratio", "stars_thresh"),
        expected_work={"n": 100, "d": 512, "requested_path_length": 20,
                       "resolved_criterion": "stars", "rep_num": 5,
                       "resolved_subsample_ratio": 0.8, "subsample_observations": 80,
                       "fit_plus_all_selection_fits": True, "allow_truncation": False})
    add("py-ct-stars-empty-d256-l4-b5", "fit_select",
        {"x": normal(100, 256, 20261014)},
        {"method": "ct", "input_type": "data", "lambda_": [1.0] * 4, "verbose": False},
        select_kwargs={"rep_num": 5, "n_jobs": 1, "verbose": False},
        seed=20261014, structure="independent Gaussian data; lambda=1 yields no graph edges",
        corner_flags=("empty_support", "coo_wrapper_overhead", "prior_minor_cost", "repeated_lambda"),
        default_omissions=("criterion", "stars_subsample_ratio", "stars_thresh"),
        expected_work={"n": 100, "d": 256, "requested_path_length": 4,
                       "resolved_criterion": "stars", "rep_num": 5,
                       "resolved_subsample_ratio": 0.8, "subsample_observations": 80,
                       "fit_plus_all_selection_fits": True, "expected_all_paths_empty": True,
                       "allow_truncation": False})

    # A fresh NumPy population/seed family, not a claimed reproduction of the
    # previous R benchmark's observations or R RNG stream.
    mb_base = normal(1000, 64, 20261015)
    for identifier, values, seed, flags, description in (
        ("py-mb-tall-n1000-d64", mb_base, 20261015, ("mb_packing_watch",), "original NumPy observation seed"),
        ("py-mb-tall-n1000-d64-fresh", normal(1000, 64, 20261016), 20261016,
         ("mb_packing_watch", "fresh_observation_seed"), "fresh observations from the same independent population"),
        ("py-mb-tall-n1000-d64-reverse", mb_base[:, ::-1], 20261015,
         ("mb_packing_watch", "reversed_column_order"), "exact original observations with variables reversed"),
    ):
        add(identifier, "fit", {"x": values},
            {"method": "mb", "input_type": "data", "verbose": False},
            seed=seed, structure=description, corner_flags=flags,
            default_omissions=("lambda_", "nlambda", "lambda_min_ratio", "scr", "sym"),
            expected_work={"n": 1000, "d": 64, "requested_path_length": 10,
                           "resolved_scr": False, "resolved_sym": "or", "allow_truncation": False})
    add("py-mb-strong-ar1-n200-d64", "fit", {"x": ar_data(200, 64, 0.8, 20261017)},
        {"method": "mb", "input_type": "data", "verbose": False},
        seed=20261017, structure="Gaussian AR(1) observations with rho=0.8; chain population precision",
        corner_flags=("correlated_active_set",),
        default_omissions=("lambda_", "nlambda", "lambda_min_ratio", "scr", "sym"),
        expected_work={"n": 200, "d": 64, "requested_path_length": 10,
                       "resolved_scr": False, "allow_truncation": False})
    screening = normal(40, 600, 20261018)
    screening[:, 1::2] = 0.8 * screening[:, ::2] + 0.6 * screening[:, 1::2]
    add("py-mb-screened-n40-d600-k39", "fit", {"x": screening},
        {"method": "mb", "input_type": "data", "lambda_": [0.7, 0.5],
         "scr": True, "scr_num": 39, "verbose": False},
        seed=20261018, structure="Gaussian correlated pairs in small-sample high dimension",
        corner_flags=("partial_screening", "screening_39_le_d_over_4", "n_lt_d"),
        expected_work={"n": 40, "d": 600, "requested_path_length": 2,
                       "scr": True, "scr_num": 39, "allow_truncation": False})

    owner_note = {
        "icov_and_cov": "Values/dtypes/shapes/writability must match; local independent F-contiguous capsule owners replace upstream C-contiguous shared cube views.",
        "private_binding_default": "matrix_list=False must retain its C-contiguous ndarray cube contract.",
        "layout_is_not_numerical_failure": True,
    }
    add("py-glasso-blockcov-d128-l24-cov", "fit",
        {"x": block_covariance(128, 8, 0.65)},
        {"method": "glasso", "input_type": "covariance",
         "lambda_": np.geomspace(0.8, 0.08, 24).tolist(), "cov_output": True, "verbose": False},
        structure="deterministic SPD block AR(1), blocks of 8, rho=.65, marginal sd .5..2",
        corner_flags=("independent_matrix_ownership", "covariance_output", "long_precision_path"),
        expected_work={"n": None, "d": 128, "requested_path_length": 24,
                       "cov_output": True, "allow_truncation": False}, layout_notes=owner_note)
    add("py-tiger-default-n100-d128", "fit", {"x": normal(100, 128, 20261019)},
        {"method": "tiger", "input_type": "data", "verbose": False},
        seed=20261019, structure="independent Gaussian observations; generated path may truncate",
        default_omissions=("lambda_", "nlambda", "lambda_min_ratio", "sym"),
        corner_flags=("generated_prefix", "n_lt_d", "deferred_precision"),
        expected_work={"n": 100, "d": 128, "requested_path_length": 10,
                       "allow_truncation": True, "equal_actual_prefix_required": True},
        layout_notes=owner_note)
    indices = np.arange(64)
    add("py-tiger-covariance-full-d64-l3", "fit",
        {"x": 0.4 ** np.abs(indices[:, None] - indices[None, :])},
        {"method": "tiger", "input_type": "covariance", "lambda_": [0.6, 0.3, 0.15],
         "verbose": False},
        structure="deterministic SPD AR(1) correlation rho=.4",
        corner_flags=("full_certified_control", "independent_matrix_ownership"),
        expected_work={"n": None, "d": 64, "requested_path_length": 3,
                       "allow_truncation": False}, layout_notes=owner_note)

    tied = np.asarray([[0, 1, 0, 7], [0, 2, 1, 7], [1, 2, 1, 7], [1, 3, 2, 7],
                       [2, 3, 2, 7], [2, 4, 3, 7], [3, 4, 3, 7], [3, 5, 4, 7]], dtype=np.float64)
    large = normal(1000, 1024, 20261020)
    for mode in ("shrinkage", "truncation"):
        add("py-npn-tied-small-" + mode, "npn", {"x": tied},
            {"npn_func": mode, "verbose": False},
            structure="ties and one constant column, deterministic 8x4 observations",
            corner_flags=("rank_ties", "constant_column", "small_wrapper_cost"),
            expected_work={"n": 8, "d": 4, "npn_func": mode})
        add("py-npn-large-" + mode, "npn", {"x": large},
            {"npn_func": mode, "verbose": False}, seed=20261020,
            structure="continuous independent Gaussian observations, same data for both score modes",
            corner_flags=("ndtri_large_probability_array",),
            expected_work={"n": 1000, "d": 1024, "npn_func": mode})

    # Canonical sparse fixture keeps O(d) actual edges, including both graph
    # triangles. Dense NPZ matrices are fixture storage only; the worker builds
    # CSC inputs before timing. Mixed fixture explicitly preserves COO duplicates.
    d = 1024
    theta = np.zeros((d, d), dtype=np.float64)
    edge = np.arange(d - 1)
    theta[edge, edge + 1] = theta[edge + 1, edge] = 1.0
    p0 = theta.copy()
    p0[::2, :] = 0.0
    p0[:, ::2] = 0.0
    p1 = theta.copy()
    p1[edge[:-1], edge[:-1] + 2] = p1[edge[:-1] + 2, edge[:-1]] = 1.0
    add("py-roc-sparse-d1024", "roc", {"theta": theta, "path_0": p0,
                                           "path_1": p1, "path_2": theta},
        {"verbose": False, "plot": False}, storage="csc",
        path_keys=["path_0", "path_1", "path_2"],
        structure="deterministic chain truth; path_0 exactly empty, other predictions have O(d) edges",
        corner_flags=("sparse_no_densification", "empty_prediction_support"),
        expected_work={"n": None, "d": d, "path_length": 3,
                       "expected_path_0_empty": True,
                       "counting": "Python strict upper triangle"})
    d = 128
    rows = np.asarray([0, 0, 0, 0, 2, 3, 7, 9], dtype=np.int64)
    cols = np.asarray([1, 4, 1, 1, 3, 2, 7, 10], dtype=np.int64)
    values = np.asarray([1e16, 2.0, -1e16, 1.0, -3.0, 8.0, 7.0, 0.0], dtype=np.float64)
    truth = np.zeros((d, d), dtype=np.float64)
    np.add.at(truth, (rows, cols), values)
    pred_rows = np.asarray([0, 0, 2, 2, 4, 5, 9], dtype=np.int64)
    pred_cols = np.asarray([1, 4, 3, 3, 5, 4, 9], dtype=np.int64)
    pred_values = np.asarray([1.0, 0.0, 3.0, -3.0, -2.0, 9.0, 4.0], dtype=np.float64)
    pred = np.zeros((d, d), dtype=np.float64)
    np.add.at(pred, (pred_rows, pred_cols), pred_values)
    dense = np.zeros((d, d), dtype=np.float64)
    dense[0, 1], dense[0, 4], dense[4, 0], dense[8, 9] = 1.0, -2.0, 5.0, 3.0
    add("py-roc-mixed-asymmetric-duplicates-d128", "roc",
        {"theta": truth, "theta_row": rows, "theta_col": cols, "theta_data": values,
         "path_0": pred, "path_0_row": pred_rows, "path_0_col": pred_cols,
         "path_0_data": pred_values, "path_1": dense, "path_2": truth},
        {"verbose": False, "plot": False},
        storage={"theta": "coo", "path_0": "coo", "path_1": "dense", "path_2": "csc"},
        sparse_payload={name: {"format": "coo", "shape": [d, d], "data": name + "_data",
                               "row": name + "_row", "col": name + "_col"}
                        for name in ("theta", "path_0")},
        path_keys=["path_0", "path_1", "path_2"],
        structure="noncanonical float64 COO duplicates, explicit zeros, asymmetric/diagonal weights; mixed dense/CSC paths",
        corner_flags=("duplicate_cancellation", "storage_order", "mixed_storage", "asymmetry", "ignored_diagonal"),
        expected_work={"n": None, "d": d, "path_length": 3,
                       "counting": "Python strict upper triangle"},
        correctness_notes=("Worker must not canonicalize/sum COO duplicates before the API call.",))

    for d, seed in ((129, 20261021), (130, 20261022), (258, 20261023)):
        add("py-ct-ric-d" + str(d) + "-r10", "fit_select", {"x": normal(100, d, seed)},
            {"method": "ct", "input_type": "data", "verbose": False},
            select_kwargs={"criterion": "ric", "rep_num": 10, "verbose": False},
            seed=seed, structure="independent Gaussian observations",
            default_omissions=("lambda_", "nlambda", "lambda_min_ratio"),
            corner_flags=("ric_panel_boundary", "refit_and_selection_metadata"),
            expected_work={"n": 100, "d": d, "requested_path_length": 20,
                           "criterion": "ric", "rep_num": 10,
                           "rep_num_is_explicit_not_default": True,
                           "panels": (d - 2) // 128 + 1,
                           "fit_plus_selection_and_refit": True, "allow_truncation": False})

    x = np.round(normal(19, 5, 20261024))
    t = np.eye(5) + normal(5, 5, 20261025) * 0.04
    np.fill_diagonal(t, 1.0)
    adj = np.zeros((5, 5), dtype=np.float64)
    adj[0, 1] = -2.0
    for method in ("score", "wald"):
        add("py-inference-nonparanormal-" + method, "inference", {"x": x, "t": t, "adj": adj},
            {"type_": "Nonparanormal", "method": method, "alpha": 0.2},
            seed={"data": 20261024, "t": 20261025},
            structure="rounded Gaussian ties with nonsymmetric positive-diagonal T and weighted adjacency",
            corner_flags=("asymmetric_t", "rank_ties", "score_variance_contraction"),
            expected_work={"n": 19, "d": 5, "method": method},
            correctness_notes=("T is intentionally asymmetric and valid in both APIs.",
                               "Compare NaN masks separately; diagonal p-values do not represent tested edges."))

    if len(cases) != 24 or len({case["id"] for case in cases}) != 24:
        raise AssertionError("Expected exactly 24 unique planned cases")
    grid = {
        "schema": 1, "upstream_commit": UPSTREAM_COMMIT,
        "measurement_scope": "24 complete Python public workflow cases; no internal helper timers",
        "fixture_generation": "NumPy only; no huge import/model/build/test execution",
        "case_count": len(cases), "cases": cases,
    }
    grid_path = out / "python-grid.json"
    grid_path.write_text(json.dumps(grid, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema": 1, "generator_sha256": sha256(Path(__file__)),
        "generator_command": [sys.executable, str(Path(__file__).resolve()), "--out", str(out)],
        "python": sys.version, "numpy_version": np.__version__,
        "upstream_commit": UPSTREAM_COMMIT,
        "grid_sha256": sha256(grid_path), "case_count": len(cases), "fixtures": identities,
        "source_only_generation": True,
        "notes": ["These are new NumPy fixtures, not earlier R RNG/observations.",
                  "Seed and default omissions describe generation/API contracts; actual binary fixture hashes pin inputs.",
                  "RIC rep_num=10 is explicit; Python API default is 20.",
                  "Mixed ROC sparse payloads preserve raw COO duplicate storage.",
                  "Both builds must see identical fixtures and keyword omissions.",
                  "No generated benchmark result is present in this manifest."],
    }
    (out / "python-fixture-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_count": len(cases), "grid": str(grid_path),
                      "grid_sha256": sha256(grid_path), "generator_sha256": manifest["generator_sha256"],
                      "fixture_bytes": sum(item["fixture_bytes"] for item in identities)}))


if __name__ == "__main__":
    main()
