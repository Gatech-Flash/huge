#!/usr/bin/env python3
"""Benchmark worker for pyhuge. Runs all method×graph×size combos and saves results as JSON.

Usage:
    python bench_pyhuge_worker.py <version_label> <output_json> [pypi_lib_path]

If pypi_lib_path is given, it is prepended to sys.path to load that version.
"""

import json
import sys
import time

import numpy as np

if len(sys.argv) < 3:
    print("Usage: bench_pyhuge_worker.py <label> <output.json> [pypi_lib_path]")
    sys.exit(1)

label = sys.argv[1]
output_path = sys.argv[2]

if len(sys.argv) >= 4:
    # Load from specified path (for PyPI version)
    pypi_path = sys.argv[3]
    sys.path.insert(0, pypi_path)

import pyhuge

print(f"[{label}] pyhuge version: {pyhuge.__version__}")

# ---------- config ----------
GRAPHS = ["hub", "cluster", "band", "random", "scale-free"]
METHODS = ["glasso", "mb", "ct", "tiger"]
SIZES = [
    {"label": "small", "n": 200, "d": 50},
    {"label": "medium", "n": 400, "d": 200},
]
NREPS = 5
SEED = 42


def f1_score(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def eval_accuracy(adj_true, adj_est):
    """Compute TPR, FPR, F1 from adjacency matrices (upper triangle)."""
    mask = np.triu(np.ones_like(adj_true, dtype=bool), k=1)
    t = adj_true[mask].astype(bool)
    e = adj_est[mask].astype(bool)
    tp = np.sum(t & e)
    fp = np.sum(~t & e)
    fn = np.sum(t & ~e)
    tn = np.sum(~t & ~e)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(tp, fp, fn)
    return {"tpr": float(tpr), "fpr": float(fpr), "f1": float(f1)}


def frobenius_rel(theta_true, theta_est):
    """Relative Frobenius error."""
    return float(np.linalg.norm(theta_est - theta_true, "fro") / np.linalg.norm(theta_true, "fro"))


results = []

for sz in SIZES:
    for graph in GRAPHS:
        print(f"\n[{label}] === {sz['label']} d={sz['d']} graph={graph} ===")

        # Generate data with fixed seed
        gen = pyhuge.huge_generator(
            n=sz["n"], d=sz["d"], graph=graph, random_state=SEED, verbose=False
        )

        # Handle both dense and sparse returns
        theta_true = gen.theta
        if hasattr(theta_true, "toarray"):
            theta_true = theta_true.toarray()
        theta_true = np.array(theta_true, dtype=float)

        adj_true = (np.abs(theta_true) > 1e-10).astype(float)
        np.fill_diagonal(adj_true, 0)

        for method in METHODS:
            print(f"  [{label}] {method} ... ", end="", flush=True)

            fn_map = {
                "glasso": pyhuge.huge_glasso,
                "mb": pyhuge.huge_mb,
                "ct": pyhuge.huge_ct,
                "tiger": pyhuge.huge_tiger,
            }
            fn = fn_map[method]

            # Warmup + time over NREPS
            times = []
            est = None
            try:
                fn(gen.data, verbose=False)  # warmup
                for rep in range(NREPS):
                    t0 = time.perf_counter()
                    est = fn(gen.data, verbose=False)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "size": sz["label"], "d": sz["d"], "graph": graph,
                    "method": method, "error": str(e),
                })
                continue

            time_mean = float(np.mean(times))

            # Pick middle lambda for accuracy evaluation
            n_lambda = len(est.path)
            mid_idx = n_lambda // 2
            adj_mid = est.path[mid_idx]
            if hasattr(adj_mid, "toarray"):
                adj_mid = adj_mid.toarray()
            adj_est = np.array(adj_mid).astype(float)

            acc = eval_accuracy(adj_true, adj_est)

            # Frobenius error for glasso and tiger (they produce icov)
            frob = None
            if method in ("glasso", "tiger") and est.icov is not None and len(est.icov) > mid_idx:
                try:
                    icov_mid = est.icov[mid_idx]
                    if hasattr(icov_mid, "toarray"):
                        icov_mid = icov_mid.toarray()
                    icov_est = np.array(icov_mid)
                    frob = frobenius_rel(theta_true, icov_est)
                except Exception:
                    pass

            row = {
                "size": sz["label"], "d": sz["d"], "graph": graph,
                "method": method, "time_mean": time_mean,
                "f1": acc["f1"], "tpr": acc["tpr"], "fpr": acc["fpr"],
            }
            if frob is not None:
                row["frob_rel"] = frob

            results.append(row)
            print(f"{time_mean:.4f}s  F1={acc['f1']:.4f}")

            # Also benchmark with stars model selection for glasso and mb
            if method in ("glasso", "mb"):
                print(f"  [{label}] {method}+stars ... ", end="", flush=True)
                try:
                    sel_times = []
                    sel_res = None
                    # Warmup
                    est_w = fn(gen.data, verbose=False)
                    pyhuge.huge_select(est_w, criterion="stars", verbose=False)
                    for rep in range(NREPS):
                        est2 = fn(gen.data, verbose=False)
                        t0 = time.perf_counter()
                        sel_res = pyhuge.huge_select(est2, criterion="stars", verbose=False)
                        t1 = time.perf_counter()
                        sel_times.append(t1 - t0)

                    sel_time_mean = float(np.mean(sel_times))

                    # Evaluate selected model
                    sel_raw = sel_res.refit if sel_res.refit is not None else sel_res.opt_path
                    if hasattr(sel_raw, "toarray"):
                        sel_raw = sel_raw.toarray()
                    sel_adj = np.array(sel_raw).astype(float)
                    sel_acc = eval_accuracy(adj_true, sel_adj)

                    results.append({
                        "size": sz["label"], "d": sz["d"], "graph": graph,
                        "method": f"{method}+stars",
                        "time_mean": sel_time_mean,
                        "f1": sel_acc["f1"], "tpr": sel_acc["tpr"], "fpr": sel_acc["fpr"],
                    })
                    print(f"{sel_time_mean:.4f}s  F1={sel_acc['f1']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[{label}] Results saved to {output_path}")
