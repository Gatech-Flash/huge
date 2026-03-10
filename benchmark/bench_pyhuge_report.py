#!/usr/bin/env python3
"""Merge two pyhuge benchmark JSON outputs and print comparison report."""

import json
import sys

with open("benchmark/bench_pyhuge_pypi.json") as f:
    pypi_res = json.load(f)
with open("benchmark/bench_pyhuge_local.json") as f:
    local_res = json.load(f)

# Index by (size, graph, method)
def index_results(rows):
    d = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["size"], r["graph"], r["method"])
        d[key] = r
    return d

pypi = index_results(pypi_res)
local = index_results(local_res)

all_keys = sorted(set(pypi.keys()) | set(local.keys()))

print()
print("=" * 100)
print("         COMPARISON: PyPI pyhuge 0.3.3  vs  LOCAL pyhuge 0.8.0")
print("=" * 100)

# Timing & F1
print("\n--- Timing & F1 Score ---\n")
print(f"{'Size':<8} {'Graph':<12} {'Method':<14} {'T(PyPI)':>9} {'T(0.8)':>9} {'Speedup':>8} {'F1.PyPI':>8} {'F1.0.8':>8} {'F1.delta':>9}")
print("-" * 100)

speedups_est = []
speedups_sel = []
f1_deltas_est = []
f1_deltas_sel = []
frob_rows = []

for key in all_keys:
    p = pypi.get(key)
    l = local.get(key)
    if not p or not l:
        continue

    speedup = p["time_mean"] / l["time_mean"] if l["time_mean"] > 0 else float("inf")
    f1_delta = l["f1"] - p["f1"]

    print(f"{key[0]:<8} {key[1]:<12} {key[2]:<14} {p['time_mean']:9.4f} {l['time_mean']:9.4f} {speedup:7.2f}x {p['f1']:8.4f} {l['f1']:8.4f} {f1_delta:+9.4f}")

    if "+" in key[2]:
        speedups_sel.append(speedup)
        f1_deltas_sel.append(f1_delta)
    else:
        speedups_est.append(speedup)
        f1_deltas_est.append(f1_delta)

    if "frob_rel" in p and "frob_rel" in l:
        frob_rows.append((key, p["frob_rel"], l["frob_rel"]))

# Frobenius
if frob_rows:
    print("\n--- Precision Matrix Relative Frobenius Error (glasso & tiger) ---\n")
    print(f"{'Size':<8} {'Graph':<12} {'Method':<14} {'Frob.PyPI':>11} {'Frob.0.8':>11} {'Delta':>11}")
    print("-" * 71)
    for key, fp, fl in frob_rows:
        print(f"{key[0]:<8} {key[1]:<12} {key[2]:<14} {fp:11.4f} {fl:11.4f} {fl - fp:+11.4f}")

# TPR/FPR
print("\n--- TPR / FPR Detail ---\n")
print(f"{'Size':<8} {'Graph':<12} {'Method':<14} {'TPR.PI':>7} {'TPR.08':>7} {'FPR.PI':>7} {'FPR.08':>7}")
print("-" * 70)
for key in all_keys:
    p = pypi.get(key)
    l = local.get(key)
    if not p or not l:
        continue
    print(f"{key[0]:<8} {key[1]:<12} {key[2]:<14} {p['tpr']:7.4f} {l['tpr']:7.4f} {p['fpr']:7.4f} {l['fpr']:7.4f}")

# Summary
print("\n--- Summary ---\n")
if speedups_est:
    import statistics
    print(f"  Graph estimation ({len(speedups_est)} configs):")
    print(f"    Mean speedup :  {statistics.mean(speedups_est):.2f}x")
    print(f"    Median speedup: {statistics.median(speedups_est):.2f}x")
    print(f"    Mean F1 delta:  {statistics.mean(f1_deltas_est):+.4f}")

if speedups_sel:
    print(f"\n  Model selection ({len(speedups_sel)} configs):")
    print(f"    Mean speedup:   {statistics.mean(speedups_sel):.2f}x")
    print(f"    Mean F1 delta:  {statistics.mean(f1_deltas_sel):+.4f}")

all_f1_deltas = f1_deltas_est + f1_deltas_sel
print("\n  Accuracy equivalence:")
if all(abs(d) < 0.02 for d in all_f1_deltas):
    print("    All F1 deltas within +/-0.02 => EQUIVALENT")
else:
    print("    Some F1 deltas exceed 0.02 (see table above)")

print()
