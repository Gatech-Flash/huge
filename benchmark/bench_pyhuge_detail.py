#!/usr/bin/env python3
"""Detailed micro-benchmark: isolate C++ core time vs Python wrapper overhead.
Run both versions in subprocess for clean isolation."""

import json
import subprocess
import sys

SCRIPT = '''
import sys, time, numpy as np, json

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import pyhuge
from pyhuge import core as _core

label = "PyPI" if len(sys.argv) > 1 else "Local"
ver = pyhuge.__version__

results = []
NREPS = 10

for d_size in [(200, 50), (400, 200)]:
    n, d = d_size
    gen = pyhuge.huge_generator(n=n, d=d, graph="hub", random_state=42, verbose=False)

    for method in ["glasso", "mb", "ct", "tiger"]:
        fn_map = {
            "glasso": pyhuge.huge_glasso,
            "mb": pyhuge.huge_mb,
            "ct": pyhuge.huge_ct,
            "tiger": pyhuge.huge_tiger,
        }
        fn = fn_map[method]

        # Warmup
        fn(gen.data, verbose=False)

        times = []
        for _ in range(NREPS):
            t0 = time.perf_counter()
            fn(gen.data, verbose=False)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        results.append({
            "label": label, "version": ver,
            "n": n, "d": d, "method": method,
            "times": times,
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "median": float(np.median(times)),
            "min": float(np.min(times)),
        })

print(json.dumps(results))
'''

def run_version(label, extra_arg=None):
    cmd = [sys.executable, "-c", SCRIPT]
    if extra_arg:
        cmd.append(extra_arg)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[{label}] STDERR:", result.stderr[-500:])
        return []
    return json.loads(result.stdout.strip())


print("Running PyPI 0.3.3 benchmark (10 reps)...")
pypi_results = run_version("PyPI", "/tmp/pyhuge_pypi_lib")

print("Running Local 0.8.0 benchmark (10 reps)...")
local_results = run_version("Local")

# Build comparison
pypi_idx = {(r["d"], r["method"]): r for r in pypi_results}
local_idx = {(r["d"], r["method"]): r for r in local_results}

print()
print("=" * 95)
print("  DETAILED TIMING COMPARISON: PyPI 0.3.3  vs  LOCAL 0.8.0  (10 reps each)")
print("=" * 95)
print()
print(f"{'d':>5} {'Method':<10} {'PyPI mean':>10} {'PyPI std':>10} {'Local mean':>10} {'Local std':>10} {'Ratio':>8} {'Verdict':>12}")
print("-" * 85)

for key in sorted(pypi_idx.keys()):
    p = pypi_idx[key]
    l = local_idx.get(key)
    if not l:
        continue
    ratio = p["mean"] / l["mean"]

    # Simple significance: if difference > 2*max(std), likely real
    diff = abs(p["mean"] - l["mean"])
    noise = 2 * max(p["std"], l["std"])
    if diff < noise:
        verdict = "NOISE"
    elif ratio > 1.05:
        verdict = "LOCAL faster"
    elif ratio < 0.95:
        verdict = "PyPI faster"
    else:
        verdict = "~SAME"

    print(f"{key[0]:>5} {key[1]:<10} {p['mean']:10.4f} {p['std']:10.4f} {l['mean']:10.4f} {l['std']:10.4f} {ratio:7.2f}x {verdict:>12}")

print()
