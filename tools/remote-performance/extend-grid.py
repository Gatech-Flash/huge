"""Add native high-dimensional workflows before running any comparisons."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1]).resolve()
grid_path = root / 'python-grid.json'
grid = json.loads(grid_path.read_text())
cases = grid['cases']
assert len(cases) == 24
manifest_path = root / 'python-fixture-manifest.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['case_count'] == 24
original_manifest = root / 'original-24-fixture-manifest.json'
assert not original_manifest.exists()
original_manifest.write_bytes(manifest_path.read_bytes())


def add(name, x, method, kwargs, selection=None, seed=None):
    fixture = root / 'python-fixtures' / (name + '.npz')
    assert not fixture.exists()
    np.savez_compressed(fixture, x=x)
    spec = {'schema': 1, 'id': name, 'operation': 'fit_select' if selection else 'fit',
            'fixture': str(fixture.relative_to(root)),
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'kwargs': {'method': method, 'verbose': False, **kwargs},
            'fixture_seed': seed, 'structure': 'Explicit frozen supplemental workflow',
            'expected_work': {'d': x.shape[1],
                              'n': x.shape[0] if kwargs.get('input_type') == 'data' else None,
                              'requested_path_length': len(kwargs['lambda_']) if 'lambda_' in kwargs
                                  else kwargs.get('nlambda', 20),
                              'allow_truncation': method == 'tiger'},
            'corner_flags': ['OpenBLAS_OpenMP_supplement'],
            'output_scope': 'complete_public_result', 'correctness_notes': []}
    if selection:
        spec['select_kwargs'] = selection
    cases.append(spec)
    spec_path = root / 'python-specs' / (name + '.json')
    assert not spec_path.exists()
    spec_path.write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
    manifest['fixtures'].append({
        'id': name, 'fixture': spec['fixture'], 'fixture_sha256': spec['fixture_sha256'],
        'fixture_bytes': fixture.stat().st_size,
        'spec': str(spec_path.relative_to(root)),
        'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'arrays': {'x': {'shape': list(x.shape), 'dtype': str(x.dtype),
            'logical_c_sha256': hashlib.sha256(x.tobytes(order='C')).hexdigest()}}})


for d in (511, 512, 513):
    indexes = np.arange(d)
    cov = 0.65 ** np.abs(indexes[:, None] - indexes[None, :])
    add('linux-glasso-ar1-d' + str(d), cov, 'glasso',
        {'input_type': 'covariance', 'lambda_': [0.3, 0.15], 'cov_output': True})
for d, seed in ((512, 401), (1024, 907)):
    x = np.random.default_rng(seed).normal(size=(100, d))
    for col in range(1, d):
        if col % 16:
            x[:, col] = 0.65 * x[:, col - 1] + np.sqrt(1 - 0.65 ** 2) * x[:, col]
    add('linux-mb-stars-d' + str(d), x, 'mb',
        {'input_type': 'data', 'lambda_': [0.35, 0.15, 0.07], 'scr': False},
        {'criterion': 'stars', 'rep_num': 5, 'n_jobs': 1, 'verbose': False}, seed)
    add('linux-ct-ric-d' + str(d), x, 'ct', {'input_type': 'data'},
        {'criterion': 'ric', 'rep_num': 10, 'n_jobs': 1, 'verbose': False}, seed)
add('linux-tiger-long-d512', np.random.default_rng(503).normal(size=(100, 512)),
    'tiger', {'input_type': 'data', 'nlambda': 64}, seed=503)
assert len(cases) == 32
grid['case_count'] = len(cases)
grid['measurement_scope'] = ('32 frozen public API cases: complete fit/fit_select '
    'and individual NPN/ROC/inference APIs; no internal helper timers. '
    'Not every case includes fitting, selection and evaluation.')
grid_path.write_text(json.dumps(grid, indent=2, allow_nan=False) + '\n')
manifest.update(case_count=len(cases), grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
    supplemental_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    original_24_manifest_sha256=hashlib.sha256(original_manifest.read_bytes()).hexdigest())
assert len(manifest['fixtures']) == 32
manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
print(json.dumps({'cases': len(cases), 'grid_sha256': hashlib.sha256(grid_path.read_bytes()).hexdigest()}))
