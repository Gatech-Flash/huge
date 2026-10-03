"""Build pinned before/after sources, then run comparisons on an actual Linux runner."""
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

BASE = 'a2452e490934aae2212327faa2c013adc3c6383f'
HERE = Path(__file__).resolve().parent
REPOSITORY = Path(os.environ['GITHUB_WORKSPACE']).resolve()
WORK = Path(os.environ['RUNNER_TEMP']) / 'huge-performance-audit'
WORK.mkdir(exist_ok=False)
RESULTS = WORK / 'results'
RESULTS.mkdir()
assert sys.platform.startswith('linux')
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
           MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           CC='gcc', CXX='g++')
env.pop('PYHUGE_NO_OPENMP', None)
env['HUGE_AUDIT_TEAM_PROBE'] = str(WORK / 'omp-team.so')


def run(command, log, cwd=WORK, runtime=env, timeout=900):
    record = {'command': command, 'cwd': str(cwd), 'timeout_seconds': timeout,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    timed_out = False
    with (RESULTS / log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=runtime, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record['owned_process_group'] = process.pid
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                returncode = None
            finally:
                # The leader may exit while owned descendants ignore TERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if returncode is None:
                returncode = process.wait(timeout=5)
    record.update(actual_returncode=returncode, timed_out=timed_out,
                  elapsed_seconds=time.monotonic() - started,
                  ended_utc=datetime.now(timezone.utc).isoformat())
    (RESULTS / (log + '.command.json')).write_text(json.dumps(record, indent=2) + '\n')
    if returncode or timed_out:
        raise RuntimeError(f'{log}: actual exit {returncode}, timeout={timed_out}')


run(['g++', '-fPIC', '-shared', '-fopenmp', str(HERE / 'omp-team.cpp'),
     '-o', env['HUGE_AUDIT_TEAM_PROBE']], 'build-team-probe.log')
environment = {'python': sys.version, 'platform': sys.platform,
               'checkout_commit': os.environ['GITHUB_SHA'],
               'workflow_run_id': os.environ['GITHUB_RUN_ID'],
               'workflow_run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
               'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'thread_environment': {k: env[k] for k in
                   ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'OMP_DYNAMIC')}}
for name in ('/sys/fs/cgroup/cpu.max', '/proc/cpuinfo'):
    path = Path(name)
    if path.is_file():
        environment[name] = path.read_text()
(RESULTS / 'runner-environment.json').write_text(json.dumps(environment, indent=2) + '\n')
run(['gcc', '--version'], 'compiler-version.log')
run([sys.executable, '-c', 'import numpy, scipy; numpy.show_config(); scipy.show_config()'],
    'numpy-scipy-configuration.log')
run(['git', 'fetch', 'origin', BASE, '--depth=1'], 'fetch-baseline.log', REPOSITORY)
archive = WORK / 'upstream.tar'
run(['git', 'archive', '--output=' + str(archive), BASE], 'archive-baseline.log', REPOSITORY)
upstream = WORK / 'upstream'
upstream.mkdir()
with tarfile.open(archive) as opened:
    opened.extractall(upstream, filter='data')
packages = {}
pins = {}
for arm, source in (('baseline', upstream / 'python-package'),
                    ('candidate', REPOSITORY / 'python-package')):
    package = WORK / arm
    shutil.copytree(source, package, ignore=shutil.ignore_patterns(
        'build', 'dist', 'site', '.venv', '.pytest_cache', '.mplconfig',
        '__pycache__', '*.egg-info', '*.so', '*.dylib', '*.pyc'))
    packages[arm] = package
    pins[arm] = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in package.rglob('*') if p.is_file()}
    run([sys.executable, 'setup.py', 'build_ext', '--inplace', '--force'],
        f'build-{arm}.log', package)
    native = list((package / 'pyhuge').glob('_native_core*.so'))
    assert len(native) == 1, native
    binary_dir = RESULTS / 'binaries' / arm
    binary_dir.mkdir(parents=True)
    shutil.copyfile(native[0], binary_dir / native[0].name)
    run(['ldd', str(native[0])], f'linkage-{arm}.log')
(RESULTS / 'source-pins.json').write_text(json.dumps(pins, indent=2) + '\n')
run([sys.executable, str(HERE / 'generate-python-specs.py'), '--out', str(WORK)],
    'generate-grid.log')
run([sys.executable, str(HERE / 'extend-grid.py'), str(WORK)], 'extend-grid.log')
for name in ('python-grid.json', 'python-fixture-manifest.json', 'original-24-fixture-manifest.json'):
    shutil.copyfile(WORK / name, RESULTS / name)
shutil.copytree(WORK / 'python-fixtures', RESULTS / 'fixtures')
shutil.copytree(WORK / 'python-specs', RESULTS / 'specs')
utility_dir = RESULTS / 'audit-utilities'
shutil.copytree(HERE, utility_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
run(['ldd', env['HUGE_AUDIT_TEAM_PROBE']], 'linkage-team-probe.log')
shutil.copyfile(env['HUGE_AUDIT_TEAM_PROBE'], RESULTS / 'binaries' / 'omp-team.so')
for threads in (1, 4):
    runtime = env.copy()
    runtime['HUGE_AUDIT_OMP_THREADS'] = str(threads)
    run([sys.executable, str(HERE / 'python-compare.py'), 'selftest'],
        f'comparator-controls-omp{threads}.log', runtime=runtime)
    run([sys.executable, str(HERE / 'python-compare.py'), 'run',
         '--grid', str(WORK / 'python-grid.json'),
         '--baseline', str(packages['baseline']), '--candidate', str(packages['candidate']),
         '--output', str(RESULTS / f'omp{threads}'), '--trials', '5',
         '--target-seconds', '0.25', '--timeout', '240'],
        f'comparison-omp{threads}.log', runtime=runtime, timeout=10800)
    summary = json.loads((RESULTS / f'omp{threads}' / 'analysis-summary.json').read_text())
    assert all(case['eligible_pairs'] == 5 for case in summary['cases']), summary
(RESULTS / 'completed.json').write_text(json.dumps({
    'completed': True, 'platform': 'actual Linux', 'baseline_commit': BASE,
    'thread_counts': [1, 4], 'BLAS': 'native extension LP64 OpenBLAS verified per worker; separate NumPy/SciPy providers recorded',
    'performance_scope': 'Paired public APIs/workflows; not universal acceleration'}, indent=2) + '\n')
print(json.dumps({'completed': True, 'results': str(RESULTS)}))
