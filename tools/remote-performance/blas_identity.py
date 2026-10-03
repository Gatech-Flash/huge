"""Identify the actual BLAS symbol used by the loaded native extension on Linux."""
import ctypes
import hashlib
import os
from pathlib import Path
import sys


class DlInfo(ctypes.Structure):
    _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]


def identity(extension):
    assert sys.platform.startswith('linux'), 'This experiment requires actual Linux'
    loaded = ctypes.CDLL(str(extension))
    address = ctypes.cast(loaded.dgemm_, ctypes.c_void_p)
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    assert dladdr(address, ctypes.byref(info)) != 0
    path = Path(info.filename.decode()).resolve()
    assert 'openblas' in str(path).lower(), str(path)
    blas = ctypes.CDLL(str(path))
    blas.openblas_get_config.restype = ctypes.c_char_p
    config = blas.openblas_get_config().decode()
    assert 'USE64BITINT' not in config, config
    assert blas.openblas_get_num_threads() == 1
    probe = ctypes.CDLL(os.environ['HUGE_AUDIT_TEAM_PROBE'])
    team_size = probe.audit_team_size()
    assert team_size == int(os.environ['OMP_NUM_THREADS']), team_size
    omp = DlInfo()
    assert dladdr(ctypes.cast(loaded.omp_get_max_threads, ctypes.c_void_p), ctypes.byref(omp))
    probe_omp = DlInfo()
    assert dladdr(ctypes.cast(probe.omp_get_max_threads, ctypes.c_void_p), ctypes.byref(probe_omp))
    assert Path(omp.filename.decode()).resolve() == Path(probe_omp.filename.decode()).resolve()
    return {'dgemm_library': str(path),
            'scope': 'Actual BLAS provider of the native extension; NumPy/SciPy may load separate bundled providers',
            'loaded_openblas_library_paths': sorted({line.split()[-1]
                for line in Path('/proc/self/maps').read_text().splitlines()
                if 'openblas' in line.lower() and line.split()[-1].startswith('/')}),
            'dgemm_library_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'configuration': config, 'blas_threads': 1, 'integer_interface': 'LP64',
            'openmp_library': str(Path(omp.filename.decode()).resolve()),
            'separate_runtime_probe_actual_team': team_size,
            'probe_is_not_solver_team_instrumentation': True}
