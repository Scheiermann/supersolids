#!/usr/bin/env python

import os
from sys import version_info
from pkg_resources import get_distribution, parse_version

def get_version(package="supersolids"):
    if version_info >= (3, 8, 0):
        from importlib.metadata import version
        package_version = version(package)
    elif version_info >= (3, 6, 0):
        package_version = get_distribution(package).version
    else:
        package_version = "unknown"

    return package_version

def check_numba_used():
    try:
        numba_version = get_version("numba")
    except ModuleNotFoundError:
        numba_version = None
    if numba_version:
        numpy_version = get_version("numpy")
        # numba needs numpy version under or equal 1.21
        if parse_version(numpy_version) < parse_version("1.24"):
            numba_used = True
            print("numba is usable!")
        else:
            print("WARNING: numba NOT used, as it needs numpy version under or equal 1.24!")
            numba_used = False
    else:
        print("WARNING: numba NOT used, as it is not installed!")
        numba_used = False

    return numba_used
    
def check_cupy_used(np):
    try:
        import cupy as cp
        cupy_used = True
        print("cupy is usable!")
    except ImportError:
        print("ImportError: cupy. numba used instead.")
        cp = np
        cupy_used = False
        cuda_used = False
    except ModuleNotFoundError:
        print("ModuleNotFound: cupy. numba used instead.")
        cp = np
        cupy_used = False
        cuda_used = False
    else:
        try:
            if cp.cuda.is_available():
                cupy_used = True
                cuda_used = True
            else:
                print("No cuda available! numba used instead.") 
                cupy_used = False
                cuda_used = False
                cp = np
        except Exception as e:
            print(f"ERROR: {e}! No cupy and cuda turned off! numba used instead.") 
            cupy_used = False
            cuda_used = False
            cp = np
        
                
    return cp, cupy_used, cuda_used

def check_cp_nb(np, gpu_off: bool = False, gpu_index: int = 1):
    numba_used = check_numba_used()
    cp, cupy_used, cuda_used = check_cupy_used(np)
    
    # use flag to turn off gpu even though it might be usable
    if gpu_off:
        cp = np
        cupy_used = False
        cuda_used = False

    if cupy_used:
        numba_used = False
        
        print(f"GPU index: {gpu_index}")
        cp.cuda.runtime.setDevice(gpu_index)

    return cp, cupy_used, cuda_used, numba_used

def get_env_variables(gpu_index_str):
    # if env variable found, it will be a string "False" or "True": trick to convert to bool
    __GPU_OFF_ENV__ = bool(os.environ.get("SUPERSOLIDS_GPU_OFF", False) in ["True", "true"])
    gpu_index_str = int(os.environ.get("SUPERSOLIDS_GPU_INDEX",0))
    __GPU_INDEX__ = int("0" if gpu_index_str=="" else gpu_index_str)

    return __GPU_OFF_ENV__, __GPU_INDEX_ENV__
