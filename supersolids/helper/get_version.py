#!/usr/bin/env python

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

def check_cp_nb(np, gpu_off = False):
    numba_used = check_numba_used()
    cp, cupy_used, cuda_used = check_cupy_used(np)
    
    # use flag to turn off gpu even though it might be usable
    if gpu_off:
        cp = np
        cupy_used = False
        cuda_used = False

    if cupy_used:
        numba_used = False

    return cp, cupy_used, cuda_used, numba_used

