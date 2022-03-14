#!/usr/bin/env python

from sys import version_info


def get_version(package="supersolids"):
    if version_info >= (3, 8, 0):
        from importlib.metadata import version
        package_version = version(package)
    elif version_info >= (3, 6, 0):
        import pkg_resources
        package_version = pkg_resources.get_distribution(package).version
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
        if numpy_version <= "1.21":
            numba_used = True
            print("numba is used!")
        else:
            print("WARNING: numba NOT used, as it needs numpy version under or equal 1.21!")
            numba_used = False
    else:
        print("WARNING: numba NOT used, as it is not installed!")
        numba_used = False

    return numba_used
