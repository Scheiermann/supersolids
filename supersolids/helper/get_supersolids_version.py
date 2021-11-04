#!/usr/bin/env python

from sys import version_info


def get_supersolids_version():
    if version_info >= (3, 8, 0):
        from importlib.metadata import version
        supersolids_version = version('supersolids')
    elif version_info >= (3, 6, 0):
        import pkg_resources
        supersolids_version = pkg_resources.get_distribution("supersolids").version
    else:
        supersolids_version = "unknown"

    return supersolids_version
