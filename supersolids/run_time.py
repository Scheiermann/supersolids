#!/usr/bin/env python


# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Helper-Functions/decorators to compute parallel and profile functions

"""

import contextlib
import time

from typing import Iterator


@contextlib.contextmanager
def run_time(name: str = "method") -> Iterator:
    start: float = time.perf_counter()
    try:
        yield
    finally:
        time_measured: float = time.perf_counter() - start
        print(f"Runtime {name}: {time_measured:.8f} s")

    return time_measured
