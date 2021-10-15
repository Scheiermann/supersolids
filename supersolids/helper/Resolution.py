#!/usr/bin/env python
import sys
from typing import Optional, List

import numpy as np


class Resolution:
    """
    Specifies the resolution of the simulation in x, y, z directions (1D, 2D, 3D).

    """
    def __init__(self,
                 x: float,
                 y: Optional[float] = None,
                 z: Optional[float] = None):
        dim = 1
        if y is not None:
            dim = dim + 1
        if z is not None:
            dim = dim + 1

        self.dim = dim
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> List[Optional[float]]:
        return str([self.x, self.y, self.z])

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def get_bounds_by_index(self, index):
        res_arr = self.to_array()
        if 0 <= index <= len(res_arr):
            return res_arr[index]
        else:
            sys.exit(f"Res index is not possible: {index}")


def ResAssert(Res, a, name="Amplitudes"):
    assert len(a) == len(Res), (
    f"Dimension of {name} is {len(a)}, but needs to be the same as dimension of Res, "
    f"which currently is {len(Res)}.")
