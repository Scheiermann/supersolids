#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Functions for Potential and initial wave function :math:`\psi_0`

"""

from typing import Optional, List


class Box:
    """
    Specifies the ranges in which the simulation is calculated (1D, 2D or 3D).
    Needs to be given in pairs (x0, x1), (y0, y1), (z0, z1).

    """

    def __init__(self,
                 x0: float, x1: float,
                 y0: Optional[float] = None, y1: Optional[float] = None,
                 z0: Optional[float] = None, z1: Optional[float] = None):
        dim = 1
        if (y0 is None) or (y1 is None):
            assert (y0 is None) or (
                        y1 is None) is None, "y0 and y1 needs to be given in combination."
        else:
            dim = dim + 1
        if (z0 is None) or (z1 is None):
            assert (z0 is None) or (
                        z1 is None) is None, "z0 and z1 needs to be given in combination."
        else:
            dim = dim + 1

        self.dim = dim
        self.x0: float = x0
        self.x1: float = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def __str__(self) -> List[Optional[float]]:
        return str([self.x0, self.x1, self.y0, self.y1, self.z0, self.z1])

    def lengths(self) -> List[float]:
        """
        Calculates the box lengths in the directions available in order [x, y, z]

        :return: List of the box length in the directions available in order [x, y, z]
        """
        if (self.y0 is None) and (self.z0 is None):
            box_lengths = [(self.x1 - self.x0)]
        elif self.z0 is None:
            box_lengths = [(self.x1 - self.x0), (self.y1 - self.y0)]
        else:
            box_lengths = [(self.x1 - self.x0),
                           (self.y1 - self.y0),
                           (self.z1 - self.z0)]

        return box_lengths

    def min_length(self):
        return min(self.lengths())


def BoxResAssert(Res, Box):
    assert len(Res) <= 3, "Dimension of Res needs to be smaller than 3."
    assert len(Box) <= 6, ("Dimension of Box needs to be smaller than 6, "
                           "as the maximum dimension of the problem is 3.")
    assert len(Box) == 2 * len(Res), (
        f"Dimension of Box is {len(Box)}, but needs to be 2 times higher than of Res, "
        f"which currently is {len(Res)}.")
