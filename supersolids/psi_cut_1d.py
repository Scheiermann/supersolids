#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation.

"""

import numpy as np
from typing import Callable, Tuple
from matplotlib import pyplot as plt

from supersolids import Schroedinger


def psi_cut_1d(System: Schroedinger,
               psi_sol_3d_cut_x: Callable = None,
               psi_sol_3d_cut_y: Callable = None,
               psi_sol_3d_cut_z: Callable = None,
               y_lim: Tuple[float, float] = (0.0, 1.0)
               ) -> None:
    """
    Creates 1D plots of the probability function of the System :math: `|\psi|^2
    and if given of the solution.

    Parameters


    Returns

    """

    cut_x = np.linspace(System.Box.x0, System.Box.x1, System.Res.x)
    cut_y = np.linspace(System.Box.y0, System.Box.y1, System.Res.y)
    cut_z = np.linspace(System.Box.z0, System.Box.z1, System.Res.z)

    prob_mitte_x = np.abs(System.psi_val[:, System.Res.y // 2, System.Res.z // 2]) ** 2.0
    prob_mitte_y = np.abs(System.psi_val[System.Res.x // 2, :, System.Res.z // 2]) ** 2.0
    prob_mitte_z = np.abs(System.psi_val[System.Res.x // 2, System.Res.y // 2, :]) ** 2.0

    plt.plot(cut_x, prob_mitte_x, "x-", color="tab:blue", label="x cut")
    plt.plot(cut_y, prob_mitte_y, "x-", color="tab:grey", label="y cut")
    plt.plot(cut_z, prob_mitte_z, "x-", color="tab:orange", label="z cut")
    plt.plot(cut_x, psi_sol_3d_cut_x(cut_x), "x-", color="tab:cyan",
             label="x cut sol")
    plt.plot(cut_y, psi_sol_3d_cut_y(y=cut_y), "x-", color="tab:green",
             label="y cut sol")
    plt.plot(cut_z, psi_sol_3d_cut_z(z=cut_z), "x-", color="tab:olive",
             label="z cut sol")
    plt.ylim(y_lim)
    plt.legend()
    plt.grid()
    plt.show()
