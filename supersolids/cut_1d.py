#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation.

"""
import functools

import numpy as np
from typing import Callable, Tuple, Optional
from matplotlib import pyplot as plt

from supersolids.Schroedinger import Schroedinger
from supersolids import functions


def cut_1d(System: Schroedinger,
           psi_sol_3d_cut_x: Optional[Callable] = None,
           psi_sol_3d_cut_y: Optional[Callable] = None,
           psi_sol_3d_cut_z: Optional[Callable] = None,
           y_lim: Tuple[float, float] = (0.0, 1.0)
           ) -> None:
    """
    Creates 1D plots of the probability function of the System :math: `|\psi|^2
    and if given of the solution.

    Parameters

    System : Schroedinger.Schroedinger
        Schrödinger equations for the specified system


    Returns

    """

    cut_x = np.linspace(System.Box.x0, System.Box.x1, System.Res.x)
    cut_y = np.linspace(System.Box.y0, System.Box.y1, System.Res.y)
    cut_z = np.linspace(System.Box.z0, System.Box.z1, System.Res.z)

    prob_mitte_x = np.abs(
        System.psi_val[:, System.Res.y // 2, System.Res.z // 2]) ** 2.0
    prob_mitte_y = np.abs(
        System.psi_val[System.Res.x // 2, :, System.Res.z // 2]) ** 2.0
    prob_mitte_z = np.abs(
        System.psi_val[System.Res.x // 2, System.Res.y // 2, :]) ** 2.0

    plt.plot(cut_x, prob_mitte_x, "x-", color="tab:blue", label="x cut")
    plt.plot(cut_y, prob_mitte_y, "x-", color="tab:grey", label="y cut")
    plt.plot(cut_z, prob_mitte_z, "x-", color="tab:orange", label="z cut")

    if psi_sol_3d_cut_x is not None:
        plt.plot(cut_x, psi_sol_3d_cut_x(x=cut_x), "x-", color="tab:cyan",
                 label="x cut sol")

    if psi_sol_3d_cut_y is not None:
        plt.plot(cut_y, psi_sol_3d_cut_y(y=cut_y), "x-", color="tab:green",
                 label="y cut sol")

    if psi_sol_3d_cut_z is not None:
        plt.plot(cut_z, psi_sol_3d_cut_z(z=cut_z), "x-", color="tab:olive",
                 label="z cut sol")

    plt.ylim(y_lim)
    plt.legend()
    plt.grid()
    plt.show()


def prepare_cuts(func: Callable, N: int, alpha_z: float,
                 e_dd: float, a_s_l_ho_ratio: float) -> Optional[Callable]:
    """
    Helper function to get R_r and R_z and set it for the given func.

    Parameters

    N : int
        Number of particles

    alpha_z : float
        Ratio between z and x frequencies of the tap :math:`w_{z} / w_{x}`

    e_dd : float
        :math:`\epsilon_{dd} = a_{dd} / a_{s}`

    a_s_l_ho_ratio : float
        :math:`a_s` in units of :math:`l_{HO}`

    Returns

    psi_sol_3d: Optional[Callable]
        If no singularity occurs, func with fixed R_r and R_z,
        (solution of func_125).


    """
    kappa = functions.get_kappa(alpha_z=alpha_z, e_dd=e_dd, x_min=0.1,
                                x_max=5.0, res=1000)
    R_r, R_z = functions.get_R_rz(kappa=kappa, e_dd=e_dd, N=N,
                                  a_s_l_ho_ratio=a_s_l_ho_ratio)
    psi_sol_3d = functools.partial(func, R_r=R_r, R_z=R_z)
    print(f"kappa: {kappa}, R_r: {R_r}, R_z: {R_z}")

    if not (np.isnan(R_r) or np.isnan(R_z)):
        print(f"")
        return psi_sol_3d
    else:
        return None
