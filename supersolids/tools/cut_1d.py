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
from pathlib import Path

import numpy as np
from typing import Callable, Tuple, Optional
from matplotlib import pyplot as plt

from supersolids.Schroedinger import Schroedinger
from supersolids.helper import functions


def cut_1d(System: Schroedinger,
           slice_indices: np.ndarray = [0, 0, 0],
           psi_sol_3d_cut_x: Optional[Callable] = None,
           psi_sol_3d_cut_y: Optional[Callable] = None,
           psi_sol_3d_cut_z: Optional[Callable] = None,
           dir_path: Path = Path(__file__).parent.parent.joinpath("results"),
           y_lim: Tuple[float, float] = (0.0, 1.0)
           ) -> None:
    """
    Creates 1D plots of the probability function of the System :math:`|\psi|^2`
    and if given of the solution.

    :param System: Schrödinger equations for the specified system

    :param slice_indices: Numpy array with indices of grid points
        in the directions x, y, z (in terms of System.x, System.y, System.z)
        to produce a slice/plane in mayavi,
        where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
        Max values is for e.g. System.Res.x - 1.

    :param psi_sol_3d_cut_x: 1D function after cut in x direction.

    :param psi_sol_3d_cut_y: 1D function after cut in y direction.

    :param psi_sol_3d_cut_z: 1D function after cut in z direction.

    :param dir_path: Path where to save 1d cut plots

    :param y_lim: Limit of y for plotting the 1D cut

    """

    # prepare the axis where to cut through
    cut_x = np.linspace(System.Box.x0, System.Box.x1, System.Res.x)
    cut_y = np.linspace(System.Box.y0, System.Box.y1, System.Res.y)
    cut_z = np.linspace(System.Box.z0, System.Box.z1, System.Res.z)

    # prepare probability values of cut
    prob_mitte_x = np.abs(
        System.psi_val[:, slice_indices[1], slice_indices[2]]) ** 2.0
    prob_mitte_y = np.abs(
        System.psi_val[slice_indices[0], :, slice_indices[2]]) ** 2.0
    prob_mitte_z = np.abs(
        System.psi_val[slice_indices[0], slice_indices[1], :]) ** 2.0

    # plot probability cuts
    plt.plot(cut_x, prob_mitte_x, "x-", color="tab:blue", label="x cut")
    plt.plot(cut_y, prob_mitte_y, "x-", color="tab:grey", label="y cut")
    plt.plot(cut_z, prob_mitte_z, "x-", color="tab:orange", label="z cut")

    # plot probability cuts of solution, if given
    if psi_sol_3d_cut_x is not None:
        functools.partial(System.psi_sol, y=slice_indices[1], z=slice_indices[2]),
        plt.plot(cut_x, psi_sol_3d_cut_x(x=cut_x), "x-", color="tab:cyan",
                 label="x cut sol")

    if psi_sol_3d_cut_y is not None:
        functools.partial(System.psi_sol, x=slice_indices[0], z=slice_indices[2]),
        plt.plot(cut_y, psi_sol_3d_cut_y(y=cut_y), "x-", color="tab:green",
                 label="y cut sol")

    if psi_sol_3d_cut_z is not None:
        functools.partial(System.psi_sol, x=slice_indices[0], y=slice_indices[1])
        plt.plot(cut_z, psi_sol_3d_cut_z(z=cut_z), "x-", color="tab:olive",
                 label="z cut sol")

    plt.ylim(y_lim)
    plt.legend()
    plt.grid()
    plt.savefig(Path(dir_path, "1d_cut.png"))
    # plt.show()


def prepare_cuts(func: Callable, N: int, alpha_z: float,
                 e_dd: float, a_s_l_ho_ratio: float) -> Optional[Callable]:
    """
    Helper function to get :math:`R_r` and :math:`R_z` and set it for the given func.

    :param func: Function to take cuts from

    :param N: Number of particles

    :param alpha_z: Ratio between z and x frequencies of the trap :math:`w_{z} / w_{x}`

    :param e_dd: Factor :math:`\epsilon_{dd} = a_{dd} / a_{s}`

    :param a_s_l_ho_ratio: :math:`a_s` in units of :math:`l_{HO}`

    :return: func with fixed :math:`R_r` and :math:`R_z`
        (zeros of :math:`func_{125}`), if no singularity occurs, else None.

    """
    kappa = functions.get_kappa(alpha_z=alpha_z, e_dd=e_dd, x_min=0.1,
                                x_max=5.0, res=1000)
    R_r, R_z = functions.get_R_rz(kappa=kappa, e_dd=e_dd, N=N,
                                  a_s_l_ho_ratio=a_s_l_ho_ratio)
    psi_sol_3d = functools.partial(func, R_r=R_r, R_z=R_z)
    print(f"kappa: {kappa}, R_r: {R_r}, R_z: {R_z}")

    if not (np.isnan(R_r) or np.isnan(R_z)):
        return psi_sol_3d
    else:
        return None
