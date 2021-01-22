#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information. Thanks!

"""
Calculates the density functions for a quantum droplet in a trap.

"""

import numpy as np

from matplotlib import pyplot as plt

from supersolids import constants
from supersolids import functions

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    N: int = 4 * 10 ** 4
    m: float = 164.0 * constants.u_in_kg

    w_x: float = 2.0 * np.pi * 30.0
    w_y: float = w_x
    w_z: float = 2.0 * np.pi * 140.0

    alpha_z: float = w_z / w_x

    a_dd: float = 8.0 * constants.a_0
    a_s: float = 10.0 * constants.a_0

    g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
        N=N, m=m, a_s=a_s, a_dd=a_dd, w_x=w_x)
    print(f"g, g_qf, epsilon_dd: {g, g_qf, e_dd}")

    kappa: np.ndarray = np.linspace(0.0, 5.0, 1000)

    y = functions.func_125(kappa, alpha_z, e_dd)
    plt.plot(kappa, y, "x-")
    plt.xlim([0.0, 5.0])
    plt.ylim([-10.0, 20.0])
    plt.grid()
    plt.show()
    kappa_root = min(kappa[y >= 0.0]) if y[-1] > 0 else min(kappa[y <= 0.0])
    # kappa_root: float = 2.6773665398494018e-05
    # kappa_root: float = 4.079079079079079
    print(f"kappa_root: {kappa_root}")
    R_r = functions.func_124(kappa_root, e_dd, N, a_s_l_ho_ratio)
    R_z = R_r / kappa_root
    print(f"R_r: {R_r}")
    print(f"R_z: {R_z}")

    r = R_r * np.linspace(-2.0, 2.0, 1000)
    n_r = functions.density_in_trap_r(r, z=0.0, R_r=R_r, R_z=R_z)
    # plt.plot(r, n_r, "x-", label="x")
    # plt.ylim([-10.0, 10.0])
    # plt.grid()
    # plt.show()
