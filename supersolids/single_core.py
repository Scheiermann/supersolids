#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D.

"""

import functools
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from supersolids.Animation.Animation import Animation
from supersolids.Schroedinger import Schroedinger
from supersolids.tools.simulate_case import simulate_case
from supersolids.tools.cut_1d import prepare_cuts
from supersolids.helper import constants, functions
from supersolids.helper.Resolution import Resolution
from supersolids.helper.Box import Box

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # Define constants (needed for the Schroedinger equation)
    # this is done at this point, to be able to use those
    # constants for the definitions of V, psi_0, psi_sol

    # due to fft of the points the res
    # needs to be 2 ** resolution_exponent
    Res: Resolution = Resolution(x=2 ** 8, y=2 ** 7, z=2 ** 5)

    MyBox: Box = Box(x0=-10, x1=10, y0=-5, y1=5, z0=-4, z1=4)

    dt: float = 2 * 10 ** -3
    # dt_func = functools.partial(functions.dt_adaptive)

    N: int = 6 * 10 ** 4
    m: float = 164.0 * constants.u_in_kg
    a_dd: float = 130.0 * constants.a_0
    a_s: float = 85.0 * constants.a_0

    w_x: float = 2.0 * np.pi * 33.0
    w_y: float = 2.0 * np.pi * 80.0
    w_z: float = 2.0 * np.pi * 167.0

    alpha_y, alpha_z = functions.get_alphas(w_x=w_x, w_y=w_y, w_z=w_z)
    g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
        N=N, m=m, a_s=a_s, a_dd=a_dd, w_x=w_x)
    print(f"g, g_qf, e_dd, alpha_y, alpha_z: "
          f"{g, g_qf, e_dd, alpha_y, alpha_z}")

    # Define functions (needed for the Schroedinger equation)
    # (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=alpha_y)
    V_3d = functools.partial(functions.v_harmonic_3d,
                             alpha_y=alpha_y,
                             alpha_z=alpha_z)

    V_3d_ddi = functools.partial(functions.dipol_dipol_interaction,
                                 r_cut=1.0 * MyBox.min_length() / 2.0)

    # functools.partial sets all arguments except x, y, z,
    # psi_0_1d = functools.partial(
    #     functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    psi_0_1d = functools.partial(
        functions.psi_gauss_1d, a=3.0, x_0=0.0, k_0=0.0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf,
                                 mu=[0.0, 0.0],
                                 var=np.array([[1.0, 0.0], [0.0, 1.0]])
                                 )

    psi_0_3d = functools.partial(
        functions.psi_gauss_3d,
        a_x=3.5, a_y=1.5, a_z=1.2,
        x_0=0.0, y_0=0.0, z_0=0.0,
        k_0=0.0)
    # psi_0_3d = functools.partial(functions.prob_in_trap, R_r=R_r, R_z=R_z)

    psi_0_noise_3d = functions.noise_mesh(
        min=0.8, max=1.4, shape=(Res.x, Res.y, Res.z))

    # Used to remember that 2D need the special pos function (g is set inside
    # of Schroedinger for convenience)
    psi_sol_1d = functions.thomas_fermi_1d
    psi_sol_2d = functions.thomas_fermi_2d_pos

    # psi_sol_3d = functions.thomas_fermi_3d
    if MyBox.dim == 3:
        psi_sol_3d: Optional[Callable] = prepare_cuts(functions.density_in_trap,
                                                      N, alpha_z, e_dd,
                                                      a_s_l_ho_ratio)
    else:
        psi_sol_3d = None

    System: Schroedinger = Schroedinger(N,
                                        MyBox,
                                        Res,
                                        max_timesteps=8001,
                                        dt=dt,
                                        dt_func=None,
                                        g=g,
                                        g_qf=g_qf,
                                        w_y=w_y,
                                        w_z=w_z,
                                        e_dd=e_dd,
                                        a_s=a_s,
                                        imag_time=True,
                                        mu=1.1,
                                        E=1.0,
                                        psi_0=psi_0_3d,
                                        V=V_3d,
                                        V_interaction=V_3d_ddi,
                                        psi_sol=psi_sol_3d,
                                        mu_sol=functions.mu_3d,
                                        psi_0_noise=None,
                                        )

    Anim: Animation = Animation(Res=System.Res,
                                plot_psi_sol=False,
                                plot_V=False,
                                alpha_psi=0.8,
                                alpha_psi_sol=0.5,
                                alpha_V=0.3,
                                camera_r_func=functools.partial(
                                    functions.camera_func_r,
                                    r_0=40.0, phi_0=45.0, z_0=50.0,
                                    r_per_frame=0.0),
                                camera_phi_func=functools.partial(
                                    functions.camera_func_phi,
                                    r_0=40.0, phi_0=45.0, z_0=50.0,
                                    phi_per_frame=5.0),
                                camera_z_func=functools.partial(
                                    functions.camera_func_z,
                                    r_0=40.0, phi_0=45.0, z_0=50.0,
                                    z_per_frame=0.0),
                                filename="anim.mp4",
                                )

    if MyBox.dim == 3:
        slice_indices = [int(Res.x / 2), int(Res.y / 2), int(Res.z / 2)]
    else:
        slice_indices = [None, None, None]

    # TODO: get mayavi lim to work
    # 3D works in single core mode
    SystemResult: Schroedinger = simulate_case(
                                    System=System,
                                    Anim=Anim,
                                    accuracy=10 ** -12,
                                    delete_input=False,
                                    dir_path=Path(__file__).parent.joinpath("results"),
                                    slice_indices=slice_indices, # from here just mayavi
                                    offscreen=True,
                                    x_lim=(-2.0, 2.0), # from here just matplotlib
                                    y_lim=(-2.0, 2.0),
                                    z_lim=(0, 0.5),
                                    )

    print("Single core done")

