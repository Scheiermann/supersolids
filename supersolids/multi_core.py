#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
DEPRECATED

Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D and 2D in multi-core.

"""

import itertools
import functools
from concurrent import futures
import psutil

import numpy as np

from supersolids.helper import constants
from supersolids.helper import functions
from supersolids.helper import simulate_case
from supersolids.helper.Resolution import Resolution
from supersolids.helper.Box import Box

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # constants needed for the Schroedinger equation

    # due to fft of the points the res
    # needs to be 2 ** resolution_exponent
    Res = Resolution(x=2 ** 6, y=2 ** 6, z=None)

    MyBox = Box(x0=-15, x1=15, y0=-15, y1=15, z0=None, z1=None)

    dt: float = 2 * 10 ** -2 # 0.001
    N: int = 3.8 * 10 ** 4 # 38000
    m: float = 164.0 * constants.u_in_kg
    a_dd: float = 130.0 * constants.a_0
    a_s: float = 85.0 * constants.a_0

    w_x: float = 2.0 * np.pi * 30.0
    w_y: float = 2.0 * np.pi * 60.0
    w_z: float = 2.0 * np.pi * 160.0

    alpha_y, alpha_z = functions.get_alphas(w_x=w_x, w_y=w_y, w_z=w_z)
    g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
        N=N, m=m, a_s=a_s, a_dd=a_dd, w_x=w_x)
    print(f"g, g_qf, epsilon_dd, alpha_y, alpha_z: "
          f"{g, g_qf, e_dd, alpha_y, alpha_z}")

    # psi_sol_3d = functions.thomas_fermi_3d
    kappa = functions.get_kappa(alpha_z=alpha_z, e_dd=e_dd,
                                x_min=0.1, x_max=5.0, res=1000)
    R_r, R_z = functions.get_R_rz(kappa=kappa, e_dd=e_dd, N=N,
                                  a_s_l_ho_ratio=a_s_l_ho_ratio)
    psi_sol_3d = functools.partial(functions.density_in_trap,
                                   R_r=R_r, R_z=R_z)
    print(f"kappa: {kappa}, R_r: {R_r}, R_z: {R_z}")

    # functions needed for the Schroedinger equation (e.g. potential: V,
    # initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=alpha_y)

    # functools.partial sets all arguments except x, y, z,
    # as multiple arguments for Schroedinger aren't implement yet
    # psi_0_1d = functools.partial(
    #     functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    psi_0_1d = functools.partial(
        functions.psi_gauss_1d, a=3.0, x_0=0.0, k_0=0.0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf,
                                 mu=[0.0, 0.0],
                                 var=np.array([[1.0, 0.0], [0.0, 1.0]])
                                 )

    # Used to remember that 2D need the special pos function (g is set inside
    # of Schroedinger for convenience)
    psi_sol_1d = functions.thomas_fermi_1d
    psi_sol_2d = functions.thomas_fermi_2d_pos

    # TODO: As g is proportional to N * a_s/w_x,
    # changing g, V, g_qf are different (maybe other variables too)
    # Multi-core: multiple cases (Schroedinger with different parameters)
    # box length in 1D: [-L,L], in 2D: [-L,L, -L,L], in 3D: [-L,L, -L,L, -L,L]
    # generators for L, g, dt to compute for different parameters
    g_step = 10
    L_generator = (4,)
    g_generator = (i for i in np.arange(g, g + g_step, g_step))
    factors = np.linspace(0.2, 0.3, max_workers)
    dt_generator = (i * dt for i in factors)
    cases = itertools.product(L_generator, g_generator, dt_generator)

    # TODO: get mayavi concurrent to work (problem with mlab.figure())
    i: int = 0
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for L, g, dt in cases:
            i = i + 1
            print(f"i={i}, L={L}, g={g}, dt={dt}")
            file_name = f"split_{i:03}.mp4"
            executor.submit(simulate_case,
                            MyBox,
                            Res,
                            max_timesteps=2001,
                            dt=dt,
                            g=g,
                            g_qf=g_qf,
                            e_dd=e_dd,
                            imag_time=True,
                            mu=1.1,
                            E=1.0,
                            psi_0=psi_0_2d,
                            V=V_2d,
                            V_interaction=None,
                            psi_sol=None,
                            mu_sol=None,
                            plot_psi_sol=False,
                            plot_V=False,
                            psi_0_noise=None,
                            alpha_psi=0.8,
                            alpha_psi_sol=0.5,
                            alpha_V=0.3,
                            accuracy=10 ** -8,
                            filename="anim.mp4",
                            x_lim=(-2.0, 2.0),
                            y_lim=(-2.0, 2.0),
                            z_lim=(0, 0.5),
                            interactive=True,
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
                            )
    print("Multi core done")

