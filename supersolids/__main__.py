#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D.

"""

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from supersolids.Animation.Animation import Animation
from supersolids.Schroedinger import Schroedinger
from supersolids.tools.simulate_case import simulate_case
from supersolids.tools.cut_1d import prepare_cuts
from supersolids.helper import constants
from supersolids.helper import functions
from supersolids.helper.Resolution import Resolution, ResAssert
from supersolids.helper.Box import Box, BoxResAssert

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # Use parser to
    parser = argparse.ArgumentParser(description="Define constants for Schrödinger equation")
    parser.add_argument("-dt", metavar="dt", type=float, default=2 * 10 ** -3, nargs="?",
                        help="Length of timestep to evolve Schrödinger system.")
    parser.add_argument("-Res", metavar="Resolution", type=json.loads,
                        default={"x": 256, "y": 128, "z": 32},
                        help="Dictionary of resolutions for the box (1D, 2D, 3D). Needs to be 2 ** int.")
    parser.add_argument("-Box", metavar="Box", type=json.loads,
                        default={"x0": -10, "x1": 10, "y0": -5, "y1": 5, "z0": -4, "z1": 4},
                        help=("Dictionary for the Box dimensionality. "
                              "Two values per dimension to set start and end (1D, 2D, 3D)."))
    parser.add_argument("-N", metavar="N", type=int, default=6 * 10 ** 4,
                        help="Number of particles in box")
    parser.add_argument("-m", metavar="m", type=int, default=164.0 * constants.u_in_kg,
                        help="Mass of a particle")
    parser.add_argument("-a_dd", metavar="a_dd", type=float, default=130.0 * constants.a_0,
                        help="Constant a_dd")
    parser.add_argument("-a_s", metavar="a_s", type=float, default=85.0 * constants.a_0,
                        help="Constant a_s")
    parser.add_argument("-w_x", metavar="w_x", type=float, default=2.0 * np.pi * 33.0,
                        help="Frequency of harmonic trap in x direction")
    parser.add_argument("-w_y", metavar="w_y", type=float, default=2.0 * np.pi * 80.0,
                        help="Frequency of harmonic trap in y direction")
    parser.add_argument("-w_z", metavar="w_z", type=float, default=2.0 * np.pi * 167.0,
                        help="Frequency of harmonic trap in z direction")
    parser.add_argument("-max_timesteps", metavar="max_timesteps", type=int, default=80001,
                        help="Simulate until accuracy or maximum of steps of length dt is reached")
    parser.add_argument("-a", metavar="Amplitude", type=json.loads,
                        default={"a_x": 3.5, "a_y": 1.5, "a_z": 1.2},
                        help="Psi amplitudes in x, y, z direction for the 3d gauss packet")
    parser.add_argument("-accuracy", metavar="accuracy", type=float, default=10 ** -12,
                        help="Simulate until accuracy or maximum of steps of length dt is reached")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to save data to")
    parser.add_argument("-V", type=functions.lambda_parsed,
                        help="Potential as lambda function. For example: "
                             "-V='lambda x,y,z: 10 * x * y'")
    parser.add_argument("-noise", metavar="noise", type=json.loads,
                        default=None, action='store', nargs=2,
                        help="Min and max of gauss noise added to psi.")
    parser.add_argument("--V_none", default=False, action="store_true",
                        help="If not used, a gauss potential is used."
                             "If used, no potential is used.")
    parser.add_argument("--real_time", default=False, action="store_true",
                        help="Switch for Split-Operator method to use imaginary time or not.")
    parser.add_argument("--plot_psi_sol", default=False, action="store_true",
                        help="Option to plot the manually given solution for the wavefunction psi.")
    parser.add_argument("--plot_V", default=False, action="store_true",
                        help="Option to plot the external potential of the system (the trap).")
    parser.add_argument("-steps_per_npz", metavar="steps_per_npz",
                        type=int, default=10,
                        help="Number of dt steps skipped between saved npz.")
    parser.add_argument("--offscreen", default=False, action="store_true",
                        help="If flag is not used, interactive animation is "
                             "shown and saved as mp4, else Schroedinger is "
                             "saved as pkl and allows offscreen usage.")
    args = parser.parse_args()
    print(f"args: {args}")

    BoxResAssert(args.Res, args.Box)
    ResAssert(args.Res, args.a)
    Res = Resolution(**args.Res)

    MyBox = Box(**args.Box)

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path

    alpha_y, alpha_z = functions.get_alphas(w_x=args.w_x, w_y=args.w_y, w_z=args.w_z)
    g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
        N=args.N, m=args.m, a_s=args.a_s, a_dd=args.a_dd, w_x=args.w_x)
    print(f"g, g_qf, e_dd, alpha_y, alpha_z: {g, g_qf, e_dd, alpha_y, alpha_z}")

    # Define functions (needed for the Schroedinger equation)
    # (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=alpha_y)
    V_3d = functools.partial(functions.v_harmonic_3d, alpha_y=alpha_y, alpha_z=alpha_z)

    V_3d_ddi = functools.partial(functions.dipol_dipol_interaction,
                                 r_cut=1.0 * MyBox.min_length() / 2.0)

    # functools.partial sets all arguments except x, y, z,
    # psi_0_1d = functools.partial(functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    a_len = len(args.a)
    if a_len == 1:
        psi_0_1d = functools.partial(functions.psi_gauss_1d, a=args.a["a_x"], x_0=2.0, k_0=0.0)
    elif a_len == 2:
        psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf,
                                     mu=[0.0, 0.0],
                                     var=np.array([[args.a["a_x"], 0.0], [0.0, args.a["a_y"]]])
                                     )
    else:
        psi_0_3d = functools.partial(
            functions.psi_gauss_3d,
            a_x=args.a["a_x"], a_y=args.a["a_y"], a_z=args.a["a_z"],
            x_0=0.0, y_0=0.0, z_0=0.0,
            k_0=0.0)
        # psi_0_3d = functools.partial(functions.prob_in_trap, R_r=R_r, R_z=R_z)

    if args.noise is None:
        psi_0_noise_3d = None
    else:
        psi_0_noise_3d = functions.noise_mesh(
            min=args.noise[0],
            max=args.noise[1],
            shape=(Res.x, Res.y, Res.z)
            )

    # Used to remember that 2D need the special pos function (g is set inside
    # of Schroedinger for convenience)
    psi_sol_1d = functions.thomas_fermi_1d
    psi_sol_2d = functions.thomas_fermi_2d_pos

    # psi_sol_3d = functions.thomas_fermi_3d
    if MyBox.dim == 3:
        psi_sol_3d: Optional[Callable] = prepare_cuts(functions.density_in_trap,
                                                      args.N, alpha_z, e_dd,
                                                      a_s_l_ho_ratio)
    else:
        psi_sol_3d = None

    if Res.dim == 1:
        x_lim = (MyBox.x0, MyBox.x1)
        y_lim = (-1, 1) # arbitrary as not used
        V_trap = V_1d
        psi_0 = psi_0_1d
        psi_sol = psi_sol_1d
        V_interaction = None
    elif Res.dim == 2:
        x_lim = (MyBox.x0, MyBox.x1)
        y_lim = (MyBox.y0, MyBox.y1)
        V_trap = V_2d
        psi_0 = psi_0_2d
        psi_sol = psi_sol_2d
        V_interaction = None
    elif Res.dim == 3:
        x_lim = (MyBox.x0, MyBox.x1) # arbitrary as not used (mayavi vs matplotlib)
        y_lim = (MyBox.y0, MyBox.y1) # arbitrary as not used (mayavi vs matplotlib)
        V_trap = V_3d
        psi_0 = psi_0_3d
        psi_sol = psi_sol_3d
        V_interaction = V_3d_ddi
    else:
        sys.exit("Spatial dimension over 3. This is not implemented.")

    if args.V_none:
        V = None
    else:
        if args.V is not None:
            V = (lambda x, y, z: V_trap(x, y, z) + args.V(x, y, z))
        else:
            V = V_trap

    System: Schroedinger = Schroedinger(args.N,
                                        MyBox,
                                        Res,
                                        max_timesteps=args.max_timesteps,
                                        dt=args.dt,
                                        g=g,
                                        g_qf=g_qf,
                                        w_x=args.w_x,
                                        w_y=args.w_y,
                                        w_z=args.w_z,
                                        e_dd=e_dd,
                                        a_s=args.a_s,
                                        imag_time=(not args.real_time),
                                        mu=1.1,
                                        E=1.0,
                                        psi_0=psi_0,
                                        V=V,
                                        V_interaction=V_interaction,
                                        psi_sol=psi_sol,
                                        mu_sol=functions.mu_3d,
                                        psi_0_noise=psi_0_noise_3d,
                                        )

    Anim: Animation = Animation(Res=System.Res,
                                plot_psi_sol=args.plot_psi_sol,
                                plot_V=args.plot_V,
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
                                    System,
                                    Anim,
                                    accuracy=args.accuracy,
                                    delete_input=False,
                                    dir_path=dir_path,
                                    slice_indices=slice_indices, # from here just mayavi
                                    offscreen=args.offscreen,
                                    x_lim=x_lim, # from here just matplotlib
                                    y_lim=y_lim,
                                    steps_per_npz=args.steps_per_npz,
                                    frame_start=0,
                                    )

    print("Single core done")
