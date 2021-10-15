#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
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
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper.simulate_case import simulate_case
from supersolids.helper.cut_1d import prepare_cuts
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
                        help="Dictionary of resolutions for the box (1D, 2D, 3D). "
                             "Needs to be 2 ** int.")
    parser.add_argument("-Box", metavar="Box", type=json.loads,
                        default={"x0": -10, "x1": 10, "y0": -5, "y1": 5, "z0": -4, "z1": 4},
                        help=("Dictionary for the Box dimensionality. "
                              "Two values per dimension to set start and end (1D, 2D, 3D)."))
    parser.add_argument("-l_0", metavar="l_0", type=float, default=None,
                        help="Help constant for dimensionless formulation of equations.")
    parser.add_argument("--N_list", metavar="N", type=int, nargs="+", default=6 * 10 ** 4,
                        help="Number of particles in box per mixture")
    parser.add_argument("--m_list", metavar="m",  type=float, nargs="+", default=[164.0],
                        help="Mass of a particles in atomic mass unit (u) per mixture")
    parser.add_argument("--mu_list", metavar="mu",  type=float, nargs="+", default=[1.0],
                        help="Mass of a particles in mu_0 per mixture")
    parser.add_argument("--a_s_list", metavar="a_s", type=float, nargs="+", default=[85.0],
                        help="Constant a_s per mixture-mixture interaction in a upper triangle "
                             "matrix e.g. for 2 mixtures, the index combinations are [11, 12, 22].")
    parser.add_argument("-a_dd", metavar="a_dd", type=float, default=130.0 * constants.a_0,
                        help="Constant a_dd (Ignored for mixtures, "
                             "as mu_list is used to construct the values)")
    parser.add_argument("-w_x", metavar="w_x", type=float, default=2.0 * np.pi * 33.0,
                        help="Frequency of harmonic trap in x direction")
    parser.add_argument("-w_y", metavar="w_y", type=float, default=2.0 * np.pi * 80.0,
                        help="Frequency of harmonic trap in y direction")
    parser.add_argument("-w_z", metavar="w_z", type=float, default=2.0 * np.pi * 167.0,
                        help="Frequency of harmonic trap in z direction")
    parser.add_argument("-max_timesteps", metavar="max_timesteps", type=int, default=80001,
                        help="Simulate until accuracy or maximum of steps of length dt is reached")
    parser.add_argument("-a", metavar="Amplitude", type=json.loads,
                        default={"a_x": 1.0, "a_y": 1.0, "a_z": 1.0},
                        help="Psi amplitudes in x, y, z direction for the gauss packet")
    parser.add_argument("-mu", metavar="Mean psi_0", type=json.loads,
                        default={"mu_x": 0.0, "mu_y": 0.0, "mu_z": 0.0},
                        help="Mean values in x, y, z direction for the gauss packet")
    parser.add_argument("-accuracy", metavar="accuracy", type=float, default=10 ** -12,
                        help="Simulate until accuracy or maximum of steps of length dt is reached")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to save data to")
    parser.add_argument("-dir_name_result", type=str, default="",
                        help="Name of directory where to save the results at. "
                             "For example the standard naming convention is movie002")
    parser.add_argument("-V", type=functions.lambda_parsed,
                        help="Potential as lambda function. For example: "
                             "-V='lambda x,y,z: 10 * x * y'")
    parser.add_argument("-noise", metavar="noise", type=json.loads,
                        default=None, action='store', nargs=2,
                        help="Min and max of gauss noise added to psi.")
    parser.add_argument("--V_none", default=False, action="store_true",
                        help="If not used, a gauss potential is used."
                             "If used, no potential is used.")
    parser.add_argument("--V_interaction", default=False, action="store_true",
                        help="Just for 3D case. Use to apply V_3d_ddi (Dipol-Dipol-Interaction).")
    parser.add_argument("--V_interaction_cut_x", default=[], nargs="+",
                        help="Min and max values for x to cut V_3d_ddi (Dipol-Dipol-Interaction).")
    parser.add_argument("--V_interaction_cut_y", default=[], nargs="+",
                        help="Min and max values for y to cut V_3d_ddi (Dipol-Dipol-Interaction).")
    parser.add_argument("--V_interaction_cut_z", default=[], nargs="+",
                        help="Min and max values for z to cut V_3d_ddi (Dipol-Dipol-Interaction).")
    parser.add_argument("--real_time", default=False, action="store_true",
                        help="Switch for Split-Operator method to use imaginary time or not.")
    parser.add_argument("--plot_psi_list", default=None, nargs="+",
                        help="Option to plot the list of wavefunctions psi.")
    parser.add_argument("--plot_psi_sol_list", default=None, nargs="+",
                        help="Option to plot the manually given solution for the wavefunction psi.")
    parser.add_argument("--plot_V", default=False, action="store_true",
                        help="Option to plot the external potential of the system (the trap).")
    parser.add_argument("-script_name", type=str, default="script",
                        help="Name of file, where to save args of the running simulate_npz")
    parser.add_argument("-script_number_regex", type=str, default="*",
                        help="Regex to find files in method reload_files.")
    parser.add_argument("--script_extensions", default=None, nargs="+",
                        help="List to reload different files extensions at the same time.")
    parser.add_argument("-script_extensions_index", type=int, default=0,
                        help="Index of list of flag script_extension, which is used to get the "
                             "script_list from which the counter is deduced.")
    parser.add_argument("-filename_steps", type=str, default="step_",
                        help="Name of file, without enumerator for the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", type=str, default="%07d",
                        help="Formatting string for the enumeration of steps.")
    parser.add_argument("-steps_per_npz", metavar="steps_per_npz",
                        type=int, default=10,
                        help="Number of dt steps skipped between saved npz.")
    parser.add_argument("--mixture", default=False, action="store_true",
                        help="Use to simulate a SchroedingerMixture.")
    parser.add_argument("--offscreen", default=False, action="store_true",
                        help="If flag is not used, interactive animation is "
                             "shown and saved as mp4, else Schroedinger is "
                             "saved as pkl and allows offscreen usage.")

    args = parser.parse_args()
    print(f"args: {args}\n")

    # apply units to input
    m_list = [m * constants.u_in_kg for m in args.m_list]
    mu_list = [mu * constants.mu_0 for mu in args.mu_list]
    a_s_list = [a_s * constants.a_0 for a_s in args.a_s_list]
    if args.l_0 is None:
        # x harmonic oscillator length
        # l_0 = np.sqrt(constants.hbar / (m_list[0] * args.w_x))
        l_0 = 1.0 / constants.u_in_kg
    else:
        l_0 = args.l_0

    dimensionless_factor = constants.hbar ** 2.0 / (m_list[0] * l_0 ** 2.0)
    a_dd_factor = (m_list[0] / constants.hbar ** 2.0) * (constants.mu_0 / (12.0 * np.pi))

    BoxResAssert(args.Res, args.Box)
    ResAssert(args.Res, args.a)
    ResAssert(args.Res, args.mu, name="means (mu)")
    Res = Resolution(**args.Res)

    MyBox = Box(**args.Box)

    cut_ratio = 0.95
    if not args.V_interaction_cut_x:
        V_interaction_cut_x = [cut_ratio * MyBox.x0, cut_ratio * MyBox.x1]
    else:
        V_interaction_cut_x = list(map(float, args.V_interaction_cut_x))

    if not args.V_interaction_cut_y:
        V_interaction_cut_y = [cut_ratio * MyBox.y0, cut_ratio * MyBox.y1]
    else:
        V_interaction_cut_y = list(map(float, args.V_interaction_cut_y))

    if not args.V_interaction_cut_z:
        V_interaction_cut_z = [cut_ratio * MyBox.z0, cut_ratio * MyBox.z1]
    else:
        V_interaction_cut_z = list(map(float, args.V_interaction_cut_z))

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path

    if args.mixture:
        a_s_array, a_dd_array = functions.get_parameters_mixture(l_0,
                                                                 mu_list,
                                                                 a_s_list,
                                                                 a_dd_factor,
                                                                 )

        print(f"g_array:\n{a_s_array}")
        print(f"U_dd_factor_array:\n{a_dd_array}\n")
    else:
        g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
            N=args.N, m=m_list[0], a_s=args.a_s, a_dd=args.a_dd, w_x=args.w_x)
        print(f"g, g_qf, e_dd: {g, g_qf, e_dd}")

    alpha_y, alpha_z = functions.get_alphas(w_x=args.w_x, w_y=args.w_y, w_z=args.w_z)
    print(f"alpha_y, alpha_z: {alpha_y, alpha_z}")

    # Define functions (needed for the Schroedinger equation)
    # (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=alpha_y)
    V_3d = functools.partial(functions.v_harmonic_3d, alpha_y=alpha_y, alpha_z=alpha_z)

    V_3d_ddi = functools.partial(functions.get_V_k_val_ddi3,
                                 x_cut=V_interaction_cut_x,
                                 y_cut=V_interaction_cut_y,
                                 z_cut=V_interaction_cut_z)
    # V_3d_ddi = functools.partial(functions.get_V_k_val_ddi,
    #                               rho_cut=0.8 * max(MyBox.lengths()[:2]),
    #                               z_cut=0.8 * MyBox.lengths()[2])

    # functools.partial sets all arguments except x, y, z,
    # psi_0_1d = functools.partial(functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    a_len = len(args.a)
    if a_len == 1:
        psi_0_1d = functools.partial(functions.psi_gauss_1d, a=args.a["a_x"],
                                     x_0=args.mu["mu_x"], k_0=0.0)
    elif a_len == 2:
        psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf,
                                     mu=[args.mu["mu_x"], args.mu["mu_y"]],
                                     var=np.array([[args.a["a_x"], 0.0], [0.0, args.a["a_y"]]])
                                     )
    else:
        psi_0_3d = functools.partial(
            functions.psi_gauss_3d,
            a_x=args.a["a_x"], a_y=args.a["a_y"], a_z=args.a["a_z"],
            x_0=args.mu["mu_x"], y_0=args.mu["mu_y"], z_0=args.mu["mu_z"],
            k_0=0.0)
        psi_0_3d_2 = functools.partial(
            functions.psi_gauss_3d,
            a_x=0.5 * args.a["a_x"], a_y=0.5 * args.a["a_y"], a_z=1.5 * args.a["a_z"],
            x_0=args.mu["mu_x"], y_0=args.mu["mu_y"], z_0=args.mu["mu_z"],
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
        if args.mixture:
            psi_sol_3d = None
        else:
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
        mu_sol = functions.mu_1d
        V_interaction = None
    elif Res.dim == 2:
        x_lim = (MyBox.x0, MyBox.x1)
        y_lim = (MyBox.y0, MyBox.y1)
        V_trap = V_2d
        psi_0 = psi_0_2d
        psi_sol = psi_sol_2d
        mu_sol = functions.mu_2d
        V_interaction = None
    elif Res.dim == 3:
        x_lim = (MyBox.x0, MyBox.x1) # arbitrary as not used (mayavi vs matplotlib)
        y_lim = (MyBox.y0, MyBox.y1) # arbitrary as not used (mayavi vs matplotlib)
        V_trap = V_3d
        psi_0 = psi_0_3d
        psi_sol = psi_sol_3d
        mu_sol = functions.mu_3d
        if args.V_interaction:
            V_interaction = V_3d_ddi
        else:
            V_interaction = None
    else:
        sys.exit("Spatial dimension over 3. This is not implemented.")

    if args.V_none:
        V = None
    else:
        if args.V is not None:
            V = (lambda x, y, z: V_trap(x, y, z) + args.V(x, y, z))
        else:
            V = V_trap

    if args.mixture:
        SchroedingerInput: SchroedingerMixture = SchroedingerMixture(
            MyBox,
            Res,
            max_timesteps=args.max_timesteps,
            dt=args.dt,
            N_list=args.N_list,
            m_list=m_list,
            a_s_array=a_s_array,
            a_dd_array=a_dd_array,
            dt_func=None,
            w_x=args.w_x,
            w_y=args.w_y,
            w_z=args.w_z,
            imag_time=(not args.real_time),
            mu_arr=None,
            E=1.0,
            V=V,
            V_interaction=V_interaction,
            psi_0_list=[psi_0, psi_0_3d_2],
            psi_0_noise_list=[psi_0_noise_3d, psi_0_noise_3d],
            psi_sol_list=[functions.thomas_fermi_3d, functions.thomas_fermi_3d],
            mu_sol_list=[functions.mu_3d, functions.mu_3d],
            input_path=Path("~/Documents/itp/master/supersolids/supersolids/").expanduser(),
            )
    else:
        SchroedingerInput: Schroedinger = Schroedinger(
            args.N_list[0],
            MyBox,
            Res,
            max_timesteps=args.max_timesteps,
            dt=args.dt,
            dt_func=None,
            g=g,
            g_qf=g_qf,
            w_x=args.w_x,
            w_y=args.w_y,
            w_z=args.w_z,
            e_dd=e_dd,
            a_s=args.a_s,
            imag_time=(not args.real_time),
            mu_arr=1.1,
            E=1.0,
            psi_0=psi_0,
            V=V,
            V_interaction=V_interaction,
            psi_sol=psi_sol,
            mu_sol=mu_sol,
            psi_0_noise=psi_0_noise_3d,
            )

    Anim: Animation = Animation(
        Res=SchroedingerInput.Res,
        plot_V=args.plot_V,
        alpha_psi_list=[],
        alpha_psi_sol_list=[],
        alpha_V=0.3,
        camera_r_func=functools.partial(functions.camera_func_r,
                                        r_0=10.0, phi_0=45.0, z_0=50.0,
                                        r_per_frame=0.0),
        camera_phi_func=functools.partial(functions.camera_func_phi,
                                          r_0=10.0, phi_0=45.0, z_0=50.0,
                                          phi_per_frame=5.0),
        camera_z_func=functools.partial(functions.camera_func_z,
                                        r_0=10.0, phi_0=45.0, z_0=50.0,
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
        SchroedingerInput,
        Anim,
        accuracy=args.accuracy,
        delete_input=False,
        dir_path=dir_path,
        dir_name_result=args.dir_name_result,
        slice_indices=slice_indices, # from here just mayavi
        offscreen=args.offscreen,
        x_lim=x_lim, # from here just matplotlib
        y_lim=y_lim,
        filename_steps=args.filename_steps,
        steps_format=args.steps_format,
        steps_per_npz=args.steps_per_npz,
        frame_start=0,
        script_name=args.script_name,
        script_args=args,
        script_number_regex=args.script_number_regex,
        script_extensions=args.script_extensions,
        script_extensions_index=args.script_extensions_index,
        )

    print("Single core done")
