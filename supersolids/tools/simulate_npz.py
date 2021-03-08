#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D in single-core.

"""
import argparse
import functools
import json
from pathlib import Path

import dill
import numpy as np

from supersolids.Animation.Animation import Animation

from supersolids.Schroedinger import Schroedinger
from supersolids.helper import functions
from supersolids.tools.simulate_case import simulate_case

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # Use parser to
    parser = argparse.ArgumentParser(
        description="Load old simulations of Schrödinger system "
                    "and continue simulation from there.")
    parser.add_argument("-dt", metavar="dt", type=float, default=2 * 10 ** -3,
                        nargs="?",
                        help="Length of timestep to evolve Schrödinger system")
    parser.add_argument("-Res", metavar="Resolution", type=json.loads,
                        default=None,
                        help="Dictionary of resolutions for the box (1D, 2D, 3D). "
                             "Needs to be 2 ** int.")
    parser.add_argument("-Box", metavar="Box", type=json.loads,
                        default=None,
                        help=("Dictionary for the Box dimensionality. "
                              "Two values per dimension to set start and end (1D, 2D, 3D)."))
    parser.add_argument("-w", metavar="Trap frequency", type=json.loads,
                        default=None,
                        help="Frequency of harmonic trap in x, y, z direction. If None, "
                        "frequency of the loaded System from the npz is taken.")
    parser.add_argument("-max_timesteps", metavar="max_timesteps", type=int,
                        default=80001,
                        help="Simulate until accuracy or maximum of steps of length dt is reached")
    parser.add_argument("-accuracy", metavar="accuracy", type=float,
                        default=10 ** -12,
                        help="Simulate until accuracy or maximum of steps of length dt is reached")
    parser.add_argument("-V", type=functions.V_3d,
                        help="Potential as lambda function. For example: "
                             "-V='lambda x,y,z: 10 * x * y'")
    parser.add_argument("-noise", metavar="noise", type=json.loads,
                        default=None, action='store', nargs=2,
                        help="Min and max of gauss noise added to psi.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str,
                        default="~/supersolids/results",
                        help="Absolute path to save data to")
    parser.add_argument("-dir_name", metavar="dir_name", type=str,
                        default="movie" + "%03d" % 1,
                        help="Name of directory where the files to load lie. "
                             "For example the standard naming convention is movie001")
    parser.add_argument("-filename_schroedinger",
                        metavar="filename_schroedinger", type=str,
                        default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-filename_npz", metavar="filename_npz",
                        type=str, default="step_" + "%06d" % 0 + ".npz",
                        help="Name of file, where psi_val is saved. "
                             "For example the standard naming convention is step_000001.npz")
    parser.add_argument("-steps_per_npz", metavar="steps_per_npz",
                        type=int, default=10,
                        help="Number of dt steps skipped between saved npz.")
    parser.add_argument("--offscreen", default=False, action="store_true",
                        help="If not used, interactive animation is shown and saved as mp4."
                             "If used, Schroedinger is saved as pkl and allows offscreen usage.")
    args = parser.parse_args()
    print(f"args: {args}")

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path

    input_path = Path(dir_path, args.dir_name)
    schroedinger_path = Path(input_path, args.filename_schroedinger)
    psi_val_path = Path(input_path, args.filename_npz)

    Anim: Animation = Animation(plot_psi_sol=False,
                                plot_V=False,
                                alpha_psi=0.8,
                                alpha_psi_sol=0.5,
                                alpha_V=0.3,
                                filename="anim.mp4",
                                )

    try:
        print("Load schroedinger")
        with open(schroedinger_path, "rb") as f:
            # WARNING: this is just the input Schroedinger at t=0
            System_loaded = dill.load(file=f)

        print(f"File at {schroedinger_path} loaded.")
        try:
            # get the psi_val of Schroedinger at other timesteps (t!=0)
            with open(psi_val_path, "rb") as f:
                System_loaded.psi_val = np.load(file=f)["psi_val"]

            # get the frame number as it encodes the number steps dt,
            # so System.t can be reconstructed
            frame = int(args.filename_npz.split(".npz")[0].split("_")[-1])
            System_loaded.t = System_loaded.dt * frame
            System_loaded.max_timesteps = args.max_timesteps

            if args.Box is None:
                Box: functions.Box = System_loaded.Box
            else:
                Box = functions.Box(**args.Box)

            if args.Res is None:
                Res: functions.Resolution = System_loaded.Res
            else:
                Res = functions.Resolution(**args.Res)

            if args.w is None:
                w_x = System_loaded.w_x
                w_y = System_loaded.w_y
                w_z = System_loaded.w_z
                alpha_y, alpha_z = functions.get_alphas(w_x=w_x, w_y=w_y, w_z=w_z)
            else:
                w_x = args.w["w_x"]
                w_y = args.w["w_y"]
                w_z = args.w["w_z"]
                alpha_y, alpha_z = functions.get_alphas(w_x=w_x, w_y=w_y, w_z=w_z)

            V_loaded = functools.partial(functions.v_harmonic_3d,
                                         alpha_y=alpha_y,
                                         alpha_z=alpha_z)

            if args.V is None:
                V = V_loaded
            else:
                if System_loaded.V is None:
                    V = (lambda x, y, z: args.V(x, y, z))
                else:
                    V = (lambda x, y, z: V_loaded(x, y, z) + args.V(x, y, z))

            System: Schroedinger = Schroedinger(System_loaded.N,
                                                Box,
                                                Res,
                                                max_timesteps=args.max_timesteps,
                                                dt=args.dt,
                                                dt_func=System_loaded.dt_func,
                                                g=System_loaded.g,
                                                g_qf=System_loaded.g_qf,
                                                w_x=w_x,
                                                w_y=w_y,
                                                w_z=w_z,
                                                a_s=System_loaded.a_s,
                                                e_dd=System_loaded.e_dd,
                                                imag_time=System_loaded.imag_time,
                                                mu=System_loaded.mu,
                                                E=System_loaded.E,
                                                V=V,
                                                psi_0_noise=None
                                                )

            # As psi_0_noise needs to be applied on the loaded psi_val and not the initial psi_val
            # we apply noise after loading the old System
            if args.noise is None:
                System.psi_val = System_loaded.psi_val
            else:
                psi_0_noise_3d: np.ndarray = functions.noise_mesh(
                    min=args.noise[0],
                    max=args.noise[1],
                    shape=(Res.x, Res.y, Res.z)
                    )
                System.psi_val = psi_0_noise_3d * System_loaded.psi_val

            SystemResult: Schroedinger = simulate_case(
                System=System,
                Anim=Anim,
                accuracy=args.accuracy,
                delete_input=True,
                dir_path=dir_path,
                offscreen=args.offscreen,
                x_lim=(-2.0, 2.0),  # from here just matplotlib
                y_lim=(-2.0, 2.0),
                z_lim=(0, 0.5),
                steps_per_npz=args.steps_per_npz,
                frame_start=frame,
                )

        except FileNotFoundError:
            print(f"File at {psi_val_path} not found.")

    except FileNotFoundError:
        print(f"File at {schroedinger_path} not found.")
