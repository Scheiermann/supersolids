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
import sys
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
                             "-V='lambda x,y,z: 0 * x * y * z'")
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
    parser.add_argument("--V_reload", default=True, action="store_false",
                        help="If not used, V will be the lambda function provided by the V flag."
                             "If used, the V is loaded from the provided Schroedinger, "
                             "plus the lambda function provided by the V flag.")

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
        print("\nLoad schroedinger")
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

            # check if changes of Box or Res, can be done
            x_step_old = (System_loaded.Box.lengths()[0] / System_loaded.Res.x)
            y_step_old = (System_loaded.Box.lengths()[1] / System_loaded.Res.y)
            z_step_old = (System_loaded.Box.lengths()[2] / System_loaded.Res.z)
            x_step_new = (Box.lengths()[0] / Res.x)
            y_step_new = (Box.lengths()[1] / Res.y)
            z_step_new = (Box.lengths()[2] / Res.z)
            box_offset_x = np.abs(System_loaded.Box.x0 - Box.x0)
            box_offset_y = np.abs(System_loaded.Box.y0 - Box.y0)
            box_offset_z = np.abs(System_loaded.Box.z0 - Box.z0)
            box_offset_x_end = np.abs(System_loaded.Box.x1 - Box.x0)
            box_offset_y_end = np.abs(System_loaded.Box.y1 - Box.y0)
            box_offset_z_end = np.abs(System_loaded.Box.z1 - Box.z0)
            box_offset_steps_x: int = int(box_offset_x / x_step_old)
            box_offset_steps_y: int = int(box_offset_y / y_step_old)
            box_offset_steps_z: int = int(box_offset_z / z_step_old)
            box_offset_steps_x_end: int = int(box_offset_x_end / x_step_old)
            box_offset_steps_y_end: int = int(box_offset_y_end / y_step_old)
            box_offset_steps_z_end: int = int(box_offset_z_end / z_step_old)

            # smaller steps than loaded are not allowed as then interpolation of psi value is needed
            # so e.g. x_step_new >= x_step_old
            if (x_step_new % x_step_old != 0) or (x_step_old > x_step_new):
                print(f"\nOld x_step {x_step_old} and new x_step {x_step_new} "
                f"need to be the same as psi values are calculated gridwise to "
                f"specific coordinates. These need to match, when changing Box "
                f"or Res."
                )
                sys.exit(1)
            if box_offset_x % x_step_old != 0.0:
                print(f"\nTo match the grids, the difference between the "
                f"minimum Box values ({box_offset_x}) "
                f"needs to be a multiple of the old x_step {x_step_old}."
                )
                sys.exit(1)
            if (y_step_new % y_step_old != 0) or (y_step_old > y_step_new):
                print(f"\nOld y_step {y_step_old} and new y_step {y_step_new} "
                f"need to be the same as psi values are calculated gridwise to "
                f"specific coordinates. These need to match, when changing Box "
                f"or Res."
                )
                sys.exit(1)
            if box_offset_y % y_step_old != 0.0:
                print(f"\nTo match the grids, the difference between the "
                f"minimum Box values ({box_offset_y}) "
                f"needs to be a multiple of the old y_step {y_step_old}."
                )
                sys.exit(1)
            if (z_step_new % z_step_old != 0) or (z_step_old > z_step_new):
                print(f"\nOld z_step {z_step_old} and new z_step {z_step_new} "
                f"need to be the same as psi values are calculated gridwise to "
                f"specific coordinates. These need to match, when changing Box "
                f"or Res."
                )
                sys.exit(1)
            if box_offset_z % z_step_old != 0.0:
                print(f"\nTo match the grids, the difference between the "
                f"minimum Box values ({box_offset_z}) "
                f"needs to be a multiple of the old z_step {z_step_old}."
                )
                sys.exit(1)

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

            V_harmonic = functools.partial(functions.v_harmonic_3d,
                                           alpha_y=alpha_y,
                                           alpha_z=alpha_z)

            # TODO: Usage of -V=None uses harmonic potential with w_x, w_y, w_z.
            # Used to get access to the function from bash
            # To get actually no potential use -V="lambda x,y,z: 0"
            if args.V is None:
                V = (lambda x, y, z: V_harmonic(x, y, z))
            else:
                if args.V_reload:
                    if System_loaded.V is None:
                        V = (lambda x, y, z: args.V(x, y, z))
                    else:
                        V = (lambda x, y, z: System_loaded.V(x, y, z) + args.V(x, y, z))
                else:
                    V = (lambda x, y, z: args.V(x, y, z))

            System: Schroedinger = Schroedinger(System_loaded.N,
                                                Box,
                                                Res,
                                                max_timesteps=args.max_timesteps,
                                                dt=args.dt,
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
                                                V_interaction=System_loaded.V_interaction,
                                                psi_0_noise=None
                                                )

            # Load psi values from System_loaded into System
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

            # remove the n-th slices, if Res is shrunk down
            if System.Res.x < System_loaded.Res.x:
                x_shrink = int(System_loaded.Res.x / System.Res.x)
                System.psi_val = System.psi_val[box_offset_steps_x:box_offset_steps_x_end, :, :]
            else:
                if x_step_new == x_step_old:
                    # Fill up the new grid points with 0, when adding grid points by changing Box or Res
                    System.psi_val = np.pad(
                        System.psi_val,
                        ((box_offset_steps_x, Res.x - System_loaded.Res.x - box_offset_steps_x),
                         (0, 0),
                         (0, 0))
                    )
                else:
                    box_offset_new_x_end = np.abs(System_loaded.Box.x1 - System.Box.x1)
                    box_offset_new_steps_x_end = int(box_offset_new_x_end / x_step_new)
                    box_offset_new_steps_x = int(box_offset_x / x_step_new)
                    discard_n_th_x = int(x_step_new / x_step_old)
                    psi_loaded_lower_res_x = System.psi_val[::discard_n_th_x, :, :]

                    System.psi_val = np.pad(
                        psi_loaded_lower_res_x,
                        ((box_offset_new_steps_x, box_offset_new_steps_x_end),
                         (0, 0),
                         (0, 0))
                    )

            if System.Res.y < System_loaded.Res.y:
                y_shrink = int(System_loaded.Res.y / System.Res.y)
                System.psi_val = System.psi_val[:, box_offset_steps_y:box_offset_steps_y_end, :]
            else:
                if y_step_new == y_step_old:
                    # Fill up the new grid points with 0, when adding grid points by changing Box or Res
                    System.psi_val = np.pad(
                        System.psi_val,
                        ((0, 0),
                         (box_offset_steps_y, Res.y - System_loaded.Res.y - box_offset_steps_y),
                         (0, 0))
                    )
                else:
                    box_offset_new_y_end = np.abs(System_loaded.Box.y1 - System.Box.y1)
                    box_offset_new_steps_y_end = int(box_offset_new_y_end / y_step_new)
                    box_offset_new_steps_y = int(box_offset_y / y_step_new)
                    discard_n_th_y = int(y_step_new / y_step_old)
                    psi_loaded_lower_res_y = System.psi_val[:, ::discard_n_th_y, :]

                    System.psi_val = np.pad(
                        psi_loaded_lower_res_y,
                        ((0, 0),
                         (box_offset_new_steps_y, box_offset_new_steps_y_end),
                         (0, 0))
                    )

            if System.Res.z < System_loaded.Res.z:
                z_shrink = int(System_loaded.Res.z / System.Res.z)
                System.psi_val = System.psi_val[:, :, box_offset_steps_z:box_offset_steps_z_end]
            else:
                if z_step_new == z_step_old:
                    # Fill up the new grid points with 0, when adding grid points by changing Box or Res
                    System.psi_val = np.pad(
                        System.psi_val,
                        ((0, 0),
                         (0, 0),
                         (box_offset_steps_z, Res.z - System_loaded.Res.z - box_offset_steps_z))
                    )
                else:
                    box_offset_new_z_end = np.abs(System_loaded.Box.z1 - System.Box.z1)
                    box_offset_new_steps_z_end = int(box_offset_new_z_end / z_step_new)
                    box_offset_new_steps_z = int(box_offset_z / z_step_new)
                    discard_n_th_z = int(z_step_new / z_step_old)
                    psi_loaded_lower_res_z = System.psi_val[:, :, :discard_n_th_z]

                    System.psi_val = np.pad(
                        psi_loaded_lower_res_z,
                        ((0, 0),
                         (0, 0),
                         (box_offset_new_steps_z, box_offset_new_steps_z_end))
                    )

            lol = np.abs(System.psi_val)
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
