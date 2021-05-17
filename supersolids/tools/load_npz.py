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
import json
from pathlib import Path

from mayavi import mlab

from supersolids.Animation.Animation import Animation

from supersolids.Animation import MayaviAnimation


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load data from")
    parser.add_argument("-dir_name", metavar="dir_name", type=str, default="movie" + "%03d" % 1,
                        help="Name of directory where the files to load lie. "
                             "For example the standard naming convention is movie001")
    parser.add_argument("-filename_schroedinger", metavar="filename_schroedinger", type=str,
                        default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-filename_steps", metavar="filename_steps", type=str, default="step_",
                        help="Name of file, without enumarator for the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", metavar="steps_format", type=str, default="%06d",
                        help="Formating string to enumerate the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is percent 06d")
    parser.add_argument("-steps_per_npz", metavar="steps_per_npz", type=int, default=10,
                        help="Number of dt steps skipped between saved npz.")
    parser.add_argument("-frame_start", metavar="frame_start", type=int, default=0,
                        help="Counter of first saved npz.")
    parser.add_argument("-slice_indices", metavar="Indices to slice the plot.", type=json.loads,
                        default={"x": 0, "y": 0, "z": 0},
                        help="Indices to slice the plot in x, y, z direction.")
    parser.add_argument("-azimuth", type=float, default=0.0, help="Phi angle in x-y-plane.")
    parser.add_argument("-elevation", type=float, default=0.0, help="Zenith angle theta in z-axis.")
    parser.add_argument("--plot_psi_sol", default=False, action="store_true",
                        help="Option to plot the manually given solution for the wavefunction psi")
    parser.add_argument("--plot_V", default=False, action="store_true",
                        help="Option to plot the external potential of the system (the trap)")
    parser.add_argument("--delete_input", default=False, action="store_true",
                        help="If flag is not used, the pictures after "
                             "animation is created and saved.")
    parser.add_argument("--arg_slices", default=False, action="store_true",
                        help="If flag is not used, psi ** 2 will be plotted on slices, "
                             "else arg(psi).")
    parser.add_argument("--ui", default=False, action="store_true",
                        help="If flag is used, ui for starting/stopping the animation pops up."
                             "This is handy as adjusting plots is just stable for stopped "
                             "animation.")

    args = parser.parse_args()
    print(f"args: {args}")
    slice_x = args.slice_indices["x"]
    slice_y = args.slice_indices["y"]
    slice_z = args.slice_indices["z"]

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path

    Anim: Animation = Animation(plot_psi_sol=args.plot_psi_sol,
                                plot_V=args.plot_V,
                                alpha_psi=0.8,
                                alpha_psi_sol=0.5,
                                alpha_V=0.3,
                                filename="anim.mp4",
                                )

    # mayavi for 3D
    MayAnim = MayaviAnimation.MayaviAnimation(Anim,
                                              dir_path=dir_path,
                                              slice_indices=[slice_x, slice_y, slice_z],
                                              )

    animate_wrapper = mlab.animate(MayAnim.animate_npz, delay=10, ui=args.ui)
    MayAnimator = animate_wrapper(dir_path=dir_path,
                                  dir_name=args.dir_name,
                                  filename_schroedinger=args.filename_schroedinger,
                                  filename_steps=args.filename_steps,
                                  steps_format=args.steps_format,
                                  steps_per_npz=args.steps_per_npz,
                                  frame_start=args.frame_start,
                                  arg_slices=args.arg_slices,
                                  azimuth=args.azimuth,
                                  elevation=args.elevation,
                                  )
    mlab.show()

    result_path = MayAnim.create_movie(dir_path=MayAnim.dir_path,
                                       input_data_file_pattern="*.png",
                                       delete_input=args.delete_input)
