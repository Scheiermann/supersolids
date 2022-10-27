#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D in single-core.

"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from mayavi import mlab

from supersolids.Animation.Animation import Animation
from supersolids.Animation import MayaviAnimation
from supersolids.Animation.MayaviAnimation import load_System, load_System_list
from supersolids.helper import get_path


def load_npz(flag_args, host=None):
    slice_x = flag_args.slice_indices["x"]
    slice_y = flag_args.slice_indices["y"]
    slice_z = flag_args.slice_indices["z"]

    try:
        dir_path = Path(flag_args.dir_path).expanduser()
        dir_path_output = Path(flag_args.dir_path_output).expanduser()
    except Exception:
        dir_path = flag_args.dir_path
        dir_path_output = flag_args.dir_path_output

    # System_list = []
    # input_path = Path(dir_path, flag_args.dir_name)
    # path_schroedinger = Path(input_path, flag_args.filename_schroedinger)
    # System = load_System(path_schroedinger, host=host)
    # for i in range(0, len(flag_args.filename_steps_list)):
    #     # create copies of the System in a list (to fill them later with different psi_val)
    #     System_list.append(deepcopy(System))

    # _, last_index, _, _ = get_path.get_path(dir_path,
    #                                         search_prefix=flag_args.filename_steps_list[0],
    #                                         file_pattern=".npz"
    #                                         )

    # frame = flag_args.frame_start
    # while True:
    #      System_list = load_System_list(System_list, flag_args.filename_steps_list, input_path,
    #                                     flag_args.steps_format, frame, flag_args.summary_name)
    #      frame = frame + flag_args.steps_per_npz

    #      if flag_args.frame_end:
    #         last_index = flag_args.frame_end 

    #      if frame == last_index + flag_args.steps_per_npz:
    #          break
    #      elif frame > last_index:
    #          frame = last_index

    Anim: Animation = Animation(plot_V=flag_args.plot_V,
                                alpha_psi_list=flag_args.alpha_psi_list,
                                alpha_psi_sol_list=flag_args.alpha_psi_sol_list,
                                alpha_V=flag_args.alpha_V,
                                filename="anim.mp4",
                                )

    # mayavi for 3D
    MayAnim = MayaviAnimation.MayaviAnimation(Anim,
                                              dir_path=dir_path,
                                              slice_indices=[slice_x, slice_y, slice_z],
                                              host=host,
                                              dir_path_output=dir_path_output,
                                              )

    animate_wrapper = mlab.animate(MayAnim.animate_npz, delay=10, ui=flag_args.ui)
    MayAnimator = animate_wrapper(dir_path=dir_path,
                                  dir_name=flag_args.dir_name,
                                  filename_schroedinger=flag_args.filename_schroedinger,
                                  filename_steps_list=flag_args.filename_steps_list,
                                  steps_format=flag_args.steps_format,
                                  steps_per_npz=flag_args.steps_per_npz,
                                  frame_start=flag_args.frame_start,
                                  frame_end=flag_args.frame_end,
                                  arg_slices=flag_args.arg_slices,
                                  azimuth=flag_args.azimuth,
                                  elevation=flag_args.elevation,
                                  distance=flag_args.distance,
                                  sum_along=flag_args.sum_along,
                                  summary_name=flag_args.summary_name,
                                  mixture_slice_index_list=flag_args.mixture_slice_index_list,
                                  no_legend=flag_args.no_legend,
                                  cut1d_y_lim=flag_args.cut1d_y_lim,
                                  cut1d_plot_val_list=flag_args.cut1d_plot_val_list,
                                  host=host,
                                  )
    mlab.show()

    result_path_anim = MayAnim.create_movie(dir_path=MayAnim.dir_path,
                                            input_data_file_pattern="anim*.png",
                                            delete_input=flag_args.delete_input)
    result_path_cut1d = MayAnim.create_movie(dir_path=MayAnim.dir_path,
                                             input_data_file_pattern="1d_cut_*.png",
                                             delete_input=flag_args.delete_input,
                                             filename="1d_cut.mp4",
                                             host=host)

def flags(args_array):
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load data from")
    parser.add_argument("-dir_path_output", type=str, default="~/supersolids/results",
                        help="Absolute path to save data to")
    parser.add_argument("-dir_name", type=str, default="movie" + "%03d" % 1,
                        help="Name of directory where the files to load lie. "
                             "For example the standard naming convention is movie001")
    parser.add_argument("-filename_schroedinger", type=str, default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-summary_name", type=str, default="",
                        help="SchroedingerSummary_ or SchroedingerMixtureSummary_ "
                             "to load attributes from.")
    parser.add_argument("--filename_steps_list", default=["step_"], nargs="+",
                        help="List of filenames, without enumerator for the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", type=str, default="%07d",
                        help="Formatting string to enumerate the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is percent 06d")
    parser.add_argument("-steps_per_npz", type=int, default=10,
                        help="Number of dt steps skipped between saved npz.")
    parser.add_argument("-frame_start", type=int, default=0, help="Counter of first saved npz.")
    parser.add_argument("-frame_end", type=int, default=None, help="Counter of last saved npz.")
    parser.add_argument("-slice_indices", metavar="Indices to slice the plot.", type=json.loads,
                        default={"x": 0, "y": 0, "z": 0},
                        help="Indices to slice the plot in x, y, z direction.")
    parser.add_argument("--mixture_slice_index_list", default=[0], type=int, nargs="+",
                        help="List of indeces of the mixtures in filename_steps_list to take "
                             "slices from.")
    parser.add_argument("-azimuth", type=float, default=0.0, help="Phi angle in x-y-plane.")
    parser.add_argument("-elevation", type=float, default=0.0, help="Zenith angle theta in z-axis.")
    parser.add_argument("-distance", type=float, default=60.0, help="Setting for zoom.")
    parser.add_argument("-sum_along", type=int, default=None,
                        help="Index to sum along to display slices. None means no sum.")
    parser.add_argument("--alpha_psi_list", default=[], nargs="+",
                        help="Option to adjust the transparency of the list of plots.")
    parser.add_argument("--alpha_psi_sol_list", default=[], nargs="+",
                        help="Option to adjust the transparency of the list of plots.")
    parser.add_argument("--alpha_V", default=0.3,
                        help="Option to adjust the transparency of the external potential V "
                             "(trap + extra).")
    parser.add_argument("--no_legend", default=False, action="store_true",
                        help="Option to add legend as text to every frame.")
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
    parser.add_argument("--cut1d_y_lim", default=[0.0, 1.0], nargs=2,
                        help="Option to set y_lim for cut1d plots.")
    parser.add_argument("--cut1d_plot_val_list", default=[False], nargs="+",
                        help="List of options to plot System.psi_val instead of "
                             + "System.psi_val ** 2 for different filename_steps "
                             + "in the cut1d plots.")

    flag_args = parser.parse_args(args_array)
    print(f"args: {flag_args}")
    # convert list of bool strings to list of bool
    flag_args.cut1d_plot_val_list = [bool_str == "True"
                                     for bool_str in flag_args.cut1d_plot_val_list]

    return flag_args


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])
    load_npz(args)
