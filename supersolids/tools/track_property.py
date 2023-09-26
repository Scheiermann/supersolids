#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
 Track a property of an Schrodinger object. For example the center of mass.

"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import dill
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional

from supersolids.helper import get_version

__GPU_OFF_ENV__, __GPU_INDEX_ENV__ = get_version.get_env_variables()
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np,
                                                               gpu_off=__GPU_OFF_ENV__,
                                                               gpu_index=__GPU_INDEX_ENV__)

from supersolids import Schroedinger
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper import get_path, functions


def get_dim(list_of_arrays, property_all):
    try:
        if list_of_arrays:
            property_length = np.shape(property_all[0])[0]
            dim = property_length
        else:
            property_length = np.shape(property_all)[0]
            dim = property_all.shape[1]
    except Exception:
        dim = 1
        
    return dim


def get_input_path(dir_path, dir_name):
    if dir_name is not None:
        input_path = Path(dir_path, dir_name)
    else:
        input_path, _, _, _ = get_path.get_path(dir_path)

    return input_path


def get_last_index(input_path, filename_steps):
    _, last_index, _, _ = get_path.get_path(input_path,
                                            search_prefix=filename_steps,
                                            file_pattern=".npz"
                                            )

    return last_index


def get_property(System: Schroedinger,
                 property_name: str = "get_center_of_mass",
                 ):
    if hasattr(System, property_name):
        property = getattr(System, property_name)
    else:
        sys.exit(f"The loaded Schroedinger object has no property named {property_name}.")

    return property


def property_check(property,
                   property_name: str = "get_center_of_mass",
                   property_func: bool = False,
                   property_args=[],
                   ):
    if property_func:
        property_result = property(*property_args)
    elif callable(property):
        sys.exit(f"{property_name} is a function, but flag property_func is not set.")
    else:
        property_result = property

    return property_result


def track_property(input_path,
                   filename_schroedinger=f"schroedinger.pkl",
                   filename_steps=f"step_",
                   steps_format: str = "%07d",
                   steps_per_npz: int = 10,
                   frame_start: int = 0,
                   frame_end: Optional[int] = None,
                   property_name: str = "get_center_of_mass",
                   property_func: bool = False,
                   property_args=[],
                   property_args_str=[],
                   property_args_frame: bool = False,
                   ):
    last_index = get_last_index(input_path, filename_steps)
    print("Load schroedinger")
    try:
        path_schroedinger = Path(input_path, filename_schroedinger)
        with open(path_schroedinger, "rb") as f:
            # WARNING: this is just the input Schroedinger at t=0
            System_loaded = dill.load(file=f)

        # read new frames until Exception (last frame read)
        frame = frame_start
        while True:
            print(f"frame={frame}")
            try:
                # if property is attribute, loading of psi_val is not needed as it is saved in Summary
                if property_func:
                    # get the psi_val of Schroedinger at other timesteps (t!=0)
                    psi_val_path = Path(input_path, filename_steps + steps_format % frame + ".npz")
                    if isinstance(System_loaded, SchroedingerMixture):
                        with open(psi_val_path, "rb") as f:
                            psi_val_pkl = np.load(file=f)["psi_val_list"]
                            System_loaded.psi_val_list = psi_val_pkl
                    else:
                        with open(psi_val_path, "rb") as f:
                            psi_val_pkl = np.load(file=f)["psi_val"]
                            System_loaded.psi_val = psi_val_pkl

                System_loaded = System_loaded.load_summary(input_path, steps_format, frame,
                                                           summary_name=None)

            except zipfile.BadZipFile:
                print(
                    f"Zipfile with frame {frame} can't be read. Maybe the simulation "
                    "was stopped before file was successfully created."
                    "Animation is built until, but without that frame.")
                break

            except FileNotFoundError:
                break

            frame = frame + steps_per_npz

            if property_args_frame:
                # add fram as extra argument
                yield property_check(get_property(System_loaded, property_name),
                                     property_name,
                                     property_func,
                                     property_args_str + property_args + [frame])
            else:
                yield property_check(get_property(System_loaded, property_name),
                                     property_name,
                                     property_func,
                                     property_args_str + property_args)

            if frame_end:
                last_index = frame_end

            if frame == last_index + steps_per_npz:
                break
            elif frame > last_index:
                frame = last_index
    except FileNotFoundError:
        print(f"{path_schroedinger} not found. Skipping!")

def property_to_array(property_tuple, list_of_arrays: bool=False):
    property_all = np.empty(shape=(1, 1))
    # initialize with first value
    for property_components_all in property_tuple:
        property_all = property_components_all
        break
    # load all other values (first value already consumed from generator)
    for i, property_components_all in enumerate(property_tuple):
        try:
            if list_of_arrays:
                try:
                    property_all = np.dstack((property_all, property_components_all))
                except Exception as e:
                    print(f"Conversion from np to cp needed:\n{e}")
                    property_all = cp.dstack((property_all, property_components_all))
            else:
                try:
                    property_all = np.vstack((property_all, property_components_all))
                except Exception as e:
                    print(f"Conversion from np to cp needed:\n{e}")
                    property_all = cp.vstack((property_all, property_components_all))
        except ValueError:
            sys.exit(f"Failed at {i}: {property_components_all}. Not enough values. "
                     "Adjust provided arguments of property.")

    return property_all


def plot_property(args, func=functions.identity):
    dir_path = Path(args.dir_path).expanduser()
    if not dir_path.exists():
        sys.exit(f"Path given to load from does not exist. Correct the input via dir_path flag.")

    input_path = get_input_path(dir_path, args.dir_name)
    print(input_path)
    if not input_path.is_dir():
        return
    if args.frame_start is None:
        _, last_index, _, _ = get_path.get_path(input_path,
                                                search_prefix=args.filename_steps,
                                                counting_format=args.steps_format,
                                                file_pattern=".npz",
                                                take_last=args.take_last)
        frame_start = last_index
    else:
        frame_start = args.frame_start

    print("Load t")
    t_tuple = track_property(input_path=input_path,
                             filename_schroedinger=args.filename_schroedinger,
                             filename_steps=args.filename_steps,
                             steps_format=args.steps_format,
                             steps_per_npz=args.steps_per_npz,
                             frame_start=frame_start,
                             frame_end=args.frame_end,
                             property_name="t",
                             property_func=False,
                             property_args=[],
                             property_args_str=[],
                             property_args_frame=False,
                             )

    t_list = property_to_array(t_tuple, list_of_arrays=False)
    t = np.ravel(np.array(t_list))
    # t = np.arange(0, len(t)) * 0.0002

    property_tuple = track_property(input_path=input_path,
                                    filename_schroedinger=args.filename_schroedinger,
                                    filename_steps=args.filename_steps,
                                    steps_format=args.steps_format,
                                    steps_per_npz=args.steps_per_npz,
                                    frame_start=frame_start,
                                    frame_end=args.frame_end,
                                    property_name=args.property_name,
                                    property_func=args.property_func,
                                    property_args=args.property_args,
                                    property_args_str=args.property_args_str,
                                    property_args_frame=args.property_args_frame,
                                    )

    property_all = property_to_array(property_tuple, list_of_arrays=args.list_of_arrays)

    if cupy_used:
        try:
            property_all = property_all.get() 
        except AttributeError as e:
            pass
        except Exception as e:
            print(f"ERROR: {e}")

    dim = get_dim(args.list_of_arrays, property_all)

    path_output = Path(input_path, f"{args.property_name + args.property_filename_suffix}")
    if dim == 1:
        property_all = property_all.ravel()
        x_range, y_range = func(t, property_all, *args.inbuild_func_args)

        plt.plot(x_range, y_range, ".-")
        plt.xlabel(rf"t with dt={args.dt}")
        plt.ylabel(f"{args.property_name}")
        plt.grid()
        plt.title(f"with property_args: {args.property_args}")
        # plt.legend()

    else:
        labels = []
        if args.subplots:
            fig, axes = plt.subplots(nrows=dim, ncols=1, squeeze=False, sharex='col')
            for i, ax in enumerate(plt.gcf().get_axes()):
                if args.list_of_arrays:
                    number_of_components = len(property_all)
                    for j in range(0, number_of_components):
                        labels.append(f"component {j} axis {i}")
                        x_range, y_range = func(t, property_all[j, i, :], *args.inbuild_func_args)
                        ax.plot(x_range, y_range, ".-", label=labels[-1])

                else:
                    labels.append(str(i))
                    x_range, y_range = func(t, property_all.T[i], *args.inbuild_func_args)
                    ax.plot(x_range, y_range, ".-", label=labels[i])
                ax.grid()
                ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            axes[0, 0].figure.text(0.05, 0.5, f"{args.property_name}", ha="center", va="center", rotation=90)
            axes[0, 0].figure.text(0.5, 0.04, rf"t with dt={args.dt}", ha="center", va="baseline")

            plt.suptitle(f"{args.property_name}({', '.join(map(str, args.property_args))})")
            plt.subplots_adjust(left=0.15, bottom=0.2)

        else:
            for i in range(0, dim):
                labels.append(str(i))
                x_range, y_range = func(t, property_all.T[i], *args.inbuild_func_args)
                plt.plot(x_range, y_range, ".-", label=labels[i])

            plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.xlabel(rf"t with dt={args.dt}")
            plt.ylabel(f"{args.property_name}")
            plt.grid()
            plt.title(f"with property_args: {args.property_args}")
            plt.legend()

    # save plot as png and values as npz
    if args.property_name:
        plt.savefig(path_output)
    with open(path_output.with_suffix(".npz"), "wb") as g:
        np.savez_compressed(g, t=t, property_all=property_all)


def flags(args_array):
    parser = argparse.ArgumentParser(
        description="Load old simulations of Schrödinger system and get property.")
    parser.add_argument("-dt", metavar="dt", type=float, nargs="?", required=True,
                        help="Length of timestep to evolve Schrödinger system.")
    parser.add_argument("-dir_path", type=str,
                        default="~/supersolids/results",
                        help="Absolute path to load data from")
    parser.add_argument("-dir_name", type=str, default="movie" + "%03d" % 1,
                        help="Name of directory where the files to load lie. "
                             "For example the standard naming convention is movie001")
    parser.add_argument("-filename_schroedinger", type=str, default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-filename_steps", type=str, default="step_",
                        help="Name of file, without enumarator for the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", type=str, default="%07d",
                        help="Formating string to enumerate the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is percent 07d")
    parser.add_argument("-steps_per_npz", type=int, default=10,
                        help="Number of dt steps skipped between saved npz.")
    parser.add_argument("-frame_start", type=int, default=None, help="Counter of first saved npz.")
    parser.add_argument("-frame_end", type=int, default=None, help="Counter of last saved npz.")
    parser.add_argument("-take_last", type=int, default=None, help="Index to automatically get the "
                        "last n-th npz number of the current movie and use it as frame_start.")
    parser.add_argument("-property_filename_suffix", type=str, default="", nargs="?",
                        help="Suffix to the filename of the property plot.")
    parser.add_argument("-property_name", type=str, default="mu",
                        help="Name of property to get from the Schroedinger object.")
    parser.add_argument("--property_func", default=False, action="store_true",
                        help="If not used, flag property_name will be a interpreted as property of "
                             "an Schroedinger object."
                             "If used, flag property_name will be a interpreted as method of an "
                             "Schroedinger object.")
    parser.add_argument("--property_args_str", default=[], action='store', nargs="*",
                        help="String arguments for property_name, if property_func is used.")
    parser.add_argument("--property_args", type=json.loads, default=[], action='store', nargs="*",
                        help="Arguments for property_name, if property_func is used.")
    parser.add_argument("--property_args_frame", default=False, action="store_true",
                        help="If used, frame number is added as argument to property_args list.")
    parser.add_argument("--subplots", default=False, action="store_true",
                        help="If used, the dimensions of the property will be plotted in subplots.")
    parser.add_argument("-inbuild_func", type=functions.lambda_parsed,
                        help="Function to construct new properties "
                             "from t and the in-build property_name.")
    parser.add_argument("--inbuild_func_args", type=json.loads, default=[], action='store', nargs="*",
                        help="Arguments for inbuild_func, if used.")
    parser.add_argument("--list_of_arrays", default=False, action="store_true",
                        help="Use to track properties that are arrays of SchroedingerMixture. "
                             "As example center_of_mass for 2 components Mixture: "
                             "[(x1, y1, z1), (x2, y2, z2)]")
    parser.add_argument("-func", type=functions.lambda_parsed,
                        help="User-defined function to construct new properties "
                             "from t and the in-build property_name. "
                             "It is called with func(t, property_all.T[i])")

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])
    # func gets gets the arguments: t, property_all.T[i]
    if args.inbuild_func and args.func:
        sys.exit(f"ERROR: Choose inbuild_func or func.\n")
    elif args.inbuild_func:
        plot_property(args, func=args.inbuild_func)
    else:
        if args.func:
            plot_property(args, func=args.func)
        else:
            # if nothing provided, use identity
            plot_property(args)
