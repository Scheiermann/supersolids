#!/usr/bin/env python
import argparse
import json
import sys
import zipfile
from pathlib import Path

import dill
import numpy as np
from matplotlib import pyplot as plt

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper.get_path import get_path
from supersolids.tools.track_property import property_check, get_property


def get_System_at_npz(dir_path: Path = Path("~/supersolids/results").expanduser(),
                      dir_name: str = "movie001",
                      filename_schroedinger: str = f"schroedinger.pkl",
                      filename_steps: str = f"step_",
                      steps_format: str = "%07d",
                      frame: int = 0,
                      ) -> Schroedinger:
    """
    Gets Schroedinger at given npz

    :return: Schroedinger System
    """
    input_path = Path(dir_path, dir_name)
    with open(Path(input_path, filename_schroedinger), "rb") as f:
        # WARNING: this is just the input Schroedinger at t=0
        System_loaded: Schroedinger = dill.load(file=f)

    System_loaded = System_loaded.load_summary(input_path, steps_format, frame)

    try:
        # get the psi_val of Schroedinger at other timesteps (t!=0)
        psi_val_path = Path(input_path, filename_steps + steps_format % frame + ".npz")
        if isinstance(System_loaded, SchroedingerMixture):
            # get the psi_val of Schroedinger at other timesteps (t!=0)
            with open(psi_val_path, "rb") as f:
                System_loaded.psi_val_list = np.load(file=f)["psi_val_list"]
        else:
            # get the psi_val of Schroedinger at other timesteps (t!=0)
            with open(psi_val_path, "rb") as f:
                System_loaded.psi_val = np.load(file=f)["psi_val"]

    except zipfile.BadZipFile:
        print(f"Zipfile with frame {frame} can't be read. Maybe the simulation "
              "was stopped before file was successfully created."
              "Animation is built until, but without that frame.")
    except FileNotFoundError:
        print(f"File not found.")

    return System_loaded


def get_property_all(args):
    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path

    property_values = []
    for i in range(args.dir_start, args.dir_end + 1):
        if args.frame is None:
            _, last_index, _, _ = get_path(Path(dir_path,
                                                args.dir_name + f"{args.dir_name_format % i}"
                                                ),
                                           search_prefix=args.filename_steps,
                                           counting_format=args.steps_format,
                                           file_pattern=".npz")
            frame = last_index
        else:
            frame = args.frame

        System = get_System_at_npz(dir_path=dir_path,
                                   dir_name=f"{args.dir_name + args.dir_name_format % i}",
                                   filename_schroedinger=args.filename_schroedinger,
                                   filename_steps=args.filename_steps,
                                   steps_format=args.steps_format,
                                   frame=frame,
                                   )
        property_values.append(property_check(get_property(System, args.property_name),
                                              args.property_name,
                                              args.property_func,
                                              args.property_args)
                               )

    v_start, v_end, v_step = args.v_arange
    d_start, d_end, d_step = args.d_arange

    v_0_list = []
    delta_list = []
    for i, v_0 in enumerate(np.arange(v_start, v_end, v_step)):
        for j, delta in enumerate(np.arange(d_start, d_end, d_step)):
            v_0_list.append(v_0)
            delta_list.append(delta)

    print(f"Extracted v_0: {v_0_list}")
    print(f"Extracted delta: {delta_list}")
    print(f"Extracted property_values: {property_values}")
    print(f"Extracted len(property_values): {len(property_values)}")

    return dir_path, v_0_list, delta_list, property_values


def plot_System_at_npz(args, dir_path, v_0_list, delta_list, property_values):
    try:
        dim = property_values[0].shape[0]
        print(f"Extracted property_values[0].shape: {property_values[0].shape}")
        print(f"Extracted property_values[0].shape[0]: {dim}")
    except Exception:
        dim = 1

    fig, ax = plt.subplots(figsize=(16, 9))
    if dim == 1:
        ax.plot(property_values, "x-")
    else:
        labels = []
        for i in range(0, dim):
            labels.append(str(i + 1))
            ax.plot(property_values[0].T[i], "x-", label=labels[i])
        ax.legend()

    v_d = zip(v_0_list, delta_list)
    # ax.xlabel(f"Amplitude of distortion (v_0)")
    ax.set_xlabel(r"Ratio $\frac{N_2}{N}$ (v_0)")
    ax.set_ylabel(f"{args.property_name}")
    ax.set_xticks(np.arange(len(v_0_list)))
    ax.set_xticklabels([round(elem, 3) for elem in v_0_list])
    ax.tick_params(axis="x", rotation=90)
    secx = ax.secondary_xaxis("top")
    secx.set_xticks(np.arange(len(delta_list)))
    secx.set_xticklabels([round(elem, 3) for elem in delta_list])
    secx.tick_params(axis="x", rotation=90)
    # secx.set_xlabel(f"Frequency of distortion (delta)")
    secx.set_xlabel(r"Scatter length $a_{12}$ (delta)")
    ax.grid()
    ax.set_title(f"with property_args: {args.property_args}")
    if args.property_name:
        fig.savefig(Path(dir_path, f"{args.property_name}"))


def plot_contour(args, dir_path, property_values):
    v_start, v_end, v_step = args.v_arange
    d_start, d_end, d_step = args.d_arange
    x = np.arange(v_start, v_end, v_step)
    y = np.arange(d_start, d_end, d_step)
    X, Y = np.meshgrid(x, y, indexing="ij")

    property_arr = np.array(property_values)
    for i in range(0, len(property_values[0])):
        fig, ax = plt.subplots()
        Z = np.reshape(property_arr[:, i], (len(x), len(y)))
        im = ax.pcolormesh(Y, X, Z, shading="auto")
        fig.colorbar(im)
        ax.set_title(f"with property_args: {args.property_args}")
        if args.property_name:
            fig.savefig(Path(dir_path, f"{args.property_name}" + "_contour_" + f"{i}"))


def flags(args_array):
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load npz data from")
    parser.add_argument("-dir_name", metavar="dir_name", type=str, default="movie",
                        help="Formatting of directory name where the files to load lie. "
                             "Use movie%03d for dir_names like movie001.")
    parser.add_argument("-dir_name_format", metavar="dir_name_format", type=str, default="%03d",
                        help="Formatting of number of directory name. "
                             "Use %03d for dir_names like movie001.")
    parser.add_argument("-filename_schroedinger", metavar="filename_schroedinger", type=str,
                        default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-filename_steps", type=str, default="step_",
                        help="Name of file, without enumarator for the files. "
                             "For example the standard naming convention is step_0000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", metavar="steps_format", type=str, default="%07d",
                        help="Formating string to enumerate the files. "
                             "For example the standard naming convention is step_0000001.npz, "
                             "the string needed is %07d")
    parser.add_argument("-frame", type=json.loads, default=None, help="Counter of first saved npz.")
    parser.add_argument("-dir_start", type=int, default=1, help="Counter of first dir name.")
    parser.add_argument("-dir_end", type=int, default=2, help="Counter of last dir name.")
    parser.add_argument("-v_arange", metavar=("v_start", "v_end", "v_step"),
                        type=json.loads, default=None, action='store',
                        nargs=3, help="v_start, v_end, v_step, to match directories properly.")
    parser.add_argument("-d_arange", metavar=("d_start", "d_end", "d_step"),
                        type=json.loads, default=None, action='store',
                        nargs=3, help="d_start, d_end, d_step, to match directories properly.")
    parser.add_argument("-property_name", type=str, default="mu",
                        help="Name of property to get from the Schroedinger object.")
    parser.add_argument("--property_func", default=False, action="store_true",
                        help="If not used, flag property_name will be a interpreted as property of "
                             "an Schroedinger object."
                             "If used, flag property_name will be a interpreted as method of an "
                             "Schroedinger object.")
    parser.add_argument("--property_args", default=[], type=json.loads,
                        action='store', nargs="*",
                        help="Arguments for property_name, if property_func is used.")

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])
    dir_path, v_0_list, delta_list, property_values = get_property_all(args)
    plot_System_at_npz(args, dir_path, v_0_list, delta_list, property_values)
    plot_contour(args, dir_path, property_values)