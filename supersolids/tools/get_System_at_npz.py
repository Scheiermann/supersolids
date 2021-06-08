#!/usr/bin/env python
import argparse
import json
import zipfile
from pathlib import Path

import dill
import numpy as np
from matplotlib import pyplot as plt

from supersolids.Schroedinger import Schroedinger
from supersolids.tools.track_property import property_check, get_property


def get_System_at_npz(dir_path: Path = Path("~/supersolids/results").expanduser(),
                      dir_name: str = "movie001",
                      filename_schroedinger: str = f"schroedinger.pkl",
                      filename_steps: str = f"step_",
                      steps_format: str = "%06d",
                      frame: int = 0,
                      ) -> Schroedinger:
    """
    Gets Schroedinger at given npz

    :return: Schroedinger System
    """
    input_path = Path(dir_path, dir_name)
    with open(Path(input_path, filename_schroedinger), "rb") as f:
        # WARNING: this is just the input Schroedinger at t=0
        System: Schroedinger = dill.load(file=f)
    try:
        # get the psi_val of Schroedinger at other timesteps (t!=0)
        psi_val_path = Path(input_path, filename_steps + steps_format % frame + ".npz")
        with open(psi_val_path, "rb") as f:
            System.psi_val = np.load(file=f)["psi_val"]
    except zipfile.BadZipFile:
        print(f"Zipfile with frame {frame} can't be read. Maybe the simulation "
              "was stopped before file was successfully created."
              "Animation is built until, but without that frame.")
    except FileNotFoundError:
        print(f"File not found.")

    return System


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load npz data from")
    parser.add_argument("-dir_name", metavar="dir_name", type=str, default="movie" + "%03d",
                        help="Formatting of directory name where the files to load lie. "
                             "Use movie%03d for dir_names like movie001.")
    parser.add_argument("-filename_schroedinger", metavar="filename_schroedinger", type=str,
                        default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-filename_steps", type=str, default="step_",
                        help="Name of file, without enumarator for the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", metavar="steps_format", type=str, default="%06d",
                        help="Formating string to enumerate the files. "
                             "For example the standard naming convention is step_000001.npz, "
                             "the string needed is percent 06d")
    parser.add_argument("-frame", type=int, default=0, help="Counter of first saved npz.")
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

    args = parser.parse_args()
    print(f"args: {args}")

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path

    property_values = []
    for i in range(args.dir_start, args.dir_end + 1):
        System = get_System_at_npz(dir_path=dir_path,
                                   dir_name=f"{args.dir_name % i}",
                                   filename_schroedinger=args.filename_schroedinger,
                                   filename_steps=args.filename_steps,
                                   steps_format=args.steps_format,
                                   frame=args.frame,
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
    print(f"Extracted property_values[0].shape: {property_values[0].shape}")
    print(f"Extracted len(property_values): {len(property_values)}")

    try:
        dim = property_values[0].shape[0]
    except Exception:
        dim = 1

    if dim == 1:
        plt.plot(property_values, "x-")
    else:
        labels = []
        for i in range(0, dim):
            labels.append(str(i + 1))
            plt.plot(property_values[0].T[i], "x-", label=labels[i])
        plt.legend()

    v_d = zip(v_0_list, delta_list)
    plt.xlabel(f"Amplitude of distortion (v_0)")
    plt.ylabel(f"{args.property_name}")
    plt.xticks(np.arange(len(v_0_list)), [round(elem, 3) for elem in v_0_list], rotation=90)
    secx = plt.Axes.secondary_xaxis(plt.gca(), "top")
    secx.set_xticks(np.arange(len(delta_list)))
    secx.set_xticklabels([round(elem, 3) for elem in delta_list])
    secx.tick_params(axis="x", rotation=90)
    secx.set_xlabel(f"Frequency of distortion (delta)")
    plt.grid()
    plt.title(f"with property_args: {args.property_args}")
    if args.property_name:
        plt.savefig(Path(dir_path, f"{args.property_name}"))
    # plt.show()
