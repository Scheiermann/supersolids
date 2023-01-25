#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper.get_path import get_path
from supersolids.tools.get_System_at_npz import get_System_at_npz


def flags(args_array):
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load npz data from")
    parser.add_argument("-dir_name", metavar="dir_name", type=str, default="movie001",
                        help="Formatting of directory name where the files to load lie. "
                             "Use movie%03d for dir_names like movie001.")
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

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])

    # home = "/bigwork/nhbbsche"
    home = "/home/dsche/supersolids"
    args.dir_path = Path(f"{home}/results/begin_gpu_01_13_dip9/")
    args.dir_name = "movie001"
    args.filename_schroedinger = "schroedinger.pkl"
    args.filename_steps = "step_"
    args.steps_format = "%07d"
    args.frame = None

    graphs_dirname = "graphs"

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path
    path_graphs = Path(dir_path, graphs_dirname)

    if args.frame is None:
        _, last_index, _, _ = get_path(Path(dir_path, args.dir_name),
                                       search_prefix=args.filename_steps,
                                       counting_format=args.steps_format,
                                       file_pattern=".npz")
        frame = last_index
    else:
        frame = args.frame

    System = get_System_at_npz(dir_path=dir_path,
                               dir_name=f"{args.dir_name}",
                               filename_schroedinger=args.filename_schroedinger,
                               filename_steps=args.filename_steps,
                               steps_format=args.steps_format,
                               frame=frame,
                               )
    print(f"{System}")

    energy_array, eigen_vectors = np.linalg.eig(bdg_H)