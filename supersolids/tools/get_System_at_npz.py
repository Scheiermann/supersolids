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


def get_property_one(args, dir_path: Path, i: int):
    if args.frame is None:
        _, last_index, _, _ = get_path(Path(dir_path, args.dir_name_list[i]),
                                       search_prefix=args.filename_steps,
                                       counting_format=args.steps_format,
                                       file_pattern=".npz")
        frame = last_index
    else:
        frame = args.frame

    System = get_System_at_npz(dir_path=dir_path,
                               dir_name=f"{args.dir_name_list[i]}",
                               filename_schroedinger=args.filename_schroedinger,
                               filename_steps=args.filename_steps,
                               steps_format=args.steps_format,
                               frame=frame,
                               )
    property_one = property_check(get_property(System, args.property_name),
                                  args.property_name,
                                  args.property_func,
                                  args.property_args_list[i])

    return property_one


def get_property_all(args, dir_path: Path):
    property_values = []
    for i in range(0, len(args.dir_name_list)):
        property_one = get_property_one(args, dir_path, i)
        property_values.append(property_one)

    print(f"Extracted property_values: {property_values}")
    print(f"Extracted len(property_values): {len(property_values)}")

    return property_values


def plot_System_at_npz(property_name, dir_path, var1_mesh, var2_mesh, property_values):
    var1_ravel = np.ravel(var1_mesh)
    var2_ravel = np.ravel(var2_mesh)

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

    ax.set_xlabel(r"Ratio $\frac{N_2}{N}$ (var1)")
    ax.set_ylabel(f"{property_name}")
    ax.set_xticks(np.arange(len(var1_ravel)))
    ax.set_xticklabels([round(elem, 3) for elem in var1_ravel])
    ax.tick_params(axis="x", rotation=90)
    secx = ax.secondary_xaxis("top")
    secx.set_xticks(np.arange(len(var2_ravel)))
    secx.set_xticklabels([round(elem, 3) for elem in var2_ravel])
    secx.tick_params(axis="x", rotation=90)
    secx.set_xlabel(r"Scatter length $a_{12}$ (var2)")
    ax.grid()
    # ax.set_title(f"with property_args: {property_args}")
    if property_name:
        fig.savefig(Path(dir_path, f"{property_name}"))


def plot_contour(property_name, dir_path, X, Y, property_values, title,
                 mesh=False, levels=None, var1_cut=None, var2_cut=None):
    property_arr = np.array(property_values)
    Z_half_list = []
    for i in range(0, len(property_values[0])):
        if mesh:
            path_output = Path(dir_path, f"{property_name}" + "_mesh_" + f"{i}")
        else:
            path_output = Path(dir_path, f"{property_name}" + "_contour_" + f"{i}")
        Z = np.reshape(property_arr[:, i], var1_mesh.shape)
        if var1_cut:
            X = X[:var1_cut, :]
            Y = Y[:var1_cut, :]
            Z = Z[:var1_cut, :]
        if var2_cut:
            X = X[:, :var2_cut]
            Y = Y[:, :var2_cut]
            Z = Z[:, :var2_cut]
        Z_half_list.append(Z / 2.0)
        plot_contour_helper(path_output, X, Y, Z, title, mesh=mesh, levels=levels)

    if mesh:
        path_output_sum = Path(dir_path, f"{property_name}" + "_mesh_sum")
    else:
        path_output_sum = Path(dir_path, f"{property_name}" + "_contour_sum")
    plot_contour_helper(path_output_sum, X, Y, sum(Z_half_list), title, mesh=mesh, levels=levels)


def plot_contour_helper(path_output, X, Y, Z, title, mesh=False, levels=None):
    fig, ax = plt.subplots(figsize=(8,6))
    if mesh:
        im = ax.pcolormesh(X, Y, Z, shading="auto")
        # text of value for every mesh-point
        for j in range(0, Z.shape[0]):
            for k in range(0, Z.shape[1]):
                text = ax.text(X[j, k], Y[j, k], np.round(Z[j, k], 4),
                               ha="center", va="center", color="black", size=5, rotation="vertical")
    else:
        if levels:
            im = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="both")
        else:
            im = ax.contourf(X, Y, Z, cmap="viridis")
        ax.contour(im)
        ax.clabel(im, inline=True, fontsize=10, colors="black")

    ax.set_xlabel(r"Scatter length $a_{12}$ (var2)")
    ax.set_ylabel(r"Ratio $\frac{N_2}{N}$ (var1)")
    fig.colorbar(im)
    ax.set_title(title)
    fig.savefig(path_output)


def flags(args_array):
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load npz data from")
    parser.add_argument("-dir_name_list", metavar="dir_name", type=str, default="movie001",
                        action="store", nargs="*",
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
    parser.add_argument("-var1_arange", type=json.loads, default=None, action='store',
                        nargs="*", help="List of values for var1 in phasediagram.")
    parser.add_argument("-var2_arange", type=json.loads, default=None, action='store',
                        nargs="*", help="List of values for var1 in phasediagram.")
    parser.add_argument("-property_name", type=str, default="mu",
                        help="Name of property to get from the Schroedinger object.")
    parser.add_argument("--property_func", default=False, action="store_true",
                        help="If not used, flag property_name will be a interpreted as property of "
                             "an Schroedinger object."
                             "If used, flag property_name will be a interpreted as method of an "
                             "Schroedinger object.")
    parser.add_argument("--property_args_list", default=[], type=json.loads,
                        action='append', nargs="*",
                        help="Arguments for property_name, if property_func is used.")

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])

    property_filename = f"{args.property_name}" + ".pkl"
    graphs_dirname = "graphs"

    var1_mesh, var2_mesh = np.meshgrid(args.var1_arange, args.var2_arange, indexing="ij")
    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path
    path_graphs = Path(dir_path, graphs_dirname)

    try:
        with open(Path(path_graphs, property_filename), "rb") as f:
            # WARNING: this is just the input Schroedinger at t=0
            property_values = dill.load(file=f)
    except:
        property_values = get_property_all(args, dir_path)
        with open(Path(path_graphs, property_filename), "wb") as f:
            dill.dump(obj=property_values, file=f)

    plot_System_at_npz(args.property_name, path_graphs, var1_mesh, var2_mesh, property_values)

    title = f"with property_args: {args.property_args_list[0]}"
    plot_contour(args.property_name + f"_level", path_graphs, var2_mesh, var1_mesh, property_values,
                 title,
                 levels=[0.75, 0.8, 0.90, 0.935, 0.97, 0.98, 0.99, 0.999, 1.0])
    plot_contour(args.property_name, path_graphs, var2_mesh, var1_mesh, property_values, title)
    plot_contour(args.property_name, path_graphs, var2_mesh, var1_mesh, property_values, title,
                 mesh=True, var2_cut=None)
