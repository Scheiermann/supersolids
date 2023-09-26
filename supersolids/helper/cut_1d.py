#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation.

"""
import functools
from pathlib import Path
import traceback
from typing import Callable, List, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from supersolids.Schroedinger import Schroedinger
from supersolids.helper import functions


def cut_1d(System_list: List[Schroedinger],
           slice_indices: np.ndarray = [0, 0, 0],
           psi_sol_3d_cut_x: Optional[Callable] = None,
           psi_sol_3d_cut_y: Optional[Callable] = None,
           psi_sol_3d_cut_z: Optional[Callable] = None,
           dir_path: Path = Path(__file__).parent.parent.joinpath("results"),
           y_lim: Tuple[float, float] = (0.0, 1.0),
           plot_val_list: List[bool] = [False],
           frame: Optional[int] = None,
           steps_format: Optional[str] = None,
           mixture_slice_index_list: List[int] = [0],
           cut_names: List[str] = ["cut_x", "cut_y", "cut_z"],
           filename_steps_list: List[str] = ["step_"],
           ) -> None:
    """
    Creates 1D plots of the probability function of the System :math:`|\psi|^2`
    and if given of the solution.

    :param System: Schrödinger equations for the specified system

    :param slice_indices: Numpy array with indices of grid points
        in the directions x, y, z (in terms of System.x, System.y, System.z)
        to produce a slice/plane in mayavi,
        where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
        Max values is for e.g. System.Res.x - 1.

    :param psi_sol_3d_cut_x: 1D function after cut in x direction.

    :param psi_sol_3d_cut_y: 1D function after cut in y direction.

    :param psi_sol_3d_cut_z: 1D function after cut in z direction.

    :param dir_path: Path where to save 1d cut plots

    :param y_lim: Limit of y for plotting the 1D cut

    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = fig.gca()
    # plt.cla()
    
    ax.set_ylim(y_lim)
    ax.set_yticks(ticks=np.linspace(*y_lim, num=11),
                  labels=np.round_(np.linspace(*y_lim, num=11), decimals=4))

    cuts = []
    probs = []
    names = []
    # markers = [".-", "x-", "o-"]
    markers = [m + "-" for m in Line2D.markers.keys() if m not in ([",", None] + list(range(12)))]
    for i, (System, m, plot_val, mixture_slice_index) in enumerate(zip(System_list,
                                                                       markers,
                                                                       plot_val_list,
                                                                       mixture_slice_index_list)):

        try:
            psi_val = System.psi_val_list[mixture_slice_index]
            psi_sol = System.psi_sol_list[mixture_slice_index]
        except Exception as e:
            print(f"cut1d: Problem with psi_val_list or psi_sol_list! "
                  + f"Trying to use System.psi_val instead.\n{e}")
            try:
                psi_val = System.psi_val
                psi_sol = System.psi_sol
            except Exception as e:
                print(f"cut1d: Problem with psi_val or psi_sol!\n{e}")
        if psi_sol is None:
            l_ho = functions.get_l_ho(System.m_list[0], System.w_x)
            g = 4.0 * np.pi * System.a_s_array[0, 0] * System.N_list[0]
            # print(f"g, l_ho, a_s: {g}, {l_ho}, {System.a_s_array[0, 0]}")
            psi_sol = functools.partial(functions.thomas_fermi_3d, g=g)

        # prepare the axis where to cut through
        cut_x = np.linspace(System.Box.x0, System.Box.x1, System.Res.x)
        cut_y = np.linspace(System.Box.y0, System.Box.y1, System.Res.y)
        cut_z = np.linspace(System.Box.z0, System.Box.z1, System.Res.z)

        if plot_val:
            # prepare probability values of cut
            prob_mitte_x = np.abs(psi_val[:, slice_indices[1], slice_indices[2]])
            prob_mitte_y = np.abs(psi_val[slice_indices[0], :, slice_indices[2]])
            prob_mitte_z = np.abs(psi_val[slice_indices[0], slice_indices[1], :])
        else:
            # prepare probability values of cut
            prob_mitte_x = np.abs(psi_val[:, slice_indices[1], slice_indices[2]]) ** 2.0
            prob_mitte_y = np.abs(psi_val[slice_indices[0], :, slice_indices[2]]) ** 2.0
            prob_mitte_z = np.abs(psi_val[slice_indices[0], slice_indices[1], :]) ** 2.0

        # plot probability cuts
        color0 = plt.get_cmap("tab20c").colors[i]
        color1 = plt.get_cmap("tab20c").colors[4 + i]
        color2 = plt.get_cmap("tab20c").colors[8 + i]

        names.append([f"{mixture_slice_index}_" + cut_name for cut_name in cut_names])
        cuts.append([cut_x, cut_y, cut_z])
        probs.append([prob_mitte_x, prob_mitte_y, prob_mitte_z])

        ax.plot(cut_x, prob_mitte_x, m, color=color0, label=f"Component {mixture_slice_index} x")
        ax.plot(cut_y, prob_mitte_y, m, color=color1, label=f"Component {mixture_slice_index} y")
        ax.plot(cut_z, prob_mitte_z, m, color=color2, label=f"Component {mixture_slice_index} z")

        # plot probability cuts of solution, if given
        if psi_sol_3d_cut_x is not None:
            psi_sol_3d_cut_x = functools.partial(psi_sol,
                                                 y=System.y[slice_indices[1]],
                                                 z=System.z[slice_indices[2]])
            psi_sol_x = psi_sol_3d_cut_x(x=cut_x)
            ax.plot(cut_x, psi_sol_x, "x-", color="tab:cyan", label="x cut sol")

        if psi_sol_3d_cut_y is not None:
            psi_sol_3d_cut_y = functools.partial(psi_sol,
                                                 x=System.x[slice_indices[0]],
                                                 z=System.z[slice_indices[2]])
            psi_sol_y = psi_sol_3d_cut_y(y=cut_y)
            ax.plot(cut_y, psi_sol_y, "x-", color="tab:green", label="y cut sol")

        if psi_sol_3d_cut_z is not None:
            psi_sol_3d_cut_z = functools.partial(psi_sol,
                                                 x=System.x[slice_indices[0]],
                                                 y=System.y[slice_indices[1]])
            psi_sol_z = psi_sol_3d_cut_z(z=cut_z)
            ax.plot(cut_z, psi_sol_z, "x-", color="tab:olive", label="z cut sol")


    ax.legend()
    ax.grid()

    if frame is None:
        frame_formatted = ""
    else:
        if steps_format is None:
            frame_formatted = f"{frame}"
        else:
            frame_formatted = f"{steps_format % frame}"

    fig.savefig(Path(dir_path, f"1d_cut_{frame_formatted}.png"), bbox_inches='tight')

    for i, filename_steps in enumerate(filename_steps_list):
        for name, cut, prob in zip(names[i], cuts[i], probs[i], strict=True):
            path_npz = Path(dir_path, filename_steps + name + "_" + frame_formatted + ".npz")
            with open(path_npz, "wb") as g:
                np.savez_compressed(g, cut=cut, prob=prob)

def plot_cuts_tuples(dir_paths_list: List[List[Path]],
                     output_path: Path,
                     frame_formatted: str,
                     y_lim: Tuple[float, float] = (0.0, 1.0),
                     labels_list: List[List[str]] = [["cut_x", "cut_y", "cut_z"]],
                     normed: bool = True,
                     ):
    cuts_list, probs_list = [], []
    for dir_paths in dir_paths_list:
        cut_probs_tuple = read_cuts_paths(dir_paths) 
        cuts_list.append(cut_probs_tuple[0])
        probs_list.append(cut_probs_tuple[1])
    # cuts1, probs1 = read_cuts(dir_path1, frame_formatted, cut_names=cut_names) 
    # cuts2, probs2 = read_cuts(dir_path2, frame_formatted) 

    fig, ax = plt.subplots(figsize=(20, 10))
    ax = fig.gca()
    # plt.cla()

    ax.set_ylim(y_lim)
    y_range = np.linspace(y_lim[0], y_lim[1], num=11)
    ax.set_yticks(ticks=y_range, labels=np.round_(y_range, decimals=4))

    markers = [str(m) + "-" for m in Line2D.markers.keys() if m not in [",", "o", "v", "<", ">"]]
    # markers = [".-", "x-", "o-"]
    middle = lambda x: int((len(x) / 2) - 1)
    for j, (labels, marker, cut_xyz, probs_xyz) in enumerate(zip(labels_list, markers, cuts_list, probs_list)):
        for i, (label, cut, prob) in enumerate(zip(labels, cut_xyz, probs_xyz)):
            if normed:
                middle_index = middle(prob)
                prob = prob / prob[middle_index]
            if len(labels_list[0]) == 1:
                # just one cut, so different colors can be used for probs_list
                ax.plot(cut, prob, marker, color=plt.get_cmap("tab20c").colors[4 * j], label=label)
            else:
                ax.plot(cut, prob, marker, color=plt.get_cmap("tab20c").colors[4 * i], label=label)
    ax.legend()
    ax.grid()

    movie_str = dir_paths[0].parent.parent.stem
    path_png = Path(output_path, f"1d_cut_{movie_str}_{frame_formatted}.png")
    fig.savefig(path_png, bbox_inches='tight')
    plt.close()

    probs_max = [[np.max(prob) for prob in probs_xyz] for probs_xyz in probs_list]
    probs_middle = [[prob[middle(prob)] for prob in prob_xyz] for prob_xyz in probs_list]

    return path_png, probs_middle, probs_max


def read_cuts_paths(dir_paths: List[Path]):
    cuts = []
    probs = []
    for dir_path in dir_paths:
        try:
            with open(dir_path, "rb") as f:
                cuts.append(np.load(file=f)["cut"])
                probs.append(np.load(file=f)["prob"])
        except Exception as e:
            traceback.print_tb(e.__traceback__)

    return cuts, probs


def read_cuts(dir_path: Path, frame_formatted: str,
              cut_names: List[str] = ["cut_x", "cut_y", "cut_z"],
              ):
    for cut_name in cut_names:
        cuts = []
        probs = []
        with open(Path(dir_path, cut_name + "_" + frame_formatted + ".npz"), "rb") as f:
            cuts.append(np.load(file=f)["cut"])
            probs.append(np.load(file=f)["prob"])

    return cuts, probs

def zip_meshes(path_mesh1, path_mesh2):
    path_mesh = path_mesh1.copy()
    for ix, iy in np.ndindex(path_mesh.shape):
        path_mesh[ix, iy] = list(zip(path_mesh1[ix, iy], path_mesh2[ix, iy]))

    return path_mesh


def prepare_cuts(func: Callable, N: int, alpha_z: float,
                 e_dd: float, a_s_l_ho_ratio: float) -> Optional[Callable]:
    """
    Helper function to get :math:`R_r` and :math:`R_z` and set it for the given func.

    :param func: Function to take cuts from

    :param N: Number of particles

    :param alpha_z: Ratio between z and x frequencies of the trap :math:`w_{z} / w_{x}`

    :param e_dd: Factor :math:`\epsilon_{dd} = a_{dd} / a_{s}`

    :param a_s_l_ho_ratio: :math:`a_s` in units of :math:`l_{HO}`

    :return: func with fixed :math:`R_r` and :math:`R_z`
        (zeros of :math:`func_{125}`), if no singularity occurs, else None.

    """
    kappa = functions.get_kappa(alpha_z=alpha_z, e_dd=e_dd, x_min=0.1,
                                x_max=5.0, res=1000)
    R_r, R_z = functions.get_R_rz(kappa=kappa, e_dd=e_dd, N=N,
                                  a_s_l_ho_ratio=a_s_l_ho_ratio)
    psi_sol_3d = functools.partial(func, R_r=R_r, R_z=R_z)
    print(f"kappa: {kappa}, R_r: {R_r}, R_z: {R_z}")

    if R_r == 0.0 or R_z == 0.0 or np.isnan(R_r) or np.isnan(R_z):
        print(f"WARNING: R_r: {R_r}, R_z: {R_z}, but cannot be 0 or nan. Setting psi_sol=None.")
        return None
    else:
        return psi_sol_3d
