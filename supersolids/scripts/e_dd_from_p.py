#!/usr/bin/env python

from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def e_dd_func(a_11, a_12, a_22, a_d_11, a_d_12, a_d_22, p):
    return (a_d_11 + a_d_22 * p ** 2.0 + 2.0 * a_d_12 * p) / (a_11 + a_22 * p ** 2.0 + 2.0 * a_12 * p)

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    var1_list = []
    var2_list = []
    border_indices_list = []

    # path_anchor = Path("/bigwork/dscheier/results/graphs/test")
    # a_11 = 100.0
    # N2_part = np.arange(0.05, 0.51, 0.05)
    # a_12_list = np.array([63.7, 65.0, 67.5, 68.75, 70.0, 70.0, 70.0, 71.25, 71.25, 71.25])
    # # p_border = np.zeros(10)
    # p_border = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90])
    # p_border_xyz = np.tile(p_border, (3,1)).T
 
    a_11 = 95.0
    path_anchor = Path("/bigwork/dscheier/results/graphs/last_frame_a11_95_xyz_p-9")
    border_indices_list.append([[0, 9], [1, 12], [2, 12], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]])
    border_indices_list.append([[0, 9], [1, 9], [2, 12], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]])
    border_indices_list.append([[0, 9], [1, 10], [2, 12], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]])
    border_indices_list.append([[0, 9], [1, 11], [2, 12], [3, 12], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12]])
    var2_list.append(np.arange(0.05, 0.51, 0.05))
    var1_list.append(np.arange(61.25, 91.0, 5.0))
    var1_list.append(np.arange(62.5, 91.0, 5.0))
    var1_list.append(np.arange(63.75, 91.0, 5.0))

    # path_anchor = Path("/bigwork/dscheier/results/graphs/last_frame_mixture_a12_grid_map_xyz_p-9")
    # a_11 = 100.0
    # border_indices_list.append([[0, 1], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 3], [9, 3]])
    # border_indices_list.append([[0, 1], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [9, 2]])
    # var2_list.append(np.arange(0.05, 0.51, 0.05))
    # var1_list.append(np.arange(60.0, 81.0, 5.0))

    a_22 = a_11
    a_dd_11 = 130.8
    a_dd_12 = 130.8 * 0.9
    a_dd_22 = 130.8 * 0.9 * 0.9

    # possible different files to load
    filenames = []
    filenames.append("probs_cuts_middle.npz")
    filenames.append("probs_cuts_max.npz")
    paths = [Path(path_anchor, filename) for filename in filenames]

    # enumerate the different filenames (example:  [n1, n2, pol])
    index_property = 2
    # plot [axis x, axis y, axis z]
    axis_bool_list = [True, False, False]

    a_12_all = np.sort(np.ravel(var1_list))
    N2_all = np.sort(np.ravel(var2_list))

    # load mesh of cuts in x, y, z
    probs_cuts_mesh_list = []
    for path in paths:
        with open(path, "rb") as f:
            probs_cuts_mesh = np.load(file=f, allow_pickle=True)["probs_cuts_mesh"]
            probs_cuts_mesh_list.append(probs_cuts_mesh)

    for file_index, filename in enumerate(filenames):
        mesh_max_arr = np.array(probs_cuts_mesh_list[file_index])
        p_simulation = np.zeros(np.shape(mesh_max_arr))
        p_simulation_xyz = mesh_max_arr.copy()
        for i, (ix, iy) in enumerate(np.ndindex(mesh_max_arr.shape)):
            p_simulation_xyz[ix, iy] = mesh_max_arr[ix, iy][index_property]
        p_simulation_mirrored_xyz = np.flip(p_simulation_xyz, axis=0)

        p_border_xyz_list = []
        for border_indices in border_indices_list:
            # get values at positions of border_index
            p_border_xyz_list.append(np.array([p_simulation_mirrored_xyz[border_index[0],
                                                               border_index[1]]
                                     for border_index in border_indices])
                                     )
            N2_part = np.array([N2_all[border_index[0]] for border_index in border_indices])
            a_12_list = np.array([a_12_all[border_index[1]] for border_index in border_indices])

        # plot
        fig, axes = plt.subplots(2, 1, figsize=(26, 14))
        fontsize = 16

        # yticks = np.linspace(1.2, 1.5, num=11)
        # labels = np.round_(np.linspace(1.2, 1.5, num=11), decimals=4)
        # axes.set_ylim(yticks[0], yticks[-1])
        # axes.set_yticks(yticks, labels)

        print(f"a_12: {a_12_list}")
        for j, (p_border_xyz, border_indices) in enumerate(zip(p_border_xyz_list, border_indices_list)):
            for i, (p, axis_bool) in enumerate(zip(p_border_xyz.T, axis_bool_list)):
                if axis_bool:
                    e_dd = e_dd_func(a_11, a_12_list, a_22, a_dd_11, a_dd_12, a_dd_22, p)
                    axes[0].plot(N2_part, e_dd, [".-", "x-", "*-"][i], label=["x", "y", "z"][i] + f"{border_indices}", color=plt.get_cmap("tab20c").colors[4 * j])
                    axes[1].plot(N2_part, p, [".-", "x-", "*-"][i], label=["x", "y", "z"][i] + f"{border_indices}", color=plt.get_cmap("tab20c").colors[4 * j])
                    print(filename)
                    # print(f"p at border: {p}")
                    # print(f"e_dd: {e_dd}")

        for ax in axes:
            ax.set_xlabel(r"$N_2/N$", fontsize=fontsize)
            ax.grid()
            ax.legend(fontsize=fontsize)

        axes[0].set_ylabel(r"$\epsilon_{dd}$", fontsize=fontsize)
        axes[1].set_ylabel(r"$p$", fontsize=fontsize)

        fig.suptitle(filename, fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(Path(path_anchor, Path(filenames[file_index]).stem), bbox_inches='tight')