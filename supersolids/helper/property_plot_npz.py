#!/usr/bin/env python

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path_input_list = []
    var1_list = []
    var2_list = []

    path_anchor = "/bigwork/dscheier/results/npz_load_21_10/"
    path_output = Path(path_anchor, "all.png")
    path_output_eps = Path(path_anchor, "all_eps.png")
    # suffix_list = ["70_75_80"]
    # suffix_list = ["65", "675", "70", "725", "75", "775", "80", "825", "85", "85_long"]
    suffix_list = ["65", "675", "70", "725", "75", "775", "80", "825", "85"]
    for suffix in suffix_list:
        filename_anchor = f"a12_"
        filename = f"{filename_anchor}{suffix}.npz"
        path_input_list.append(Path(f"{path_anchor}{filename}"))

    func = lambda x, y: (x[1:], np.abs(np.diff(y)) / np.abs(y[1:]))

    var1_list.append(np.arange(65.0, 65.1, 1.0)) # a11 65
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 65

    var1_list.append(np.arange(67.5, 67.6, 1.0)) # a11 675
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 675

    var1_list.append(np.arange(70.0, 70.1, 1.0)) # a11 70
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 70

    var1_list.append(np.arange(72.5, 72.6, 1.0)) # a11 72.5
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 72.5

    var1_list.append(np.arange(75.0, 75.1, 1.0)) # a11 75
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 75

    var1_list.append(np.arange(77.5, 77.6, 1.0)) # a11 77.5
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 77.5

    var1_list.append(np.arange(80.0, 80.1, 1.0)) # a11 80
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 80

    var1_list.append(np.arange(82.5, 82.6, 1.0)) # a11 825
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 825

    var1_list.append(np.arange(85.0, 85.1, 1.0)) # a11 85
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 85

    # var1_list.append(np.arange(85.0, 85.1, 1.0)) # a11 85_long
    # var2_list.append(np.arange(2.2, 5.01, 0.2)) # tilt 85_long

    axis = 0
    lambda_frame = -1
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex='col', figsize=(16,9))
    fig_eps, axes_eps = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex='col', figsize=(16,9))

    for k, (path_property_npz, suffix_list) in enumerate(zip(path_input_list, suffix_list)):
        with open(path_property_npz, "rb") as f:
            # t = np.load(file=f)["t"]
            data = np.load(file=f)
            # mesh_t = data["x"]
            mesh_property_all = data["y"]
            # path_mesh = data["z"]
        for iy, var1 in enumerate(var1_list[k]):
            comp1 = mesh_property_all[:, iy, 0, axis, lambda_frame]
            comp2 = mesh_property_all[:, iy, 1, axis, lambda_frame]
            COM = np.abs(comp2 - comp1) / 2.0
            x_range, y_range = var2_list[k], COM
            axes[0,0].plot(x_range, y_range, "x-", label=str(var1))
            y_range = COM - x_range
            axes_eps[0,0].plot(x_range, y_range, "x-", label=str(var1))
            np.savetxt(path_property_npz.with_suffix(".csv"), np.vstack((x_range,y_range)).T, delimiter=" ")


    axes[0,0].set_ylabel("COM", rotation=0, size='large')
    axes[0,0].set_xlabel(rf"$\epsilon$ tilt")
    axes[0,0].grid()
    axes[0,0].legend()
    axes[0,0].set_title(f"lambda:")
    print(f"Save to {path_output}")
    fig.savefig(path_output)


    axes_eps[0,0].set_ylabel(r"COM - $\epsilon$", rotation=90, size='large')
    axes_eps[0,0].set_xlabel(rf"$\epsilon$ tilt")
    axes_eps[0,0].grid()
    axes_eps[0,0].legend()
    axes_eps[0,0].set_title(f"lambda:")
    fig_eps.savefig(path_output_eps)
    print(f"Save to {path_output_eps}")
