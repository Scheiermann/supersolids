#!/usr/bin/env python

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path_input_list = []
    var1_list = []
    var2_list = []
    slice_first_list = []
    slice_last_list = []

    path_anchor = "/bigwork/dscheier/results/npz_load_11_01/"
    path_anchor_in = Path("/bigwork/dscheier/results/graphs/")
    # path_anchor_in = Path(path_anchor, "input")
    path_anchor_out = Path(path_anchor, "output", "all")
    # path_anchor = "/bigwork/dscheier/results/npz_load_21_10/"
    num = "005"
    path_output = Path(path_anchor_out, f"{num}_all_.png")
    path_output_eps = Path(path_anchor_out, f"{num}_all_eps.png")
    path_output_a12 = Path(path_anchor_out, f"{num}_all_a12.png")
    # suffix_list = ["725_75_775_80"]
    # suffix_list = ["65", "675", "70", "725", "75", "775", "80", "825", "85", "725_long", "75_long", "775_long", "80_long", "85_long"]
    # suffix_list = ["65", "675", "70", "725", "75", "775", "80", "825", "85"]

    filename_anchor = f"a12_"
    # for suffix in suffix_list:
    #     filename = f"{filename_anchor}{suffix}.npz"
    #     path_input_list.append(Path(f"{path_anchor}{filename}"))

    func = lambda x, y: (x[1:], np.abs(np.diff(y)) / np.abs(y[1:]))

    # path_input_list.append(Path(f"{path_anchor_in}", f"{filename}"))
    split_out_list = []

    ##### [0, 1, 0.2]

    # split_out_list = ["725", "75", "775", "80", "825"]
    folder = "last_frame_ramp_10_10_map_x_1"
    filename = "property_all_ramp_10_10_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(72.5, 82.6, 2.5)) # a11 10_10
    var2_list.append(np.arange(0.0, 1.01, 0.2)) # tilt 10_10
    slice_first_list.append(None)
    slice_last_list.append(None)

    # split_out_list = ["70"]
    folder = "last_frame_ramp_10_10_small_map_x_1"
    filename = "property_all_ramp_10_10_small_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(70.0, 70.1, 10.0)) # a11 10_10_small
    var2_list.append(np.arange(0.0, 1.01, 0.2)) # tilt 10_10_small
    slice_first_list.append(None)
    slice_last_list.append(None)

    ##### [0, 10, 2.0]

    # split_out_list = ["70"]
    folder = "last_frame_ramp_05_10_a12=70_map_x_1"
    filename = "property_all_ramp_05_10_a12=70_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(70.0, 70.1, 10.0)) # a11 05_10_a12=70
    var2_list.append(np.arange(0.0, 10.01, 2.0)) # tilt 05_10_a12=70
    slice_first_list.append(None)
    slice_last_list.append(None)

    # split_out_list = ["75", "85", "95", "105"]
    folder = "last_frame_ramp_05_10_map_x_1"
    filename = "property_all_ramp_05_10_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(75.0, 105.1, 10.0)) # a11 05_10
    var2_list.append(np.arange(0.0, 10.01, 2.0)) # tilt 05_10
    slice_first_list.append(None)
    slice_last_list.append(None)

    ##### [2.2, 7, 0.2]

    # split_out_list = ["725_long", "75_long", "775_long", "80_long"]
    folder = "last_frame_ramp_24_10_long_map_x_1"
    filename = "property_all_ramp_24_10_long_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(72.5, 80.1, 2.5)) # a11 24_10_long
    var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 24_10_long
    slice_first_list.append(None)
    slice_last_list.append(6)

    folder = "last_frame_ramp_11_01_65_long_wide_map_x_1"
    filename = "property_all_ramp_11_01_65_long_wide_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(65.0, 65.1, 1.0)) # a11 11_01_65_long_wide
    var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 11_01_65_long_wide
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_11_01_85_long_wide_map_x_1"
    filename = "property_all_ramp_11_01_85_long_wide_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(85.0, 85.1, 1.0)) # a11 11_01_85_long_wide
    var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 11_01_85_long_wide
    slice_first_list.append(None)
    slice_last_list.append(None)

    ##### [2.2, 5, 0.2]

    folder = "last_frame_ramp_11_10_85_long_map_x_1"
    filename = "property_all_ramp_11_10_85_long_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(85.0, 85.1, 1.0)) # a11 85_long
    var2_list.append(np.arange(2.2, 5.01, 0.2)) # tilt 85_long
    slice_first_list.append(None)
    slice_last_list.append(7)


    ##### [0, 2, 0.2]

    folder = "last_frame_ramp_21_10_map_x_1"
    filename = "property_all_ramp_21_10_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(70.0, 80.1, 5.0)) # a11 21_10
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 21_10
    slice_first_list.append(None)
    slice_last_list.append(None)


    ##### [0, 2, 0.2]

    folder = "last_frame_ramp_11_10_65_map_x_1"
    filename = "property_all_ramp_11_10_65_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(65.0, 65.1, 1.0)) # a11 65
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 65
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_21_10_675_map_x_1"
    filename = "property_all_ramp_21_10_675_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(67.5, 67.6, 1.0)) # a11 675
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 675
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_21_10_725_map_x_1"
    filename = "property_all_ramp_21_10_725_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(72.5, 72.6, 1.0)) # a11 72.5
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 72.5
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_11_10_775_map_x_1"
    filename = "property_all_ramp_11_10_775_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(77.5, 77.6, 1.0)) # a11 77.5
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 77.5
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_21_10_825_map_x_1"
    filename = "property_all_ramp_21_10_825_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(82.5, 82.6, 1.0)) # a11 825
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 825
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_11_10_85_map_x_1"
    filename = "property_all_ramp_11_10_85_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(85.0, 85.1, 1.0)) # a11 85
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 85
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_21_10_90_map_x_1"
    filename = "property_all_ramp_21_10_90_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(90.0, 90.1, 1.0)) # a11 90
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 90
    slice_first_list.append(None)
    slice_last_list.append(None)

    folder = "last_frame_ramp_21_10_925_map_x_1"
    filename = "property_all_ramp_21_10_925_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(92.5, 92.6, 1.0)) # a11 925
    var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 925
    slice_first_list.append(None)
    slice_last_list.append(None)

    experiment_suffix = "ramp_11_04_675_long_wide"
    folder = f"last_frame_{experiment_suffix}_map_x_1"
    filename = f"property_all_{experiment_suffix}_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(67.5, 67.6, 5.0)) # a11 11_04_675_long_wide
    var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 11_04_675_long_wide
    slice_first_list.append(None)
    slice_last_list.append(None)

    experiment_suffix = "ramp_11_04_70_long_wide"
    folder = f"last_frame_{experiment_suffix}_map_x_1"
    filename = f"property_all_{experiment_suffix}_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(70.0, 70.1, 5.0)) # a11 11_04_70_long_wide
    var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 11_04_70_long_wide
    slice_first_list.append(None)
    slice_last_list.append(None)

    experiment_suffix = "ramp_11_04_725_long_wide"
    folder = f"last_frame_{experiment_suffix}_map_x_1"
    filename = f"property_all_{experiment_suffix}_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(72.5, 72.6, 5.0)) # a11 11_04_725_long_wide
    var2_list.append(np.arange(3.2, 7.01, 0.2)) # tilt 11_04_725_long_wide
    slice_first_list.append(None)
    slice_last_list.append(None)

    experiment_suffix = "ramp_11_04_75_775_80_long_wide"
    folder = f"last_frame_{experiment_suffix}_map_x_1"
    filename = f"property_all_{experiment_suffix}_map_x_0_get_center_of_mass_paper_framestart_0.npz"
    path_input_list.append(Path(f"{path_anchor_in}", f"{folder}", f"{filename}"))
    var1_list.append(np.arange(75.0, 80.1, 2.5)) # a11 11_04_75_775_80_long_wide
    var2_list.append(np.arange(3.2, 7.01, 0.2)) # tilt 11_04_75_775_80_long_wide
    slice_first_list.append(None)
    slice_last_list.append(None)

    # Create a results dir, if there is none
    if not path_anchor_out.is_dir():
        path_anchor_out.mkdir(parents=True)
    
    good_colors = [True, False, False]
    axis = 0
    lambda_frame = -1
    COM_tilt_max_list = []
    COM_tilt_max_list_x = []
    for good_color in good_colors:
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex='col', figsize=(25,12))
        fig_eps, axes_eps = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex='col', figsize=(25,12))
        fig_a12, axes_a12 = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex='col', figsize=(25,12))

        # try to find good colors for the same var1 values
        a = np.sort(np.concatenate(var1_list).ravel())
        values, counts = np.unique(a, return_counts=True)
        colors_plasma = plt.get_cmap("plasma").colors
        color_step = int(256 / len(values))
        good_colors_exist = max(counts) <= 4 and len(values) <= 15
        if good_colors_exist and good_color:
            path_anchor_out_dir = Path(path_anchor_out, "good")
            if not path_anchor_out_dir.is_dir():
                path_anchor_out_dir.mkdir(parents=True)
            colors0 = plt.get_cmap("tab20").colors
            colors1 = plt.get_cmap("tab20b").colors
            colors2 = plt.get_cmap("tab20c").colors
            colors = colors0 + colors1 + colors2
        else:
            path_anchor_out_dir = Path(path_anchor_out, "ok")
            if not path_anchor_out_dir.is_dir():
                path_anchor_out_dir.mkdir(parents=True)
            colors = colors_plasma
            print("Too many values for good color palette!")

        for k, (path_property_npz, slice_first, slice_last) in enumerate(zip(path_input_list, slice_first_list, slice_last_list)):
            print(f"Load {path_property_npz}")
            with open(path_property_npz, "rb") as f:
                # t = np.load(file=f)["t"]
                data = np.load(file=f, allow_pickle=True)
                mesh_t = data["x"]
                mesh_property_all = data["y"]
                path_mesh = data["z"]
            for iy, var1 in enumerate(var1_list[k]):
                if split_out_list:
                    split_out_path = Path(path_anchor_out_dir, filename_anchor + split_out_list[iy] + ".npz")
                    np.savez_compressed(split_out_path, x=mesh_t, y=mesh_property_all, z=path_mesh)
                if slice_first and slice_last:
                    comp1 = mesh_property_all[:, iy, 0, axis, lambda_frame][slice_first:slice_last]
                    comp2 = mesh_property_all[:, iy, 1, axis, lambda_frame][slice_first:slice_last]
                    x_range = var2_list[k][slice_first:slice_last]
                elif slice_first:
                    comp1 = mesh_property_all[:, iy, 0, axis, lambda_frame][slice_first:]
                    comp2 = mesh_property_all[:, iy, 1, axis, lambda_frame][slice_first:]
                    x_range = var2_list[k][slice_first:]
                elif slice_last:
                    comp1 = mesh_property_all[:, iy, 0, axis, lambda_frame][:slice_last]
                    comp2 = mesh_property_all[:, iy, 1, axis, lambda_frame][:slice_last]
                    x_range = var2_list[k][:slice_last]
                else:
                    comp1 = mesh_property_all[:, iy, 0, axis, lambda_frame]
                    comp2 = mesh_property_all[:, iy, 1, axis, lambda_frame]
                    x_range = var2_list[k]

                COM = np.abs(comp2 - comp1) / 2.0
                y_range = COM
                label = f"{path_property_npz.parent.stem} {var1}"

                color_big_index = np.where(values == var1)[0][0]
                if good_colors_exist:
                    color_index = 4 * color_big_index
                else:
                    color_index = color_step * color_big_index
                axes[0,0].plot(x_range, y_range, "x-", label=label, color=colors[color_index])

                var1_str = str(var1).replace(".", "")
                path_csv = Path(path_anchor_out_dir, str(path_property_npz.stem) + f"_{var1_str}.csv")
                np.savetxt(path_csv, np.vstack((x_range,y_range)).T, delimiter=" ")

                y_range = COM - x_range
                axes_eps[0,0].plot(x_range, y_range, "x-", label=label, color=colors[color_index])
                path_special = Path(path_anchor_out_dir, str(path_csv.stem) + f"_COM_tilt_{var1_str}.csv")
                np.savetxt(path_special, np.vstack((x_range,y_range)).T, delimiter=" ")

                if max(x_range) == 2.0:
                    COM_tilt_max_list.append(np.max(y_range))
                    COM_tilt_max_list_x.append(var1)

        zipped_sorted_by_max = zip(*sorted(zip(COM_tilt_max_list_x, COM_tilt_max_list), key=lambda t: t[1]))
        x, y = map(np.array, zipped_sorted_by_max)
        axes_a12[0,0].plot(x, y, "x-")

        axes[0,0].set_ylabel("COM", rotation=0, size='large')
        axes[0,0].set_xlabel(rf"$\epsilon$ tilt")
        axes[0,0].grid()
        axes[0,0].legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
        axes[0,0].set_title(f"lambda:")
        print(f"Save to {path_output}")
        fig.tight_layout()
        fig.subplots_adjust(right=0.70)
        fig.savefig(path_output, bbox_inches='tight')

        axes_eps[0,0].set_ylabel(r"COM - $\epsilon$", rotation=90, size='large')
        axes_eps[0,0].set_xlabel(rf"$\epsilon$ tilt")
        axes_eps[0,0].grid()
        axes_eps[0,0].legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
        axes_eps[0,0].set_title(f"lambda:")
        print(f"Save to {path_output_eps}")
        fig_eps.savefig(path_output_eps, bbox_inches='tight')

        axes_a12[0,0].set_ylabel(r"max(COM - $\epsilon$)", rotation=90, size='large')
        axes_a12[0,0].set_xlabel(rf"$a_{12}$")
        axes_a12[0,0].grid()
        axes_a12[0,0].legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
        axes_a12[0,0].set_title(f"lambda:")
        print(f"Save to {path_output_a12}")
        fig_a12.tight_layout()
        fig_a12.subplots_adjust(right=0.70)
        fig_a12.savefig(path_output_a12, bbox_inches='tight')

