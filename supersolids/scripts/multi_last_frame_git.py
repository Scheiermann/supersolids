#!/usr/bin/env python

import numpy as np

from pathlib import Path
from typing import List, Tuple

from supersolids.helper.last_frame import last_frame


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    path_anchor_input_list: List[Path] = []
    var1_list = []
    var2_list = []

    use_edited = False

    # experiment_suffix = "x"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(2.0, 31.0, 2.0))
    # var2_list.append(np.arange(80000, 81000, 10000))

    # experiment_suffix = "gpu"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(12.0, 17.0, 2.0))
    # var2_list.append(np.arange(50.0, 96.0, 5.0))

    # experiment_suffix = "gpu"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var2_list.append(np.arange(12.0, 17.0, 2.0))
    # var1_list.append(np.arange(50.0, 96.0, 5.0))

    # experiment_suffix = "y_N15k"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(80.0, 97.0, 2.0))
    # var2_list.append(np.arange(150000, 151000, 10000))

    # experiment_suffix = "yN"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(50.0, 96.0, 5.0))

    # var2_list.append(np.arange(80000, 81000, 10000))
    # experiment_suffix = "xN"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(12.0, 17.0, 2.0))
    # var2_list.append(np.arange(50000, 151000, 10000))

    # var1_list.append(np.arange(2.0, 31.0, 2.0))
    # var2_list.append(np.arange(80000, 81000, 5000))

    # experiment_suffix = "droplet"
    # var1_list.append(np.arange(92.0, 101.0, 2.0))
    # var2_list.append(np.arange(55000, 81000, 5000))

    # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_19/"))
    # var1_list.append(np.arange(0.005, 0.05, 0.005))
    # var2_list.append(np.arange(0.6, 0.66, 0.05))

#     experiment_suffix = "mixture_a12_grid"
#     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
#     var1_list.append(np.arange(60.0, 81.0, 5.0))
#     var2_list.append(np.arange(0.05, 0.51, 0.05))
#     movie_start_list = [1]
#     movie_end_list = [50]
#     # suffix_list = ["_map_xyz_p-9"]
#     # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
#     # suffix_list = ["_map_rolled_x"]
#     # suffix_list = ["_map_x"]
#     movie_take_last_list: int = [3, 1]
#     suffix_list = ["_map_x_0", "_map_x_1"]
#     cut_names: List[str] = ["cut_x"]
#     normed_plots = True
#     # normed_plots = False
#     if normed_plots:
#         suffix_list[0] += "_normed"


    # experiment_suffix = "ramp_fixed_large"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(65.0, 72.0, 1.25))
    # var2_list.append(np.arange(0.0, 21.0, 5.0))
    # movie_start_list = [1]
    # movie_end_list = [30]
    # # suffix_list = ["_map_xyz_p-9"]
    # # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # # suffix_list = ["_map_rolled_x"]
    # # suffix_list = ["_map_x"]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # # normed_plots = True
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "ramp_fixed"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(65.0, 72.0, 1.25))
    # var2_list.append(np.arange(0.0, 0.9, 0.2))
    # movie_start_list = [1]
    # movie_end_list = [30]
    # # suffix_list = ["_map_xyz_p-9"]
    # # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # # suffix_list = ["_map_rolled_x"]
    # # suffix_list = ["_map_x"]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # # normed_plots = True
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "stacked_a11"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(90.0, 110.1, 5.0))
    # var2_list.append(np.arange(2.0, 4.1, 0.5))
    # movie_start_list = [1]
    # movie_end_list = [25]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "pretilt"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(85.0, 100.1, 2.5))
    # var2_list.append(np.arange(85.0, 100.1, 2.5))
    # movie_start_list = [1]
    # movie_end_list = [49]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "pretilt0.05_a11_100"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(50.0, 100.1, 10.0))
    # var2_list.append(np.arange(100.0, 100.1, 2.5))
    # movie_start_list = [1]
    # movie_end_list = [6]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "pretilt0.05"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(50.0, 80.1, 5.0))
    # var2_list.append(np.arange(50.0, 80.1, 5.0))
    # movie_start_list = [1]
    # movie_end_list = [49]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "pretilt0.25"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(50.0, 80.1, 5.0))
    # var2_list.append(np.arange(50.0, 80.1, 5.0))
    # movie_start_list = [1]
    # movie_end_list = [49]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

#     experiment_suffix = "pretilt0.05to100_test"
#     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
#     var1_list.append(np.arange(80.0, 100.1, 5.0))
#     var2_list.append(np.arange(80.0, 80.1, 5.0))
#     movie_start_list = [1]
#     movie_end_list = [5]
#     movie_take_last_list: int = [2, 1]
#     # suffix_list = ["_map_x_0", "_map_x_1"]
#     suffix_list = ["_z_map_x_0", "_z_map_x_1"]
#     property_filename_list = ["E_paper_0.png", "mu_arr_paper_0.png"]
#     cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
#     normed_plots = False
#     if normed_plots:
#         suffix_list[0] += "_normed"

    # experiment_suffix = "ramp_fixed_0_test_24_8"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # # var3_list.append(np.arange(1.0, 4.1, 1.0)) # x box
    # var1_list.append(np.arange(0.5, 0.76, 0.25)) # z box
    # var2_list.append(np.arange(0.5, 0.51, 0.2)) # y box
    # movie_start_list = [1]
    # movie_end_list = [2]
    # movie_take_last_list: int = [2]
    # suffix_list = [f"_map_x_1"]
    # # property_filename_list = []
    # # property_filename_list = ["E_paper_framestart_0.png"]
    # property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "pretilt0.05_a11_60to100"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(80.0, 100.1, 5.0)) # a11
    # var2_list.append(np.arange(60.0, 100.1, 10.0)) # a12
    # movie_start_list = [1]
    # movie_end_list = [25]
    # movie_take_last_list: int = [4, 3]
    # suffix_list = [f"_map_x_1"]
    # property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # # experiment_suffix = "pretilt0.05_a11_70to120"
    # experiment_suffix = "pretilt0.05_a11_70to120_Res4x"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var2_list.append(np.arange(80.0, 100.1, 5.0)) # a11
    # var1_list.append(np.arange(70.0, 120.1, 10.0)) # a12
    # movie_start_list = [1]
    # movie_end_list = [30]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_z_0", f"_map_z_1"]
    # property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

#     # experiment_suffix = "stacked_a11_01_09"
#     experiment_suffix = "stacked_a11_05_09"
#     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
#     var2_list.append(np.arange(80.0, 80.1, 5.0)) # a11
#     # var1_list.append(np.arange(2.0, 4.1, 0.5)) # h
#     var1_list.append(np.arange(5.0, 30.1, 5.0)) # h
#     movie_start_list = [1]
#     # movie_end_list = [5]
#     movie_end_list = [6]
#     movie_take_last_list: int = [2, 1]
#     suffix_list = [f"_map_x_0", f"_map_x_1"]
#     property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png"]
#     cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
#     normed_plots = False
#     if normed_plots:
#         suffix_list[0] += "_normed"

    # # experiment_suffix = "ramp_05_09"
    # experiment_suffix = "ramp_09_09_10**eps"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(65.0, 100.1, 5.0)) # a11
    # var2_list.append(np.arange(-1.0, 2.1, 1.0)) # tilt
    # movie_start_list = [1]
    # movie_end_list = [32]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_z_0", f"_map_z_1"]
    # # movie_take_last_list: int = [4, 3]
    # # suffix_list = [f"_map_x_0", f"_map_x_1"]
    # property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png",
    #                           "get_center_of_mass_paper_framestart_0.png", "get_parity_paper_framestart_0.png"]
    # list_of_arrays_list = [False, False, True, True]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    experiment_suffix = "ramp_13_09_10**eps"
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    var1_list.append(np.arange(70.0, 70.1, 5.0)) # a11
    var2_list.append(np.arange(3.0, 8.1, 1.0)) # tilt
    movie_start_list = [1]
    movie_end_list = [6]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_z_0", f"_map_z_1"]
    movie_take_last_list: int = [4, 3]
    suffix_list = [f"_map_x_0", f"_map_x_1"]
    property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png",
                              "get_center_of_mass_paper_framestart_0.png",
                              "get_parity_paper_framestart_0.png", "check_N_paper_framestart_0.png"]
    list_of_arrays_list = [False, False, True, True, False]
    cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    normed_plots = False
    if normed_plots:
        suffix_list[0] += "_normed"

    # experiment_suffix = "ramp_29_8"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(-1.0, 2.1, 1.0)) # tilt, 0.0, 0.1, 1.0, 10.0
    # var2_list.append(np.arange(65.0, 66.1, 1.5)) # a12=65.0
    # movie_start_list = [1]
    # movie_end_list = [4]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_x_1"]
    # property_filename_list = ["E_paper_framestart_0.png", "mu_arr_paper_framestart_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"


    # experiment_suffix = "pretilt0.05to100_test_init1"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # # var3_list.append(np.arange(1.0, 4.1, 1.0)) # x box
    # var1_list.append(np.arange(0.5, 1.8, 0.25)) # z box
    # var2_list.append(np.arange(0.5, 0.51, 0.2)) # y box
    # movie_start_list = [1]
    # movie_end_list = [6]
    # movie_take_last_list: int = [1]
    # suffix_list = [f"_map_x_1"]
    # property_filename_list = []
    # # property_filename_list = ["E_paper_framestart_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"


    # experiment_suffix = "pretilt0.05to100_test_init100"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # # var3_list.append(np.arange(1.0, 4.1, 1.0)) # x box
    # var1_list.append(np.arange(0.2, 0.81, 0.2)) # y box
    # var2_list.append(np.arange(2.0, 4.1, 1.0)) # z box

# #     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
# #     # var3_list.append(np.arange(1.0, 4.1, 1.0)) # x box
# #     var1_list.append(np.arange(0.2, 0.81, 0.2)) # y box
# #     var2_list.append(np.arange(2.0, 4.1, 1.0)) # z box

# #     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
# #     # var3_list.append(np.arange(1.0, 4.1, 1.0)) # x box
# #     var1_list.append(np.arange(0.2, 0.81, 0.2)) # y box
# #     var2_list.append(np.arange(2.0, 4.1, 1.0)) # z box
# #     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
# #     # var3_list.append(np.arange(1.0, 4.1, 1.0)) # x box
# #     var1_list.append(np.arange(0.2, 0.81, 0.2)) # y box
# #     var2_list.append(np.arange(2.0, 4.1, 1.0)) # z box

    # suf = "z2"
    # # movie_start_list = [1]
    # # movie_end_list = [12]
    # movie_start_list = [13]
    # movie_end_list = [24]
    # # movie_start_list = [25]
    # # movie_end_list = [36]
    # # movie_start_list = [37]
    # # movie_end_list = [48]
    # # movie_start_list = [1, 13, 25, 37]
    # # movie_end_list = [12, 24, 36, 48]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_x_0_{suf}", f"_map_x_1_{suf}"]
    # property_filename_list = ["E_paper_framestart_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "pretilt0.05to100"
    # # experiment_suffix = "pretilt0.25to100"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(80.0, 100.1, 5.0))
    # var2_list.append(np.arange(80.0, 100.1, 5.0))
    # movie_start_list = [1]
    # movie_end_list = [25]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_map_x_0", "_map_x_1"]
    # property_filename_list = ["E_paper_0.png", "mu_arr_paper_0.png"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # normed_plots = False
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    # experiment_suffix = "mixture_a12_small_grid"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(0.6, 0.81, 0.05))
    # var2_list.append(np.arange(0.01, 0.051, 0.005))

    # path_anchor_input_list.append(Path(f"/run/media/dsche/ITP Transfer/begin_mixture_{experiment_suffix}/"))
    # var1_list.append(np.arange(0.60, 0.726, 0.025))
    # var2_list.append(np.arange(0.01, 0.041, 0.01))
    # var1_list.append(np.arange(0.50, 0.751, 0.025))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))

    # var1_list.append(np.arange(0.76, 0.81, 0.02))
    # var2_list.append(np.arange(0.50, 0.51, 0.05))

    # var1_list.append(np.arange(0.75, 0.80, 0.001))
    # var2_list.append(np.arange(0.50, 0.51, 0.05))

    # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_13/"))
    # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_13/"))
    # var1_list.append(np.arange(0.6, 0.91, 0.05))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))

    # experiment_suffix = "a11_95"
    # # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6125/"))
    # # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_15_6125/"))
    # path_anchor_input_list.append(Path("/bigwork/dscheier/results/begin_mixture_15_6125/"))
    # var1_list.append(np.arange(61.25, 91.0, 5.0))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))

    # # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_15/"))
    # # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_15/"))
    # path_anchor_input_list.append(Path("/bigwork/dscheier/results/begin_mixture_15/"))
    # var1_list.append(np.arange(62.5, 91.0, 5.0))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))

    # # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6375/"))
    # # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_15_6375/"))
    # path_anchor_input_list.append(Path("/bigwork/dscheier/results/begin_mixture_15_6375/"))
    # var1_list.append(np.arange(63.75, 91.0, 5.0))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))
    # movie_start_list = [1, 1, 1]
    # movie_end_list = [60, 60, 60]
    # # suffix_list = ["_xyz_p-9"]
    # # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    # # suffix_list = ["_rolled_x"]
    # # movie_take_last_list: int = [1]
    # # suffix_list = ["_x"]
    # movie_take_last_list: int = [3, 1]
    # suffix_list = ["_x_0", "_x_1"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = True
    # if normed_plots:
    #     suffix_list = [suffix + "_normed" for suffix in suffix_list]

    # nrow_components = 1
    # ncol_components = 2
    nrow_components = 2
    ncol_components = 1
    
    frames = False
    # frames = True
    if frames:
        frame_start = 1000
        frame_step = 1000
        frame_end = 100001
        # frame_end = 20001
        frames = np.arange(frame_start, frame_end, frame_step)
    else:
        frames = np.array([False])

    dir_suffix_list = [f"last_frame_{experiment_suffix}" + suf for suf in suffix_list]
    filename_out_list = [f"last_frame_{experiment_suffix}" + suf for suf in suffix_list]
    file_suffix = ""
    # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"
    merge_suffix = suffix_list[0]

    movie_skip = None

    # if simulation for movie_number was continued in dir with name movie_number + number_of_movies
    # check_further_list = [1, 1, 1, 1]
    # check_further_list = [2, 2]
    # check_further_list = [2]
    # check_further_list = [2]
    check_further_list = [2, 2, 2]

    dir_name = "movie"
    counting_format = "%03d"

    # input for get_last_png_in_last_anim
    dir_name_png = "movie"
    counting_format_png = "%03d"
    filename_pattern: str = "anim"
    filename_format = "%05d"
    frame_format = "%07d"
    # filename_pattern: str = "1d_cut_"

    y_lim: Tuple[int] = (0, 1.2)
    # mixture_slice_index_list: List[int] = [0, 0, 0]
    # mesh_remap_index_list: List[int] = [1]
    mesh_remap_index_list: List[int] = []
    # filename_steps_list: List[str] = ["mixture_step_", "mixture_step_", "mixture_mixture_step_pol_"]
    # mixture_slice_index_list: List[int] = [0, 1, 0]
    # filename_steps_list: List[str] = ["step_", "step_", "pol_"]
    mixture_slice_index_list: List[int] = [0, 1]
    filename_steps_list: List[str] = ["step_", "step_"]

    video = False
    if video:
        filename_format = None
        filename_extension = ".mp4"

        margin = 10
        width = 1920
        height = 1200
        fps = 0.1

    else:
        filename_extension = ".png"
        margin = None
        width = None
        height = 1200
        fps = None

    path_graphs = Path(path_anchor_input_list[0].parent, "graphs")

    # adjust to shrink images, so RAM gets not overloaded
    dpi_ratio = 1.00
    dpi_ratio_all = 1.0

    ######## END OF USER INPUT #####################################################################
    path_anchor_output_list = [Path(path_graphs, dir_suffix) for dir_suffix in dir_suffix_list]
    # property_filename_suffix_list = [dir_suffix + file_suffix for dir_suffix in dir_suffix_list]

    # Create a results dir, if there is none
    for path_anchor_output in path_anchor_output_list:
        if not path_anchor_output.is_dir():
            path_anchor_output.mkdir(parents=True)

    if frames.any():
        for frame in frames:
            last_frame(frame,
                       var1_list,
                       var2_list,
                       experiment_suffix,
                       movie_take_last_list,
                       path_anchor_output_list,
                       suffix_list,
                       merge_suffix,
                       filename_out_list,
                       path_anchor_input_list,
                       movie_start_list,
                       movie_end_list,
                       check_further_list,
                       movie_skip,
                       dir_name,
                       counting_format, 
                       nrow_components,
                       ncol_components,
                       dpi_ratio,
                       dpi_ratio_all,
                       use_edited,
                       dir_name_png,
                       counting_format_png,
                       filename_pattern,
                       filename_format,
                       filename_extension,
                       frame_format,
                       video,
                       margin,
                       width,
                       height,
                       fps,
                       mesh_remap_index_list=mesh_remap_index_list,
                       y_lim=y_lim,
                       cut_names=cut_names,
                       mixture_slice_index_list=mixture_slice_index_list,
                       filename_steps_list=filename_steps_list,
                       normed_plots=normed_plots,
                       property_filename_list=property_filename_list,
                       list_of_arrays_list=list_of_arrays_list,
                       )
    else:
        frame = None
        last_frame(frame,
                   var1_list,
                   var2_list,
                   experiment_suffix,
                   movie_take_last_list,
                   path_anchor_output_list,
                   suffix_list,
                   merge_suffix,
                   filename_out_list,
                   path_anchor_input_list,
                   movie_start_list,
                   movie_end_list,
                   check_further_list,
                   movie_skip,
                   dir_name,
                   counting_format, 
                   nrow_components,
                   ncol_components,
                   dpi_ratio,
                   dpi_ratio_all,
                   use_edited,
                   dir_name_png,
                   counting_format_png,
                   filename_pattern,
                   filename_format,
                   filename_extension,
                   frame_format,
                   video,
                   margin,
                   width,
                   height,
                   fps,
                   mesh_remap_index_list=mesh_remap_index_list,
                   y_lim=y_lim,
                   cut_names=cut_names,
                   mixture_slice_index_list=mixture_slice_index_list,
                   filename_steps_list=filename_steps_list,
                   normed_plots=normed_plots,
                   property_filename_list=property_filename_list,
                   list_of_arrays_list=list_of_arrays_list,
                   )