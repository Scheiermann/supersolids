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
    movie_start_list = []
    movie_end_list = []

    use_edited = False

    # experiment_suffix = "ramp_13_09_10**eps"
    # experiment_suffix = "ramp_21_09_a12=70"
    # experiment_suffix = "ramp_test00"
    # experiment_suffix = "ramp_05_10"
    # experiment_suffix = "ramp_05_10_a12=70"
    # experiment_suffix = "ramp_11_10_65"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(65.0, 65.1, 1.0)) # a11 65
    # var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 65
    # movie_start_list = [1]
    # movie_end_list = [11]
    # experiment_suffix = "ramp_11_10_775"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(77.5, 77.6, 1.0)) # a11 77.5
    # var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 77.5
    # movie_start_list = [1]
    # movie_end_list = [11]
    # experiment_suffix = "ramp_11_10_85"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(85.0, 85.1, 1.0)) # a11 85
    # var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 85
    # movie_start_list = [1]
    # movie_end_list = [11]
    # experiment_suffix = "ramp_10_10_small"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(70.0, 70.1, 1.0)) # a11 10_10_small
    # var2_list.append(np.arange(0.0, 1.01, 0.2)) # tilt 10_10_small
    # movie_start_list = [1]
    # movie_end_list = [6]
    # experiment_suffix = "ramp_10_10"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # var1_list.append(np.arange(72.5, 82.6, 2.5)) # a11 10_10
    # var2_list.append(np.arange(0.0, 1.01, 0.2)) # tilt 10_10
    # movie_start_list = [1]
    # movie_end_list = [30]
    # experiment_suffix = "ramp_21_10"
    # var1_list.append(np.arange(70.0, 80.1, 5.0)) # a11 21_10
    # experiment_suffix = "ramp_21_10_675"
    # var1_list.append(np.arange(67.5, 67.6, 5.0)) # a11 21_10_675
    # experiment_suffix = "ramp_21_10_725"
    # var1_list.append(np.arange(72.5, 72.6, 5.0)) # a11 21_10_725
    # experiment_suffix = "ramp_21_10_825"
    # var1_list.append(np.arange(82.5, 82.6, 5.0)) # a11 21_10_825
    # var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 21_10

    # experiment_suffix = "ramp_21_10_90"
    # var1_list.append(np.arange(90.0, 90.1, 5.0)) # a11 24_10_90
    # var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 24_10_90
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [11]

    # experiment_suffix = "ramp_21_10_925"
    # var1_list.append(np.arange(92.5, 92.6, 5.0)) # a11 24_10_925
    # var2_list.append(np.arange(0.0, 2.01, 0.2)) # tilt 24_10_925
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [11]

    # experiment_suffix = "ramp_28_10_65_long"
    # var1_list.append(np.arange(65.0, 65.1, 5.0)) # a11 24_10_65_long
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 24_10_65_long
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_28_10_85_long"
    # var1_list.append(np.arange(85.0, 85.1, 5.0)) # a11 24_10_85_long
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 24_10_85_long
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_11_01_65_long_wide"
    # var1_list.append(np.arange(85.0, 85.1, 5.0)) # a11 24_10_85_long
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 24_10_85_long
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_11_01_85_long_wide"
    # var1_list.append(np.arange(85.0, 85.1, 5.0)) # a11 24_10_85_long
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 24_10_85_long
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_11_04_675_long_wide"
    # var1_list.append(np.arange(67.5, 67.6, 5.0)) # a11 11_04_675_long_wide
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 11_04_675_long_wide
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_11_04_70_long_wide"
    # var1_list.append(np.arange(70.0, 70.1, 5.0)) # a11 11_04_70_long_wide
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 11_04_70_long_wide
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_11_04_725_long_wide"
    # var1_list.append(np.arange(72.5, 72.6, 5.0)) # a11 11_04_725_long_wide
    # var2_list.append(np.arange(3.2, 7.01, 0.2)) # tilt 11_04_725_long_wide
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [6]
    # movie_end_list = [25]

    # experiment_suffix = "ramp_11_04_75_775_80_long_wide"
    # var1_list.append(np.arange(75.0, 80.1, 2.5)) # a11 11_04_75_775_80_long_wide
    # var2_list.append(np.arange(3.2, 7.01, 0.2)) # tilt 11_04_75_775_80_long_wide
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [60]

    # # experiment_suffix = "gpu_11_18"
    # experiment_suffix = "gpu_11_18_real_w-1"
    # # var1_list.append(np.arange(62.5, 97.6, 2.5)) # a12 gpu_11_18 
    # # var2_list.append(np.arange(-1.0, -0.95, 0.1)) # omega_epsilon gpu_11_18
    # var2_list.append(np.arange(62.5, 97.6, 2.5)) # a12 gpu_11_18 
    # var1_list.append(np.arange(-1.0, -0.95, 0.1)) # omega_epsilon gpu_11_18
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [111]
    # movie_end_list = [125]

    # experiment_suffix = "gpu_12_05"
    # var2_list.append(np.arange(62.5, 97.6, 2.5)) # a_12 gpu_12_05
    # var1_list.append(np.arange(-1.0, -0.95, 0.1)) # omega_epsilon gpu_12_06
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # # movie_start_list = [21]
    # # movie_end_list = [35]
    # movie_start_list.append(21)
    # movie_end_list.append(35)

    # experiment_suffix = "gpu_12_06"
    # a12_array = np.array([57.5, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5, 62.0, 63.0, 63.5, 64.0, 64.5, 65.5, 66.0, 66.5, 67.0, 68.0, 68.5, 69.0, 69.5])
    # var2_list.append(a12_array) # a12 gpu_12_06 
    # var1_list.append(np.arange(-1.0, -0.95, 0.1)) # omega_epsilon gpu_12_06
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # # movie_start_list = [31]
    # # movie_end_list = [52]
    # movie_start_list.append(31)
    # movie_end_list.append(52)

    # experiment_suffix = "gpu_11_18"
    experiment_suffix = "gpu_11_18_real_w-1"
    var2_list.append(np.arange(62.5, 97.6, 2.5)) # a12 gpu_11_18 
    var1_list.append(np.arange(-1.0, -0.95, 0.1)) # omega_epsilon gpu_11_18
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    movie_start_list.append(111)
    movie_end_list.append(125)


    experiment_suffix = "gpu_12_07"
    a12_array = np.array([57.5, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5, 62.0, 63.0, 63.5, 64.0, 64.5, 65.5, 66.0, 66.5, 67.0, 68.0, 68.5, 69.0, 69.5])
    var2_list.append(a12_array) # a12 gpu_12_06 
    var1_list.append(np.arange(-1.0, -0.95, 0.1)) # omega_epsilon gpu_12_06
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    movie_start_list.append(31)
    movie_end_list.append(52)

    experiment_suffix = "gpu_dipol_9_10"

    # experiment_suffix = "ramp_24_10_long"
    # var1_list.append(np.arange(72.5, 80.1, 2.5)) # a11 24_10_long
    # var2_list.append(np.arange(2.2, 7.01, 0.2)) # tilt 24_10_long
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [100]

    # experiment_suffix = "ramp_11_10_85_long"
    # var1_list.append(np.arange(85.0, 85.6, 5.0)) # a11 21_10_long
    # var2_list.append(np.arange(2.2, 5.01, 0.2)) # tilt 21_10
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [15]

    # experiment_suffix = "stacked_05_10_a11"
    # var1_list.append(np.arange(80.0, 110.1, 10.0)) # a11 stack
    # var2_list.append(np.arange(0.0, 30.1, 5.0)) # stack
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [28]

    # movie_start_list = [1, 1, 1, 1, 1]
    # movie_end_list = [11, 6, 11, 6, 30]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_z_0", f"_map_z_1"]
    # movie_take_last_list: int = [4, 3]
    movie_take_last_list: int = [2, 1]
    suffix_list = [f"_map_x_0", f"_map_x_1"]
    # movie_take_last_list: int = [1]
    # suffix_list = [f"_map_x_0"]
    # property_filename_list = []
    # list_of_arrays_list = []

    # property_filename_list = ["E_paper_framestart_0.png",
    #                           "get_E_explicit_paper_framestart_0.png", 
    #                           "mu_arr_paper_framestart_0.png",
    #                           "get_center_of_mass_paper_framestart_0.png",
    #                           "get_parity_paper_framestart_0.png",
    #                           ]
    # list_of_arrays_list = [False, False, False, True, True]

    # property_filename_list = ["E_paper_framestart_350000.png",
    #                           "get_E_explicit_paper_framestart_350000.png", 
    #                           "mu_arr_paper_framestart_350000.png",
    #                           "get_center_of_mass_paper_framestart_350000.png",
    #                           "get_parity_paper_framestart_350000.png",
    #                           ]
    # list_of_arrays_list = [False, False, False, True, True]

    # property_filename_list = ["E_paper_framestart_350000.png",
    #                           "get_E_explicit_paper_framestart_350000.png", 
    #                           "mu_arr_paper_framestart_350000.png",
    #                           "get_center_of_mass_paper_framestart_350000.png",
    #                           "get_parity_paper_framestart_350000.png",
    #                           "monopolar_paper_framestart_350000.png", 
    #                           "monopolar_paper_framestart_350000_fft.png", 
    #                           ]
    # list_of_arrays_list = [False, False, False, True, True, True, True]

    # property_filename_list = ["monopolar_paper_framestart_350000.png", 
    #                           "monopolar_paper_framestart_350000_fft.png", 
    #                           ]
    # list_of_arrays_list = [True, True]

    property_filename_list = ["monopolar_paper_framestart_350000_fft.png"]
    list_of_arrays_list = [True]

    # list_of_arrays_list = [False]
    # list_of_arrays_list = [False, False, False]
    # list_of_arrays_list = [False, False, False, True]
    # list_of_arrays_list = [False, False, False, True, True]
    cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    normed_plots = False
    if normed_plots:
        suffix_list[0] += "_normed"

    # nrow_components = 1
    # ncol_components = 2
    nrow_components = 2
    ncol_components = 1
    
    frames = False
    # frames = True
    if frames:
        # frame_start = 0
        frame_start = 350000
        frame_step = 10000
        frame_end = 350001
        # frame_end = 1350001
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
    # check_further_list = [2, 2, 2]
    check_further_list = [0, 0, 0]

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

    # video = True
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