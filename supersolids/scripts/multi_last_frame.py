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

    # experiment_suffix = "ramp_11_10_85_long"
    # var1_list.append(np.arange(85.0, 85.6, 5.0)) # a11 21_10_long
    # var2_list.append(np.arange(2.2, 5.01, 0.2)) # tilt 21_10
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list = [1]
    # movie_end_list = [15]

    experiment_suffix = "stacked_05_10_a11"
    var1_list.append(np.arange(80.0, 110.1, 10.0)) # a11 stack
    var2_list.append(np.arange(0.0, 30.1, 5.0)) # stack
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    movie_start_list = [1]
    movie_end_list = [28]

    # movie_start_list = [1, 1, 1, 1, 1]
    # movie_end_list = [11, 6, 11, 6, 30]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = [f"_map_z_0", f"_map_z_1"]
    # movie_take_last_list: int = [4, 3]
    movie_take_last_list: int = [2, 1]
    suffix_list = [f"_map_x_0", f"_map_x_1"]
    # property_filename_list = []
    # list_of_arrays_list = []
    property_filename_list = ["E_paper_framestart_0.png",
                              "get_E_explicit_paper_framestart_0.png", 
                              "mu_arr_paper_framestart_0.png",
                              "get_center_of_mass_paper_framestart_0.png",
                              "get_parity_paper_framestart_0.png",
                              ]
    # list_of_arrays_list = [False]
    # list_of_arrays_list = [False, False, False]
    # list_of_arrays_list = [False, False, False, True]
    list_of_arrays_list = [False, False, False, True, True]
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
        frame_start = 0
        # frame_step = 1
        # frame_end = 101
        frame_step = 10000
        frame_end = 75001
        # frame_start = 1000
        # frame_step = 1000
        # frame_end = 100001
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