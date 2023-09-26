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
    movie_list_list = []
    property_filename_list_list = []

    # movie_start_list = []
    # movie_end_list = []

    # if simulation for movie_number was continued in dir with name movie_number + number_of_movies
    check_further_list = []


    use_edited = False

#     experiment_suffix = "gpu_2023_08_23"
#     a12_array = np.arange(60.0, 75.1, 1.0)
#     # a12_array = np.arange(40.0, 75.1, 1.0)
#     var2_list.append(a12_array) # a12
#     var1_list.append(np.arange(0.1, 0.11, 0.1)) # omega_epsilon gpu_12_06
#     path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
#     # movie_start_list.append(1)
#     # movie_end_list.append(36)
#     # movie_start_list.append(151)
#     # movie_end_list.append(186)
#     # movie_start_list.append(201)
#     # movie_end_list.append(216)
#     # movie_start_list.append(231)
#     # movie_end_list.append(246)
#     movie_start_list.append(271)
#     movie_end_list.append(286)
#     check_further_list.append(0)

    # experiment_suffix = "gpu_2023_09_04"
    # a12_array = np.arange(76.0, 90.1, 1.0)
    # var2_list.append(a12_array) # a12
    # var1_list.append(np.arange(0.1, 0.11, 0.2)) # omega_epsilon gpu_12_06
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # start, end, step = 21, 35, 1
    # movie_list_list.append(np.arange(start, end + 1, step))
    # check_further_list.append(0)
    # property_filename_list_list.append(["monopolar_paper_framestart_150000.png",
    #                                     "monopolar_paper_framestart_150000_fft.png"])
    # path_overlay = Path("/bigwork/dscheier/results/albert/BdG_albert/excitations.npz")

    # experiment_suffix = "gpu_dipol_2023_09_dip_10"

    # experiment_suffix = "gpu_2023_08_23"
    # a12_array = np.arange(60.0, 75.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 501, 548, 3
    # var1_list.append(np.arange(0.1, 0.51, 0.2)[::step]) # omega_epsilon gpu_12_06
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # # movie_start_list.append(501)
    # # movie_end_list.append(548)
    # check_further_list.append(0)
    # # property_filename_list = []
    # # property_filename_list.append(np.array(["monopolar_paper_framestart_350000.png"] * len(movie_list_list[-1])))
    # # property_filename_list.append(np.array(["monopolar_paper_framestart_350000_fft.png"] * len(movie_list_list[-1])))
    # # property_filename_list_list.append(property_filename_list)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])

    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(45.0, 75.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 211, 241, 1
    # var1_list.append(np.arange(0.1, 0.11, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # # movie_start_list.append(501)
    # # movie_end_list.append(548)
    # check_further_list.append(0)
    # # property_filename_list = []
    # # property_filename_list.append(np.array(["monopolar_paper_framestart_350000.png"] * len(movie_list_list[-1])))
    # # property_filename_list.append(np.array(["monopolar_paper_framestart_350000_fft.png"] * len(movie_list_list[-1])))
    # # property_filename_list_list.append(property_filename_list)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])

    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(45.0, 75.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 111, 141, 1
    # var1_list.append(np.arange(0.1, 0.11, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # # movie_start_list.append(501)
    # # movie_end_list.append(548)
    # check_further_list.append(0)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])

    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(40.0, 84.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 306, 350, 1
    # var1_list.append(np.arange(0.01, 0.011, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # check_further_list.append(0)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9_gamma0_01"

    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(40.0, 84.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 406, 450, 1
    # var1_list.append(np.arange(0.001, 0.0011, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # check_further_list.append(0)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9_gamma0_001"


    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(40.0, 84.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 506, 550, 1
    # var1_list.append(np.arange(0.0001, 0.0011, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # check_further_list.append(0)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])
    # path_overlay = None
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9_gamma0_0001"


    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(35.0, 84.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 601, 650, 1
    # var1_list.append(np.arange(0.00001, 0.000011, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # check_further_list.append(0)
    # property_filename_list_list.append(["monopolar_paper_framestart_350000.png",
    #                                     "monopolar_paper_framestart_350000_fft.png"])
    # path_overlay = None
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9_gamma0_00001"

    # experiment_suffix = "gpu_2023_09_04_dip9"
    # a12_array = np.arange(35.0, 106.1, 1.0)
    # var2_list.append(a12_array) # a12
    # start, end, step = 1, 72, 1
    # var1_list.append(np.arange(1.0, 1.1, 0.2))
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    # check_further_list.append(0)
    # path_overlay = None
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9_ground"

    experiment_suffix = "gpu_2023_09_25_stacked"
    a12_array = np.arange(80.0, 80.1, 1.0)
    var2_list.append(a12_array) # a12
    start, end, step = 21, 27, 1
    var1_list.append(np.array([0.1, 1.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    movie_list_list.append(np.arange(start, end + 1, step)) # noise [0.1, 0.3, 0.5], take just 0.1
    check_further_list.append(0)
    property_filename_list_list.append([])
    path_overlay = None
    experiment_suffix = "gpu_dipol_2023_09_25_stacked"

    fft_start = 1
    # fft_end = 35
    # fft_end = 50
    # fft_end = 12
    fft_end = 24
    # fft_end = 60
    # fft_end = 80
    # fft_end = 100
    # fft_end = 160
    # fft_end = 200

    # fake
    # experiment_suffix = "gpu_2023_08_23"
    # a12_array = np.arange(40.5, 75.1, 1.0)
    # var2_list.append(a12_array) # a12
    # var1_list.append(np.arange(0.1, 0.11, 0.1)) # omega_epsilon gpu_12_06
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # movie_start_list.append(101)
    # movie_end_list.append(135)
    # check_further_list.append(0)


    # experiment_suffix = "gpu_dipol_2023_08_23"
    # experiment_suffix = "gpu_dipol_2023_09_04"
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9"
    # experiment_suffix = "gpu_dipol_2023_09_04_dip9_dt0_0002"

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

    # property_filename_list = ["monopolar_paper_framestart_350000_fft.png"]
    # list_of_arrays_list = [True]


    list_of_arrays_list = [True, True]
    # property_filename_list = []
    # list_of_arrays_list = []
    
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
        frame_start = 4000
        # frame_start = 350000
        frame_step = 1000
        # frame_step = 10000
        frame_end = 8001
        # frame_end = 350001
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

    dir_name = "movie"
    counting_format = "%03d"

    # input for get_last_png_in_last_anim
    dir_name_png = "movie"
    counting_format_png = "%03d"
    filename_pattern: str = "anim"
    filename_format = "%05d"
    frame_format = "%07d"
    # filename_pattern: str = "1d_cut_"

    # y_lim: Tuple[int] = (0, 1.2)
    y_lim: Tuple[int] = (0, 0.12)
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
        # fps = 0.1
        fps = 1.0

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
                       movie_list_list,
                       # movie_start_list,
                       # movie_end_list,
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
                       # property_filename_list=property_filename_list,
                       property_filename_list_list=property_filename_list_list,
                       list_of_arrays_list=list_of_arrays_list,
                       fft_start=fft_start,
                       fft_end=fft_end,
                       path_overlay=path_overlay,
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
                   movie_list_list,
                   # movie_start_list,
                   # movie_end_list,
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
                   # property_filename_list=property_filename_list,
                   property_filename_list_list=property_filename_list_list,
                   list_of_arrays_list=list_of_arrays_list,
                   fft_start=fft_start,
                   fft_end=fft_end,
                   path_overlay=path_overlay,
                   )
