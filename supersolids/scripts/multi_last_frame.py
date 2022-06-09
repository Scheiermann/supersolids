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

    experiment_suffix = "mixture_a12_grid"
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    var1_list.append(np.arange(60.0, 81.0, 5.0))
    var2_list.append(np.arange(0.05, 0.51, 0.05))
    movie_start_list = [1]
    movie_end_list = [50]
    # suffix_list = ["_map_xyz_p-9"]
    # cut_names: List[str] = ["cut_x", "cut_y", "cut_z"]
    suffix_list = ["_map_x"]
    cut_names: List[str] = ["cut_x"]
    normed_plots = True
    # normed_plots = False
    if normed_plots:
        suffix_list[0] += "_normed"

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
    # suffix_list = ["_x"]
    # cut_names: List[str] = ["cut_x"]
    # normed_plots = True
    # if normed_plots:
    #     suffix_list[0] += "_normed"

    nrow_components = 1
    ncol_components = 1
    # ncol_components = 2
    
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

    movie_take_last_list: int = [1]
    # suffix_list = [""]
    # movie_take_last_list: int = [2, 1]
    # suffix_list = ["_0", "_1"]
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
    mixture_slice_index_list: List[int] = [0, 1, 0]
    mesh_remap_index_list: List[int] = []
    filename_steps_list: List[str] = ["mixture_step_", "mixture_step_", "mixture_mixture_step_pol_"]

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
                   )