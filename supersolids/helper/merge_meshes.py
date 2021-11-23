#!/usr/bin/env python
from itertools import chain, zip_longest
from pathlib import Path

import numpy as np


def check_if_further(path_anchor_input, dir_name, counting_format, movie_number,
                     experiment_step, check_further=1):
    check_range = (np.arange(0, check_further + 1) * experiment_step).tolist()[::-1]
    for movie_number_add in check_range:
        movie_number_check = movie_number + movie_number_add
        path_movie = Path(path_anchor_input, f"{dir_name}{counting_format % movie_number_check}")
        try:
            files = sorted([x for x in path_movie.glob("*") if x.is_file()])
            if len(files) > 0:
                break
        except:
            continue

    return path_movie


def merge_mesh(x1, y1, r1, x2, y2, r2):
    """
    Divakar's second suggested solution using searchsort.
    """
    a1 = x1[0, :]
    a2 = x2[0, :]
    b1 = y1[:, 0]
    b2 = y2[:, 0]

    a = np.sort(np.unique(np.concatenate((a1, a2))))
    b = np.sort(np.unique(np.concatenate((b1, b2))))

    # Initialize o/p array
    x, y = np.meshgrid(a, b)
    r_out = np.full([len(a), len(b)], Path("."))

    ind_a1 = np.searchsorted(a, a1)
    ind_b1 = np.searchsorted(b, b1)
    r_out[np.ix_(ind_a1, ind_b1)] = r1

    ind_a2 = np.searchsorted(a, a2)
    ind_b2 = np.searchsorted(b, b2)
    r_out[np.ix_(ind_a2, ind_b2)] = r2

    return x, y, r_out


def merge_meshes(var_mesh_list, path_mesh_list, number_of_meshes):
    var_mesh_x, var_mesh_y, path_mesh = (var_mesh_list[0][0].T, var_mesh_list[0][1].T,
                                         path_mesh_list[0].T)
    for i in range(1, number_of_meshes):
        var_mesh_x, var_mesh_y, path_mesh = merge_mesh(
            var_mesh_x, var_mesh_y, path_mesh,
            var_mesh_list[i][0].T, var_mesh_list[i][1].T, path_mesh_list[i].T
        )

    path_mesh_T = path_mesh.T
    return var_mesh_x, var_mesh_y, path_mesh_T


def merge_meshes_old(path_in_list_list):
    path_in_list_tupled = list(zip_longest(*path_in_list_list))
    path_in_list_flatten = list(chain.from_iterable(path_in_list_tupled))
    path_in_list = list(filter(None, path_in_list_flatten))

    return path_in_list


def merge_paths(path_in_list_list):
    path_in_list_tupled = list(zip_longest(*path_in_list_list))
    path_in_list_flatten = list(chain.from_iterable(path_in_list_tupled))
    path_in_list = list(filter(None, path_in_list_flatten))

    return path_in_list