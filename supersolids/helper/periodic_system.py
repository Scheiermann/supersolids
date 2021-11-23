#!/usr/bin/env python
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def paste_together(path_in_list: List[Path], path_out: Path, nrow: int, ncol: int, ratio=None):
    fs = []
    for i, path_in in enumerate(path_in_list):
        if path_in is None:
            size = get_first_im_size(path_in_list, ratio)
            im = Image.new("RGB", size)
        else:
            im = Image.open(path_in, 'r')
            if ratio:
                size = (int(ratio * size) for size in im.size)
                im = im.resize(size, Image.ANTIALIAS)

        fs.append(im)

    x, y = fs[0].size
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * (i % ncol), y * int(i / ncol)
        cvs.paste(fs[i], (px, py))

    cvs.save(path_out, format='png')


def get_first_im_size(path_in_list, ratio):
    for i, path_in in enumerate(path_in_list):
        try:
            im = Image.open(path_in, 'r')
            if ratio:
                size = tuple(int(ratio * size) for size in im.size)
                break
            else:
                size = im.size
                break
        except:
            continue

    return size


def periodic_system(path_in_list: List[Path], path_out: Path, nrow: int, ncol: int):
    path_in_arr: np.ndarray[Path] = np.array([path_in_list]).reshape((nrow, ncol))
    path_in_list_mirrored: List[Path] = np.ravel(np.flip(path_in_arr, axis=0))
    paste_together(path_in_list_mirrored, path_out, nrow, ncol)