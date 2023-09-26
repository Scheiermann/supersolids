#!/usr/bin/env python
from pathlib import Path
import traceback
from typing import List

from moviepy.editor import VideoFileClip, clips_array

import numpy as np
from PIL import Image


def paste_together(path_in_list: List[Path], path_out: Path, nrow: int, ncol: int, ratio=None):
    fs = []
    for i, path_in in enumerate(path_in_list):
        size = get_first_im_size(path_in_list, ratio)
        if path_in is None:
            im = Image.new("RGB", size)
        else:
            try:
                im = Image.open(path_in, 'r')
            except Exception as e:
                im = Image.new("RGB", size)
                traceback.print_tb(e.__traceback__)

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

def paste_together_videos(path_mesh, path_out_video: Path,
                          margin: int = 10, width: int = 1920, height: int = 1200, fps=5):
    shape = np.shape(path_mesh)
    video_list = [VideoFileClip(str(path)).margin(margin) for path in path_mesh.ravel()]
    video_array = np.reshape(np.asarray(video_list), shape)
    # clip1 = VideoFileClip("merged.mp4").margin(10) # add 10px contour
    # path_mesh_videos_borders = map(lambda obj: obj.margin(margin), video_mesh)
    video_array2 = list(video_array)
    final_clip = clips_array(video_array2)
    final_clip.resize(width=width).write_videofile(str(path_out_video), fps=fps)
 

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