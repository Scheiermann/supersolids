#!/usr/bin/env python

"""
Functions for Potential and initial wave function psi_0

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import functools
import numpy as np
from pathlib import Path

import ffmpeg
from mayavi import mlab

from supersolids import Animation, functions


def get_image_path(dir_path: Path, dir_name: str = "movie",
                   counting_format: str = "%03d"):
    # "movie" and "%03d" strings are hardcoded
    # in mayavi movie_maker _update_subdir
    existing = sorted([x for x in dir_path.glob(dir_name + "*") if x.is_dir()])
    last_index = int(existing[-1].name.split(dir_name)[1])
    input_path = Path(dir_path, dir_name + counting_format % last_index)

    return input_path


@mlab.animate(delay=10, ui=True)
def anim(x, y, z, func, R=3):
    prob_3d = np.abs(func(x, y, z)) ** 2
    p = mlab.contour3d(
        x,
        y,
        z,
        prob_3d,
        colormap="spectral",
        opacity=0.5,
        transparent=True)

    slice_x = mlab.volume_slice(x, y, z, prob_3d, colormap="spectral",
                                plane_orientation="x_axes",
                                slice_index=resolution // 2,
                                extent=[-R, R, -R, R, 0.0, 5.0])
    slice_y = mlab.volume_slice(x, y, z, prob_3d, colormap="spectral",
                                plane_orientation="y_axes",
                                slice_index=resolution // 2,
                                extent=[-R, R, -R, R, 0.0, 5.0])

    end = 100
    x_step = 0.1

    for i in range(0, end, 1):
        x_0 = x_step * i
        sc = 1.01 ** i * np.abs(func(x, y, z, x_0=x_0)) ** 2

        slice_x.mlab_source.trait_set(scalars=sc)
        slice_y.mlab_source.trait_set(scalars=sc)
        p.mlab_source.trait_set(scalars=sc)
        # mlab.process_ui_events()
        # time.sleep(0.1)
        yield


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # due to fft of the points the resolution needs to be 2 **
    # resolution_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    L = 10
    R = 8
    fig = mlab.figure()
    mlab.title("")
    ax = mlab.axes(line_width=2, nb_labels=5)
    ax.axes.visibility = True

    x, y, z = np.mgrid[-R:R:complex(0, resolution), -
                       R:R:complex(0, resolution), -R:R:complex(0, resolution)]
    psi_0_3d = functools.partial(
        functions.psi_gauss_3d,
        a=1,
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        k_0=0.0)

    dir_path = Path(__file__).parent
    fig.scene.disable_render = False
    # anti_aliasing default is 8, and removes resolution issues when
    # downscaling, but takes longer
    fig.scene.anti_aliasing_frames = 8
    fig.scene.movie_maker.directory = dir_path
    fig.scene.movie_maker.record = True
    anim = anim(x, y, z, psi_0_3d, R=R)

    mlab.show()

    input_path = get_image_path(dir_path)
    input_data = Path(input_path, "*.png")
    output_path = Path(input_path, "anim.mp4")

    # requires either mencoder or ffmpeg to be installed on your system
    # from command line:
    # ffmpeg -f image2 -r 10 -i anim%05d.png -qscale 0 anim.mp4 -pass 2
    ffmpeg.input(
        input_data,
        pattern_type="glob",
        framerate=25).output(
        str(output_path)).run()
