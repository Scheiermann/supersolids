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

from mayavi import mlab

from supersolids import Animation, functions

@mlab.animate(delay=10, ui=True)
def anim(x, y, func, R=3):
    prob_2d = np.abs(func(x, y)) ** 2
    p = mlab.surf(x, y, prob_2d, representation="wireframe", extent=[-R, R, -R, R, 0.0, 1.0])
    end = 100
    x_step = 0.1

    for i in range(0, end, 1):
        x_0 = x_step*i
        sc = 1.01 ** i * np.abs(func(x, y, x_0=x_0)) ** 2

        p.mlab_source.trait_set(scalars=sc)
        yield


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    L = 10
    R = 8

    psi_0_2d = functools.partial(functions.psi_gauss_2d, a=1, x_0=0.0, y_0=0.0, k_0=0.0)

    x, y = np.mgrid[-L:L:complex(0, resolution), -L:L:complex(0, resolution)]

    fig = mlab.figure()
    fig.scene.disable_render = False
    # anti_aliasing default is 8, and removes resolution issues when downscaling, but takes longer
    fig.scene.anti_aliasing_frames = 8
    fig.scene.movie_maker.directory = "."
    fig.scene.movie_maker.record = True
    anim = anim(x, y, psi_0_2d, R=R)

    # requires either mencoder or ffmpeg to be installed on your system
    # from command line
    # ffmpeg -f image2 -r 10 -i anim%05d.png -qscale 0 anim.mp4 -pass 2

    ax = mlab.axes(line_width=2, nb_labels=5)
    mlab.title("")
    mlab.show()

