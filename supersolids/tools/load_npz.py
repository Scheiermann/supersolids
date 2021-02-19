#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D in single-core.

"""

from pathlib import Path

from mayavi import mlab

from supersolids.Animation.Animation import Animation

from supersolids.Animation import MayaviAnimation


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    dir_path = Path.home().joinpath("supersolids", "results")

    Anim: Animation = Animation(plot_psi_sol=False,
                                plot_V=False,
                                alpha_psi=0.8,
                                alpha_psi_sol=0.5,
                                alpha_V=0.3,
                                filename="anim.mp4",
                                )

    # mayavi for 3D
    MayAnim = MayaviAnimation.MayaviAnimation(Anim,
                                              dir_path=dir_path,
                                              )

    MayAnimator = MayAnim.animate_npz()
    mlab.show()

    result_path = MayAnim.create_movie(dir_path=MayAnim.dir_path,
                                       input_data_file_pattern="*.png",
                                       delete_input=True)
