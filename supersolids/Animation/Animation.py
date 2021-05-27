#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Base class for Animations

"""
from typing import Optional, Callable

from supersolids.helper.Resolution import Resolution


class Animation:
    def __init__(self,
                 Res: Resolution = Resolution(x=2 ** 8, y=2 ** 7, z=2 ** 5),
                 plot_psi_sol: bool = True,
                 plot_V: bool = False,
                 alpha_psi: float = 0.8,
                 alpha_psi_sol: float = 0.53,
                 alpha_V: float = 0.3,
                 camera_r_func: Optional[Callable] = None,
                 camera_phi_func: Optional[Callable] = None,
                 camera_z_func: Optional[Callable] = None,
                 filename: str = "split.mp4",
                 ):
        """
        Base class with configured properties for the animation.

        :param Res: functoins.Res
            Number of grid points in x, y, z direction.
            Needs to have half size of box dictionary.
            Keywords x, y, z are used.

        :param plot_psi_sol: Condition if :math:`\psi_{sol}` should be plotted.

        :param plot_V: Condition if V should be plotted.

        :param alpha_psi: Alpha value for plot transparency of :math:`\psi`

        :param alpha_psi_sol: Alpha value for plot transparency of :math:`\psi_{sol}`

        :param alpha_V: Alpha value for plot transparency of V

        :param camera_r_func: r component of the movement of the camera.

        :param camera_phi_func: phi component of the movement of the camera.

        :param camera_z_func: z component of the movement of the camera.

        :param filename: Filename with filetype to save the movie to

        """
        self.Res = Res
        self.dim = Res.dim
        self.plot_psi_sol = plot_psi_sol
        self.plot_V = plot_V

        self.alpha_psi: float = alpha_psi
        self.alpha_psi_sol: float = alpha_psi_sol
        self.alpha_V: float = alpha_V

        self.camera_r_func: Optional[Callable] = camera_r_func
        self.camera_phi_func: Optional[Callable] = camera_phi_func
        self.camera_z_func: Optional[Callable] = camera_z_func

        self.filename: str = filename
