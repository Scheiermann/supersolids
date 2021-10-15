#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Base class for Animations

"""
from typing import Optional, Callable, List

from supersolids.helper.Resolution import Resolution


class Animation:
    def __init__(self,
                 Res: Resolution = Resolution(x=2 ** 8, y=2 ** 7, z=2 ** 5),
                 plot_V: bool = False,
                 alpha_psi_list: List[float] = [],
                 alpha_psi_sol_list: List[float] = [],
                 alpha_V: float = 0.3,
                 camera_r_func: Optional[Callable] = None,
                 camera_phi_func: Optional[Callable] = None,
                 camera_z_func: Optional[Callable] = None,
                 filename: str = "split.mp4",
                 ):
        """
        Base class with configured properties for the animation.

        :param Res: functions.Res
            Number of grid points in x, y, z direction.
            Needs to have half size of box dictionary.
            Keywords x, y, z are used.

        :param plot_psi_sol: Condition if :math:`\psi_{sol}` should be plotted.

        :param plot_V: Condition if V should be plotted.

        :param alpha_psi_list: Alpha value for plot transparency of :math:`\psi`

        :param alpha_psi_sol_list: Alpha value for plot transparency of :math:`\psi_{sol}`

        :param alpha_V: Alpha value for plot transparency of V

        :param camera_r_func: r component of the movement of the camera.

        :param camera_phi_func: phi component of the movement of the camera.

        :param camera_z_func: z component of the movement of the camera.

        :param filename: Filename with filetype to save the movie to

        """
        if not alpha_psi_list:
            alpha_psi_list = [0.8]
        if not alpha_psi_sol_list:
            alpha_psi_sol_list = [0.53]

        self.Res: Resolution = Res
        self.dim: int = Res.dim
        self.plot_V: bool = plot_V

        self.alpha_psi_list: List[float] = alpha_psi_list
        self.alpha_psi_sol_list: List[float] = alpha_psi_sol_list
        self.alpha_V: float = alpha_V

        self.camera_r_func: Optional[Callable] = camera_r_func
        self.camera_phi_func: Optional[Callable] = camera_phi_func
        self.camera_z_func: Optional[Callable] = camera_z_func

        self.filename: str = filename
