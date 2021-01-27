#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Base class for Animations

"""
from typing import Optional, Callable

from supersolids import functions


class Animation:
    def __init__(self,
                 Res: functions.Resolution,
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

        Parameters

        plot_psi_sol : bool
            Condition if :math:`\psi_{sol}` should be plotted.

        plot_V : bool
            Condition if V should be plotted.

        alpha_psi : float
            Alpha value for plot transparency of :math:`\psi`

        alpha_psi_sol : float
            Alpha value for plot transparency of :math:`\psi_{sol}`

        alpha_V : float
            Alpha value for plot transparency of V

        camera_r_func : Callable, function
            r component of the movement of the camera.

        camera_phi_func : Callable, function
            phi component of the movement of the camera.

        camera_z_func : Callable, function
            z component of the movement of the camera.

        filename : str
            Filename with filetype to save the movie to

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
