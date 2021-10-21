#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation.

"""

from typing import Callable, Optional

from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution


class SchroedingerSummary:
    """
    Saves the properties of a Schroedinger system without the arrays,
    to save disk space when saving it with dill (pickle).
    """

    def __init__(self, System) -> None:
        """
        Schrödinger equations for the specified system.

        """
        assert isinstance(System.Res, Resolution), (f"box: {type(System.Res)} "
                                                    f"is not type {type(Resolution)}")
        assert isinstance(System.Box, Box), (f"box: {type(System.Box)} is not type {type(Box)}")

        self.name: str = System.name
        self.t: float = System.t
        self.N: int = System.N
        self.Box: Box = System.Box
        self.Res: Resolution = System.Res
        self.max_timesteps: int = System.max_timesteps
        self.dt: float = System.dt
        self.dt_func: Optional[Callable] = System.dt_func
        self.g: float = System.g
        self.g_qf: float = System.g_qf
        self.w_x: float = System.w_x
        self.w_y: float = System.w_y
        self.w_z: float = System.w_z
        self.a_s: float = System.a_s
        self.e_dd: float = System.e_dd
        self.imag_time: bool = System.imag_time
        self.dim: int = System.dim
        self.mu_arr: float = System.mu_arr
        self.E: float = System.E
        self.psi_0: Callable = System.psi_0
        self.V: Callable = System.V
        self.V_interaction: Callable = System.V_interaction
        self.psi_sol: Optional[Callable] = System.psi_sol
        self.mu_sol: Optional[Callable] = System.mu_sol

    def copy_to(self, System):
        System.name: str = self.name
        System.t: float = self.t
        System.N: int = self.N
        System.Box: Box = self.Box
        System.Res: Resolution = self.Res
        System.max_timesteps: int = self.max_timesteps
        System.dt: float = self.dt
        System.dt_func: Optional[Callable] = self.dt_func
        System.g: float = self.g
        System.g_qf: float = self.g_qf
        System.w_x: float = self.w_x
        System.w_y: float = self.w_y
        System.w_z: float = self.w_z
        System.a_s: float = self.a_s
        System.e_dd: float = self.e_dd
        System.imag_time: bool = self.imag_time
        System.dim: int = self.dim
        System.mu_arr: float = self.mu_arr
        System.E: float = self.E
        System.psi_0: Callable = self.psi_0
        System.V: Callable = self.V
        System.V_interaction: Callable = self.V_interaction
        System.psi_sol: Optional[Callable] = self.psi_sol
        System.mu_sol: Optional[Callable] = self.mu_sol
