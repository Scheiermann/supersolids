#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation.

"""
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np

from supersolids.SchroedingerSummary import SchroedingerSummary
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution


class SchroedingerMixtureSummary:
    """
    Saves the properties of a Schroedinger system without the arrays,
    to save disk space when saving it with dill (pickle).
    """

    def __init__(self, SystemMixture) -> None:
        """
        Schrödinger equations for the specified system.

        """
        self.name: str = SystemMixture.name
        self.t: float = SystemMixture.t
        self.Box: Box = SystemMixture.Box
        self.Res: Resolution = SystemMixture.Res
        self.max_timesteps: int = SystemMixture.max_timesteps
        self.dt: float = SystemMixture.dt
        self.dt_func: Optional[Callable] = SystemMixture.dt_func
        self.N_list: List[float] = SystemMixture.N_list
        self.m_list: List[float] = SystemMixture.m_list
        self.g_array: np.ndarray = SystemMixture.a_s_array
        self.U_dd_factor_array: np.ndarray = SystemMixture.a_dd_array
        self.w_x: float = SystemMixture.w_x
        self.w_y: float = SystemMixture.w_y
        self.w_z: float = SystemMixture.w_z
        self.imag_time: bool = SystemMixture.imag_time
        self.dim: int = SystemMixture.dim
        self.mu_arr: float = SystemMixture.mu_arr
        self.E: float = SystemMixture.E
        self.V: Callable = SystemMixture.V
        self.V_interaction: Callable = SystemMixture.V_interaction
        self.psi_0_list: List[np.ndarray] = SystemMixture.psi_0_list
        self.psi_sol_list: List[Optional[Callable]] = SystemMixture.psi_sol_list
        self.mu_sol_list: List[Optional[Callable]] = SystemMixture.mu_sol_list
        self.input_path: Path = SystemMixture.input_path

    def copy_to(self, SystemMixture):
        SystemMixture.name: str = self.name
        SystemMixture.t: float = self.t
        SystemMixture.Box: Box = self.Box
        SystemMixture.Res: Resolution = self.Res
        SystemMixture.max_timesteps: int = self.max_timesteps
        SystemMixture.dt: float = self.dt
        SystemMixture.dt_func: Optional[Callable] = self.dt_func
        SystemMixture.N_list: List[float] = self.N_list
        SystemMixture.m_list: List[float] = self.m_list
        SystemMixture.a_s_array: np.ndarray = self.g_array
        SystemMixture.a_dd_array: np.ndarray = self.U_dd_factor_array
        SystemMixture.w_x: float = self.w_x
        SystemMixture.w_y: float = self.w_y
        SystemMixture.w_z: float = self.w_z
        SystemMixture.imag_time: bool = self.imag_time
        SystemMixture.dim: int = self.dim
        SystemMixture.mu_arr: np.ndarray = self.mu_arr
        SystemMixture.E: float = self.E
        SystemMixture.V: Callable = self.V
        SystemMixture.V_interaction: Callable = self.V_interaction
        SystemMixture.psi_0_list: List[np.ndarray] = self.psi_0_list
        SystemMixture.psi_sol_list: List[Optional[Callable]] = self.psi_sol_list
        SystemMixture.mu_sol_list: List[Optional[Callable]] = self.mu_sol_list
        SystemMixture.input_path: Path = self.input_path
