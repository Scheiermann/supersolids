#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation (eGPE) for dipolar mixtures.


"""
from copy import deepcopy
import functools
import pickle
import sys

import dill
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, List, Tuple

from scipy import ndimage
from scipy.interpolate import interpolate
from scipy.integrate import quad_vec
from scipy.ndimage import distance_transform_edt

from supersolids.helper import constants, functions, get_path, get_version

__GPU_OFF_ENV__, __GPU_INDEX_ENV__ = get_version.get_env_variables()
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np,
                                                               gpu_off=__GPU_OFF_ENV__,
                                                               gpu_index=__GPU_INDEX_ENV__)
import supersolids.helper.numbas as numbas

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.SchroedingerMixtureSummary import SchroedingerMixtureSummary
from supersolids.helper import constants, functions
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution
from supersolids.helper.run_time import run_time

class SchroedingerMixtureNumpy(SchroedingerMixture):
    """
    SchroedingerMixture Object with all array as numpy arrays

    """
    def __init__(self, SystemMixture) -> None:
        self.name: str = "SchroedingerMixtureNumpy_"
        self.t: float = SystemMixture.t
        self.Box: Box = SystemMixture.Box
        self.Res: Resolution = SystemMixture.Res
        self.max_timesteps: int = SystemMixture.max_timesteps
        self.dt: float = SystemMixture.dt
        self.dt_func: Optional[Callable] = SystemMixture.dt_func
        self.N_list: List[float] = SystemMixture.N_list
        self.m_list: List[float] = SystemMixture.m_list
        self.a_s_factor: float = SystemMixture.a_s_factor
        self.a_dd_factor: float = SystemMixture.a_dd_factor
        self.lhy_factor: float = SystemMixture.lhy_factor
        self.a_s_array: np.ndarray = SystemMixture.a_s_array
        self.a_dd_array: np.ndarray = SystemMixture.a_dd_array
        self.w_x: float = SystemMixture.w_x
        self.w_y: float = SystemMixture.w_y
        self.w_z: float = SystemMixture.w_z
        self.imag_time: bool = SystemMixture.imag_time
        self.dim: int = SystemMixture.dim
        self.mu_arr: float = SystemMixture.mu_arr
        self.E: float = SystemMixture.E
        self.nA_max: int = SystemMixture.nA_max
        self.V: Callable = SystemMixture.V
        self.V_interaction: Callable = SystemMixture.V_interaction
        self.psi_0_list: List[np.ndarray] = SystemMixture.psi_0_list
        self.psi_sol_list: List[Optional[Callable]] = SystemMixture.psi_sol_list
        self.mu_sol_list: List[Optional[Callable]] = SystemMixture.mu_sol_list
        self.input_path: Path = SystemMixture.input_path
        self.monopolar: Optional[float] = None

        self.psi_val_list: List[cp.ndarray] = SystemMixture.psi_val_list
        self.psi_sol_val_list: List[cp.ndarray] = SystemMixture.psi_sol_val_list
        self.mu_sol_val_list: List[float] = SystemMixture.mu_sol_val_list

        self.input_path: Path = SystemMixture.input_path
        self.V: Callable = SystemMixture.V
        self.V_interaction: Callable = SystemMixture.V_interaction
        self.k_squared: np.ndarray = SystemMixture.k_squared

        self.U: complex = SystemMixture.U
        self.mu_lhy_interpolation_list = SystemMixture.mu_lhy_interpolation_list
        self.energy_helper_function = SystemMixture.energy_helper_function

        self.x, self.dx, self.kx, self.dkx = functions.get_grid_helper(self.Res, self.Box, 0)
        if self.dim >= 2:
            self.y, self.dy, self.ky, self.dky = functions.get_grid_helper(self.Res, self.Box, 1)
        if self.dim >= 3:
            self.z, self.dz, self.kz, self.dkz = functions.get_grid_helper(self.Res, self.Box, 2)

        self.V_k_val: np.ndarray = SystemMixture.V_k_val
        if self.dim == 2:
            self.x_mesh = SystemMixture.x_mesh
            self.y_mesh = SystemMixture.y_mesh
            self.pos = SystemMixture.pos 
        elif self.dim == 3:
            self.x_mesh = SystemMixture.x_mesh
            self.y_mesh = SystemMixture.y_mesh
            self.z_mesh = SystemMixture.z_mesh
            self.kz_mesh = SystemMixture.kz_mesh

        self.V_val: np.ndarray = SystemMixture.V
        self.A: np.ndarray = SystemMixture.A
        self.H_kin: np.ndarray = SystemMixture.H_kin
        self.H_kin_list: List[np.ndarray] = SystemMixture.H_kin_list
