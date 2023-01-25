#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Function speeded up with numba

"""

import numba
import numpy as np

from supersolids.helper import get_version

__GPU_OFF_ENV__, __GPU_INDEX_ENV__ = get_version.get_env_variables()
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np, gpu_off=__GPU_OFF_ENV__, gpu_index=__GPU_INDEX_ENV__)


from numba import njit, types, f8, c16
from typing import Callable


@njit('c16[:,:,:](f8[:,:,:], f8, f8, c16[:,:,:], f8[:,:,:], f8[:,:,:])')
def get_H_pot_exponent_terms_jit(V_val: cp.ndarray,
                                 a_dd_factor: float,
                                 a_s_factor: float,
                                 dipol_term: cp.ndarray,
                                 contact_interaction: cp.ndarray,
                                 mu_lhy: cp.ndarray) -> cp.ndarray:
    return (V_val
            + a_dd_factor * dipol_term
            + a_s_factor * contact_interaction
            + mu_lhy
            )


@njit(cache=True)
def get_H_pot_exponent_terms_jit(V_val: cp.ndarray,
                                 a_dd_factor: float,
                                 a_s_factor: float,
                                 dipol_term: cp.ndarray,
                                 contact_interaction: cp.ndarray,
                                 mu_lhy: cp.ndarray) -> cp.ndarray:
    return (V_val
            + a_dd_factor * dipol_term
            + a_s_factor * contact_interaction
            + mu_lhy
            )


@njit('c16[:,:,:](c16, f8, c16[:,:,:], f8)')
def get_H_pot_jit(U: cp.complex, dt: float, terms: cp.ndarray,
                  split_step: float = 0.5) -> cp.ndarray:
    return np.exp(U * (split_step * dt) * terms)


@njit((c16, c16, types.Array(c16, 3, "C"), c16), cache=True)
def get_H_pot_jit(U: cp.complex, dt: float, terms: cp.ndarray,
                  split_step: float = 0.5) -> cp.ndarray:
    return np.exp(U * (split_step * dt) * terms)


@njit('f8[:](f8[:], f8, f8, f8, f8)')
def f_lam(A: cp.ndarray, lam: float,
          eta_aa: float, eta_bb: float, eta_ab: float) -> cp.ndarray:
    return (eta_aa * A
            + eta_bb * (1.0 - A)
            + lam * np.sqrt((eta_aa * A - eta_bb * (1.0 - A)) ** 2.0
                            + 4.0 * eta_ab ** 2.0 * A * (1.0 - A)))


@njit(cache=True)
def f_lam(A: cp.ndarray, lam: float,
              eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
    return (eta_aa * A
            + eta_bb * (1.0 - A)
            + lam * np.sqrt((eta_aa * A - eta_bb * (1.0 - A)) ** 2.0
                            + 4.0 * eta_ab ** 2.0 * A * (1.0 - A)))


@njit('f8[:](f8[:], f8, f8, f8, f8)')
def eta_dVdna_jit(A: cp.ndarray, lam: float,
                  eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
    term = ((eta_aa * (eta_aa * A - eta_bb * (1 - A)) + 2 * eta_ab ** 2 * (1 - A))
             / np.sqrt((eta_aa * A - eta_bb * (1 - A)) ** 2 + 4 * eta_ab ** 2 * A * (1 - A))
            )
    term_nan_free = np.where(~np.isnan(term), term, 0.0)

    return eta_aa + lam * term_nan_free


@njit(cache=True)
def eta_dVdna_jit(A: cp.ndarray, lam: float,
                  eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
    term = ((eta_aa * (eta_aa * A - eta_bb * (1 - A)) + 2 * eta_ab ** 2 * (1 - A))
             / np.sqrt((eta_aa * A - eta_bb * (1 - A)) ** 2 + 4 * eta_ab ** 2 * A * (1 - A))
            )
    term_nan_free = np.where(~np.isnan(term), term, 0.0)

    return eta_aa + lam * term_nan_free


@njit('f8[:](f8[:], f8, f8, f8, f8)')
def eta_dVdnb_jit(A: cp.ndarray, lam: float,
                  eta_aa: float, eta_bb: float, eta_ab: float) -> cp.ndarray:
    term = ((eta_bb * (eta_bb * (1 - A) - eta_aa * A) + 2 * eta_ab ** 2 * A)
             / np.sqrt((eta_aa * A - eta_bb * (1 - A)) ** 2 + 4 * eta_ab ** 2 * A * (1 - A))
            )
    term_nan_free = np.where(~np.isnan(term), term, 0.0)

    return eta_bb + lam * term_nan_free


@njit(cache=True)
def eta_dVdnb_jit(A: cp.ndarray, lam: float,
                  eta_aa: float, eta_bb: float, eta_ab: float) -> cp.ndarray:
    term = ((eta_bb * (eta_bb * (1 - A) - eta_aa * A) + 2 * eta_ab ** 2 * A)
             / np.sqrt((eta_aa * A - eta_bb * (1 - A)) ** 2 + 4 * eta_ab ** 2 * A * (1 - A))
            )
    term_nan_free = np.where(~np.isnan(term), term, 0.0)

    return eta_bb + lam * term_nan_free
    

@njit('f8[:,:,:](c16[:,:,:], f8)')
def get_density_jit(func_val: cp.ndarray, p: float) -> cp.ndarray:
    """
    Calculates :math:`|\psi|^2` for 1D, 2D or 3D (depending on self.dim).

    :param func_val: Array of function values to get p-norm for.

    :return: :math:`|\psi|^2`

    """
    return func_val.real ** p + func_val.imag ** p


@njit(cache=True)
def get_density_jit(func_val: cp.ndarray, p: float) -> cp.ndarray:
    """
    Calculates :math:`|\psi|^2` for 1D, 2D or 3D (depending on self.dim).

    :param func_val: Array of function values to get p-norm for.

    :return: :math:`|\psi|^2`

    """
    return func_val.real ** p + func_val.imag ** p

