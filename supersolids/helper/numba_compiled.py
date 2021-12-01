#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Function speeded up with numba

"""

import numpy as np
import numba as nb

from numba.pycc import CC


cc = CC('numba_compiled')
cc.verbose = True


@cc.export('get_H_pot_exponent_terms_jit',
           'c16[:,:,:](f8[:,:,:], f8, f8, c16[:,:,:], f8[:,:,:], f8[:,:,:])')
def get_H_pot_exponent_terms_jit(V_val: np.ndarray,
                                 a_dd_factor: float,
                                 a_s_factor: float,
                                 dipol_term: np.ndarray,
                                 contact_interaction: np.ndarray,
                                 mu_lhy: np.ndarray) -> np.ndarray:
    return (V_val
            + a_dd_factor * dipol_term
            + a_s_factor * contact_interaction
            + mu_lhy
            )


@cc.export('get_H_pot_jit', 'c16[:,:,:](c16, f8, c16[:,:,:], f8)')
def get_H_pot_jit(U: np.complex, dt: float, terms: np.ndarray,
                  split_step: float = 0.5) -> np.ndarray:
    return np.exp(U * (split_step * dt) * terms)


@cc.export('f_lam', 'f8[:](f8[:], f8, f8, f8, f8)')
def f_lam(A: np.ndarray, lam: float,
          eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
    return (eta_aa * A
            + eta_bb * (1.0 - A)
            + lam * np.sqrt((eta_aa * A - eta_bb * (1.0 - A)) ** 2.0
                            + 4.0 * eta_ab ** 2.0 * A * (1.0 - A)))


@cc.export('eta_dVdna_jit', 'f8[:](f8[:], f8, f8, f8, f8)')
def eta_dVdna_jit(A: np.ndarray, lam: float,
                  eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
    return (eta_aa
            + lam * (eta_aa * (eta_aa * A - eta_bb * (1 - A))
                     + 2 * eta_ab ** 2 * (1 - A)
                     )
            / np.sqrt((eta_aa * A - eta_bb * (1 - A)) ** 2
                      + 4 * eta_ab ** 2 * A * (1 - A)
                      )
            )


@cc.export('eta_dVdnb_jit', 'f8[:](f8[:], f8, f8, f8, f8)')
def eta_dVdnb_jit(A: np.ndarray, lam: float,
                  eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
    return (eta_bb
            + lam * (eta_bb * (eta_bb * (1 - A) - eta_aa * A)
                     + 2 * eta_ab ** 2 * A)
            / np.sqrt((eta_aa * A - eta_bb * (1 - A)) ** 2
                      + 4 * eta_ab ** 2 * A * (1 - A)
                      )
            )

if __name__ == "__main__":
    cc.compile()