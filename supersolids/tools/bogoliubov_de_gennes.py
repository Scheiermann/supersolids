#!/usr/bin/env python
import argparse
import functools
import json
import logging
import os
import psutil
import sys
from pathlib import Path
import traceback

from matplotlib import pyplot as plt

import dill
import numpy as np
import dask.array as da
from dask.distributed import Client

from scipy.ndimage import minimum_position, maximum_position
from scipy.special import eval_hermite, factorial
from scipy.special import hermite
from scipy.sparse.linalg import eigs, eigsh
from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
from scipy.sparse.linalg import LinearOperator

from scipy.sparse import csr_matrix, load_npz, save_npz 

from supersolids.helper import constants, functions, get_path, get_version
from supersolids.helper.run_time import run_time
from supersolids.helper.Resolution import Resolution, ResAssert



def flags(args_array):
    parser = argparse.ArgumentParser(description="Load old simulations of Schrödinger system "
                                                 "and create movie.")
    parser.add_argument("-dir_path", metavar="dir_path", type=str, default="~/supersolids/results",
                        help="Absolute path to load npz data from")
    parser.add_argument("-dir_name", metavar="dir_name", type=str, default="movie001",
                        help="Formatting of directory name where the files to load lie. "
                             "Use movie%03d for dir_names like movie001.")
    parser.add_argument("-filename_schroedinger", metavar="filename_schroedinger", type=str,
                        default="schroedinger.pkl",
                        help="Name of file, where the Schroedinger object is saved")
    parser.add_argument("-filename_steps", type=str, default="step_",
                        help="Name of file, without enumarator for the files. "
                             "For example the standard naming convention is step_0000001.npz, "
                             "the string needed is step_")
    parser.add_argument("-steps_format", metavar="steps_format", type=str, default="%07d",
                        help="Formating string to enumerate the files. "
                             "For example the standard naming convention is step_0000001.npz, "
                             "the string needed is %07d")
    parser.add_argument("-frame", type=json.loads, default=None, help="Counter of first saved npz.")
    parser.add_argument("-mode", type=str, default="dask",
                        help="Different ways of calculation. Choose between dask, cupy, flat.")
    parser.add_argument("-graphs_dirname", type=str, default="graphs",
                        help="Name of directory for the results.")
    parser.add_argument("--recalculate", default=False, action="store_true",
                        help="Ignores saved results for the parameters, "
                             "then recalculates and overwrites the old results.")
    parser.add_argument("--dipol", default=False, action="store_true",
                        help="Calculates dipolterms for the BdG matrix.")
    parser.add_argument("--dask_dipol",  default=False, action="store_true",
                        help="Use dask to calculate dipol part in parallel.")
    parser.add_argument("--ground_state", default=False, action="store_true",
                        help="Assumes ground states as inputs. Meaning psi=conjugate(psi).")
    parser.add_argument("-print_num_eigenvalues", type=int, default=20,
                        help="Number of eigenvalues printed.")
    parser.add_argument("-nx", type=int, default=4,
                        help="Number of Hermite polynomials used for x axis.")
    parser.add_argument("-ny", type=int, default=4,
                        help="Number of Hermite polynomials used for y axis.")
    parser.add_argument("-nz", type=int, default=4,
                        help="Number of Hermite polynomials used for z axis.")
    parser.add_argument("-stepper_x", type=int, default=1,
                        help="Stepper in x direction to sample psi for the bogoliubov.")
    parser.add_argument("-stepper_y", type=int, default=1,
                        help="Stepper in y direction to sample psi for the bogoliubov.")
    parser.add_argument("-stepper_z", type=int, default=1,
                        help="Stepper in z direction to sample psi for the bogoliubov.")
    parser.add_argument("-l_0", metavar="l_0", type=float, default=None,
                        help="Help constant for dimensionless formulation of equations.")
    parser.add_argument("-label", type=str, default="", help="Label to name result dirnames.")
    parser.add_argument("--get_eigenvalues", default=False, action="store_true",
                        help="Use flag to calculate eigenvalues")
    parser.add_argument("--pytorch", default=False, action="store_true",
                        help="Use flag to use pytorch method (GPU or CPU) to get eigenvalues")
    parser.add_argument("--arnoldi", default=False, action="store_true",
                        help="Use flag to use arnoldi method to get eigenvalues")
    parser.add_argument("-arnoldi_num_eigs", type=int, default=30,
                        help="Number of eigenvalues computed by arnoldi method.")
    parser.add_argument("--gpu_off", default=False, action="store_true",
                        help="Use flag to turn off gpu eventhouh it might be usable")
    parser.add_argument("--cut_hermite_orders", default=False, action="store_true",
                        help="Use flag to cut the dimension of nx, ny, nz "
                             "down with the condition (i + j + k) <= (nx + ny + nz) // 2")
    parser.add_argument("--cut_hermite_values", default=False, action="store_true",
                        help="Use flag to cut the dimension of nx, ny, nz "
                             "by using a minimal cut of for the entries.")
    parser.add_argument("--reduced_version", default=False, action="store_true",
                        help="Use (a - b) @ (a + b)")
    # parser.add_argument("--lin_op", default=False, action="store_true",
    #                     help="Use eigs with linear Operator (without defining the bogoliubov matrix).")
    parser.add_argument("-gpu_index", type=int, default=0, help="Use to set index of cuda device.")
    parser.add_argument("-csr_cut_off_0", type=float, default=0.001,
                        help="Cut values under this to 0 when saving the bogoliubov matrix for CSR. "
                              "Essentially making the matrix more sparse and saving memory.")

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


class BogOperator(LinearOperator):
    def __init__(self, psi, g_qf,
                 a_s, a_dd,
                 a_s_factor, a_dd_factor,
                 v, fvp, kx, ky, kz, mu, nx, ny, nz,
                 eigen_max,
                 cupy_used=False,
                 ):
        self.cupy_used = cupy_used
        if self.cupy_used:
            self.psi = cp.array(psi)
            self.v = cp.array(v)
            self.fvp = cp.array(fvp)
            self.kx = cp.array(kx)
            self.ky = cp.array(ky)
            self.kz = cp.array(kz)
        else:
            self.psi = psi
            self.v = v
            self.fvp = fvp
            self.kx = kx
            self.ky = ky
            self.kz = kz

        self.g_qf = g_qf
        self.a_s_factor = a_s_factor
        self.a_s = a_s
        self.a_dd_factor = a_dd_factor
        self.a_dd = a_dd
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.mu = mu
        self.eigen_max = eigen_max
        self.shape = (len(np.ravel(self.psi)), len(np.ravel(self.psi)))
        # self.dtype = np.dtype(dtype)
        # print(f"psi: {psi} {psi.dtype}")
        print(f"g_qf: {g_qf} {type(g_qf)}")
        print(f"a_s: {a_s} {type(a_s)}")
        print(f"a_dd: {a_dd} {type(a_dd)}")
        print(f"a_s_factor: {a_s_factor} {type(a_s_factor)}")
        print(f"a_dd_factor: {a_dd_factor} {type(a_dd_factor)}")
        # print(f"v: {v} {v.dtype}")
        # print(f"fvp: {fvp} {fvp.dtype}")
        # print(f"kx: {kx} {kx.dtype}")
        # print(f"ky: {ky} {ky.dtype}")
        # print(f"kz: {kz} {kz.dtype}")
        print(f"nx: {nx} {type(nx)}")
        print(f"ny: {ny} {type(ny)}")
        print(f"nz: {nz} {type(nz)}")
        print(f"mu: {mu} {type(mu)}")

    def test_H(self, dV):
        return np.sum(self.psi * self.gpe_ham(self.psi)) * dV

    def test_kin(self, dV):
        return np.sum(self.psi * self.kin_part(self.psi)) * dV

    def test_v(self, dV):
        return np.sum(self.psi * self.v_part(self.psi)) * dV

    def norm_psi(self, dV):
        return np.sum(self.psi * self.psi) * dV

    def dipolar(self, field, conj):
        if conj:
            if self.cupy_used:
                dip = cp.fft.fftn(self.fvp * cp.fft.ifftn(field))
            else:
                dip = np.fft.fftn(self.fvp * np.fft.ifftn(field))
        else:
            if self.cupy_used:
                dip = cp.fft.ifftn(self.fvp * cp.fft.fftn(field))
            else:
                dip = np.fft.ifftn(self.fvp * np.fft.fftn(field))

        return dip
        
    def laplacian(self, x):
        if self.cupy_used:
            lap = -cp.fft.ifftn((self.kx ** 2 + self.ky ** 2 + self.kz ** 2) * cp.fft.fftn(x))
        else:
            lap = -np.fft.ifftn((self.kx ** 2 + self.ky ** 2 + self.kz ** 2) * np.fft.fftn(x))

        return lap

    def v_part(self, x):
        ham_psi = self.v * x
        return ham_psi

    def kin_part(self, x):
        ham_psi = -0.5 * self.laplacian(x)
        return ham_psi


    def gpe_ham(self, x, mu_factor=1.0):
        if self.cupy_used:
            x = cp.array(x)
            dens = cp.abs(self.psi) ** 2.0
            ham_psi = (-0.5 * self.laplacian(x)
                       + x * (self.v
                              + self.a_s_factor * self.a_s * dens
                              + self.a_dd_factor * self.a_dd * self.dipolar(dens, conj=False)
                              + g_qf * cp.abs(self.psi) ** 3.0 
                              - mu_factor * self.mu
                              )
                       )
        else:
            dens = np.abs(self.psi) ** 2.0
            ham_psi = (-0.5 * self.laplacian(x)
                       + x * (self.v
                              + self.a_s_factor * self.a_s * dens
                              + self.a_dd_factor * self.a_dd * self.dipolar(dens, conj=False)
                              + g_qf * np.abs(self.psi) ** 3.0 
                              - mu_factor * self.mu
                              )
                       )

        return ham_psi

    def gpe_X(self, x, conj):
        if self.cupy_used:
            x = cp.array(x)
            ham_psi = (self.a_dd_factor * self.a_dd * self.psi * self.dipolar(self.psi * x, conj)
                       + x * (self.a_s_factor * self.a_s * cp.abs(self.psi) ** 2.0
                              + 1.5 * g_qf * cp.abs(self.psi) ** 3.0)
                      )
        else:
            ham_psi = (self.a_dd_factor * self.a_dd * self.psi * self.dipolar(self.psi * x, conj)
                       + x * (self.a_s_factor * self.a_s * np.abs(self.psi) ** 2.0
                              + 1.5 * g_qf * np.abs(self.psi) ** 3.0)
                      )
        return ham_psi

    def _matvec(self, x):
        if self.cupy_used:
            x = cp.array(x)

        x = x.reshape((self.nx, self.ny, self.nz))
        y = self.gpe_ham(x) + 2.0 * self.gpe_X(x, conj=False)

        if self.cupy_used:
            matvec = cp.asnumpy(cp.ravel(self.gpe_ham(y) - self.eigen_max * x))
        else:
            matvec = np.ravel(self.gpe_ham(y) - self.eigen_max * x)

        return matvec

    def _rmatvec(self, x):
        if self.cupy_used:
            x = cp.array(x)

        x = x.reshape((self.nx, self.ny, self.nz))
        y = self.gpe_ham(x)

        if self.cupy_used:
            rmatvec = cp.asnumpy(cp.ravel(self.gpe_ham(y) + 2.0 * self.gpe_X(y, conj=True) - cp.conj(self.eigen_max) * x))
        else:
            rmatvec = np.ravel(self.gpe_ham(y) + 2.0 * self.gpe_X(y, conj=True) - np.conj(self.eigen_max) * x) 

        return rmatvec


def mat2d(mat, label="", precision=4, formatter={'float': '{:0.3f}'.format}, linewidth=250):

    with cp.printoptions(precision=precision, suppress=True, formatter=formatter,
                         linewidth=linewidth):
        if cupy_used:
            print(f"{label}\n{np.matrix(cp.asnumpy(mat))}") 
        else:
            print(f"{label}\n{np.matrix(mat)}") 



def harmonic_eigenstate(x, n, a, cupy_used=True):
    prefactor = np.sqrt(1.0 / (factorial(n) * 2.0 ** n )) * np.pi ** -0.25 * a ** 0.25
    # herm = hermite(n)(x)
    herm = hermite(n)(np.sqrt(a) * x)
    result = prefactor * np.exp(- a * x ** 2.0 / 2.0) * herm
    if cupy_used:
        cp.asarray(result)

    return result


def HO_1D(ind, x, a=1):
    return (1./np.sqrt(2.0 ** ind * factorial(ind) * np.sqrt(np.pi)) * a ** 0.25
            * np.exp(-a * x ** 2.0 /2.0) * eval_hermite(ind, np.sqrt(a) * x))


def indices(ind, ind_y_max, ind_z_max):
    ind_x = ind//(ind_y_max * ind_z_max)
    ind_y = (ind - ind_x * ind_y_max * ind_z_max)//ind_z_max
    ind_z = ind - ind_x * ind_y_max * ind_z_max - ind_y * ind_z_max
    return ind_x, ind_y, ind_z


def position(System, pos):
    pos_x, pos_y, pos_z = indices(pos, System.Res.y, System.Res.z)
    x = System.Box.x0 + System.dx * pos_x
    y = System.Box.y0 + System.dy * pos_y
    z = System.Box.z0 + System.dz * pos_z    

    return x, y, z


def operator_ravel(operator, pos, nymax, nzmax, cupy_used=True):
    pos_x, pos_y, pos_z = indices(pos, nymax, nzmax)
    operator_list = []
    index_list = []
    for ix, iy, iz in zip(pos_x[0, :], pos_y[0, :], pos_z[0, :]):
        operator_list.append(operator[ix, iy, iz])
        index_list.append((ix, iy, iz))
       
    if cupy_used:
        operator_ordered = cp.array(operator_list)
    else:
        operator_ordered = np.array(operator_list)

    return operator_ordered


def position_revert(x, y, z, xmax, ymax, zmax, nxmax, nymax, nzmax):
    dx, dy, dz = 2 * xmax/nxmax, 2 * ymax/nymax, 2 * zmax/nzmax
    pos_x = (x + xmax) / dx
    pos_y = (y + ymax) / dy
    pos_z = (z + zmax) / dz

    pos = np.vectorize(int)(pos_z + pos_x * nymax * nzmax + pos_y * nzmax)

    return pos


def HO_3D_old(x, y, z, ind, ind_y_max, ind_z_max, a_y=1, a_z=1, cupy_used=True):
    a_x = 1
    logging.info("Get ind_x")
    ind_x, ind_y, ind_z = indices(ind, ind_y_max, ind_z_max)
    logging.info(f"ind.shape: {ind.shape}")
    logging.info(f"ind_x.shape: {ind_x.shape}")
    # logging.info(f"ind_x: {ind_x}")
    # logging.info(f"ind_y: {ind_y}")
    # logging.info(f"ind_z: {ind_z}")
    logging.info(f"x.shape: {x.shape}")
    logging.info(f"y.shape: {y.shape}")
    logging.info(f"z.shape: {z.shape}")
    logging.info("Get herm_3d")
    herm_3d = HO_1D(ind_x, x, a_x) * HO_1D(ind_y, y, a_y) * HO_1D(ind_z, z, a_z)
    logging.info(f"herm_3d: {round(herm_3d.nbytes / 1024 / 1024,2)}MB")
    logging.info(f"herm_3d.dtype: {herm_3d.dtype}")

    if cupy_used:
        herm_3d = cp.array(herm_3d)

    return herm_3d


def HO_3D(x, y, z, ind_x, ind_y, ind_z, a_y=1, a_z=1, cupy_used=True):
    a_x = 1
    logging.info(f"ind_x.shape: {ind_x.shape}")
    logging.info(f"x.shape: {x.shape}")
    logging.info("Get herm_3d")
    herm_3d = HO_1D(ind_x, x, a_x) * HO_1D(ind_y, y, a_y) * HO_1D(ind_z, z, a_z)
    logging.info(f"herm_3d: {round(herm_3d.nbytes / 1024 / 1024,2)}MB")

    if cupy_used:
        herm_3d = cp.array(herm_3d)

    return herm_3d


def get_hermite_dipol(System, index_dict, dim, entry, ground_state=True):
    logging.info(f"dipol entry: {entry}")
    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)
    dV = System.volume_element()

    if ground_state:
        op = np.real(System.psi_val_list[0])
        # op2 = op
    else:
        if entry == "A":
            op = System.psi_val_list[0]
            op2 = cp.conjugate(System.psi_val_list[0])
        elif entry == "B":
            op = System.psi_val_list[0]
            op2 = System.psi_val_list[0]
        elif entry == "C":
            op = cp.conjugate(System.psi_val_list[0])
            op2 = cp.conjugate(System.psi_val_list[0])
        elif entry == "D":
            op = cp.conjugate(System.psi_val_list[0])
            op2 = System.psi_val_list[0]
        else:
            sys.exit("No such entry. Choose between A, B, C, D.")

    if cupy_used:
       op = cp.array(op)
       V_k_val = cp.array(System.V_k_val)
       if not ground_state:
           op2 = cp.array(op2)
    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    # if ground_state:
    #     op_helper = harmonic_raveled * cp.ravel(op2)
    #     gamma_summed = op_helper * cp.fft.ifftn(V_k_val * cp.fft.fftn(op_helper))
    # else:
    #     gamma_summed = (harmonic_raveled * cp.ravel(op)) * cp.fft.ifftn(V_k_val * cp.fft.fftn(harmonic_raveled * cp.ravel(op2)))

    j_nr = cp.zeros((dim, System.Res.x * System.Res.y * System.Res.z))
    f_rn = cp.zeros((System.Res.x * System.Res.y * System.Res.z, dim))
    logging.info(f"f_rn.itemsize: {f_rn.itemsize}")
    logging.info(f"f_rn.itemsize: {f_rn.nbytes}")
    logging.info(f"j_nr.itemsize: {f_rn.nbytes}")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    # harmonic_vector = cp.zeros((dim, System.Res.x, System.Res.y, System.Res.z))
    # for i in range(dim):
    #     comb1 = index_dict[i]
    #     harmonic_vector[i, :, :, :] = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y=a_y, a_z=a_z)
    # for i in range(dim):
    #     j_nr[i, :] = cp.ravel(harmonic_vector[i, :, :, :])
    #     f_rn[:, i] = cp.ravel(cp.fft.ifftn(V_k_val * cp.fft.fftn(harmonic_vector[i, :, :, :] * op2)))

    harmonic_vector_i = cp.zeros((System.Res.x, System.Res.y, System.Res.z))
    logging.info(f"dim: {dim}")
    # logging.info(f"index_dict: {index_dict}")
    logging.info(f"len(index_dict): {len(index_dict)}")
    for i in range(dim):
        comb1 = index_dict[i]
        harmonic_vector_i = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y=a_y, a_z=a_z)
        j_nr[i, :] = cp.ravel(harmonic_vector_i)
        if ground_state:
            # op2 = op
            f_rn[:, i] = cp.ravel(cp.real(cp.fft.ifftn(V_k_val * cp.fft.fftn(harmonic_vector_i * op))))
        else:
            f_rn[:, i] = cp.ravel(cp.fft.ifftn(V_k_val * cp.fft.fftn(harmonic_vector_i * op2)))

        if i == 0:
            logging.info(f"harmonic_vector_i.itemsize: {harmonic_vector_i.nbytes}")
            logging.info(f"harmonic_vector_i.shape: {harmonic_vector_i.shape}")
            logging.info(f"harmonic_vector_i.dtype: {harmonic_vector_i.dtype}")
            logging.info(f"harmonic_vector_i.itemsize: {harmonic_vector_i.itemsize}")
        harmonic_vector_i = None

    if cupy_used:
        f_rn = cp.asnumpy(f_rn)
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    if entry in ["A", "D"]:
        if ground_state or (entry == "A"):
           dens_without_N = cp.abs(op) ** 2.0
        else:
           dens_without_N = cp.abs(op2) ** 2.0

        dip_helper = cp.ravel(cp.fft.ifftn(V_k_val * cp.fft.fftn(dens_without_N)))
        if ground_state:
            dip_helper = cp.real(dip_helper)

        # for i, (l, m) in enumerate(np.ndindex(dim, dim)):
        #     for j, (l, m) in enumerate(np.ndindex(dim, dim)):
        #     beta_summed[i, j] = cp.dot(cp.ravel(harmonic_vector[i, :, :, :]), cp.ravel(harmonic_vector[j, :, :, :]) * dip_helper)
        try:
            h_rn = cp.zeros((System.Res.x * System.Res.y * System.Res.z, dim))
            logging.info(f"h_rn.shape: {h_rn.shape}")
        except Exception as e: 
            logging.info(f"WARNING: numpy used for h_rn")
            h_rn = np.zeros((System.Res.x * System.Res.y * System.Res.z, dim))
        try:
            # for i in range(dim):
            #     h_rn[:, i] = (j_nr[i, :] * dip_helper)
            h_rn = (j_nr * dip_helper).T
            dip_helper = None
            beta_summed = cp.dot(j_nr, h_rn)
        except Exception as e: 
            logging.info(f"WARNING: numpy used for beta_summed")
            h_rn = None
            j_nr = cp.asnumpy(j_nr)
            dip_helper = cp.asnumpy(dip_helper)
            h_rn = np.zeros((System.Res.x * System.Res.y * System.Res.z, dim))
            h_rn = (j_nr * dip_helper).T
            beta_summed = np.dot(j_nr, h_rn)

        if cupy_used:
            beta_summed = cp.asnumpy(beta_summed)

        h_rn = None
        dip_helper = None
        logging.info(f"beta_summed.shape: {beta_summed.shape}")

    logging.info(f"V_k_val.itemsize: {V_k_val.itemsize}")
    logging.info(f"V_k_val.itemsize: {V_k_val.nbytes}")
    if not ground_state:
        logging.info(f"op2.itemsize: {op2.itemsize}")
        logging.info(f"op2.itemsize: {op2.nbytes}")

    # g_nr = cp.zeros((dim, System.Res.x * System.Res.y * System.Res.z))
    # for i in range(dim):
    #     g_nr[i, :] = j_nr[i, :] * cp.ravel(op)
    try:
        j_nr *= cp.ravel(op)
        # if cupy_used:
        #     f_rn = cp.array(f_rn)
        j_nr = cp.asnumpy(j_nr)
        f_rn = cp.asnumpy(f_rn)
        gamma_summed = np.dot(j_nr, f_rn)
        # gamma_summed = cp.dot(j_nr, f_rn)
    except Exception as e: 
        logging.info(f"WARNING: numpy used for gamma_summed")
        gamma_summed = None
        print(f"j_nr: {type(j_nr)}")
        print(f"f_rn: {type(f_rn)}")
        print(f"op: {type(op)}")
        j_nr = cp.asnumpy(j_nr)
        f_rn = cp.asnumpy(f_rn)
        op = cp.asnumpy(op)
        j_nr *= np.ravel(op)
        gamma_summed = np.dot(j_nr, f_rn)

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    if cupy_used:
        gamma_summed = cp.asnumpy(gamma_summed)

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    # logging.info(f"g_nr.shape: {g_nr.shape}")
    logging.info(f"f_rn.shape: {f_rn.shape}")
    logging.info(f"j_nr.dtype: {j_nr.dtype}")
    # logging.info(f"g_nr.dtype: {g_nr.dtype}")
    logging.info(f"f_rn.dtype: {f_rn.dtype}")
    logging.info(f"op.dtype: {op.dtype}")
    logging.info(f"gamma_summed.shape: {gamma_summed.shape}")

    g_nr = None
    f_rn = None

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    if entry in ["A", "D"]:
        hermite_dipol = (gamma_summed + beta_summed) * dV

        if cupy_used:
            mempool = cp.get_default_memory_pool()
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")
    else:
        # release gpu memory
        j_nr = None
        hermite_dipol = gamma_summed * dV
    
    if cupy_used:
        hermite_dipol = cp.asnumpy(hermite_dipol)

    logging.info(f"hermite_dipol.shape: {hermite_dipol.shape}")

    return hermite_dipol


def get_hermite_dipol_old(System, comb1, comb2, ground_state=True, entry="A", beta_helper=None, gamma_helper=None):
    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)

    # phi_i = harmonic_eigenstate_3d_dask(System.x_mesh, System.y_mesh, System.z_mesh, comb1[0], comb1[1], comb1[2], a_y=a_y, a_z=a_z)
    # phi_j = harmonic_eigenstate_3d_dask(System.x_mesh, System.y_mesh, System.z_mesh, comb2[0], comb2[1], comb2[2], a_y=a_y, a_z=a_z)

    phi_i = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y=a_y, a_z=a_z)
    phi_j = harmonic_eigenstate_3d(System, comb2[0], comb2[1], comb2[2], a_y=a_y, a_z=a_z)

    if ground_state:
        op = System.psi_val_list[0]
        op2 = op
    else:
        if entry == "A":
            op = System.psi_val_list[0]
            op2 = np.conjugate(System.psi_val_list[0])
        elif entry == "B":
            op = System.psi_val_list[0]
            op2 = System.psi_val_list[0]
        elif entry == "C":
            op = np.conjugate(System.psi_val_list[0])
            op2 = np.conjugate(System.psi_val_list[0])
        elif entry == "D":
            op = np.conjugate(System.psi_val_list[0])
            op2 = System.psi_val_list[0]
        else:
            sys.exit("No such entry. Choose between A, B, C, D.")

    if cupy_used:
       op = cp.array(op)
       op2 = cp.array(op2)
       V_k_val = cp.array(System.V_k_val)

    if gamma_helper is None:
        gamma_helper = cp.fft.ifftn(V_k_val * cp.fft.fftn(phi_i * op2))
    herm_dip = phi_j * op * gamma_helper

    if entry in ["A", "D"]:
        if ground_state or (entry == "A"):
           dens_without_N = cp.abs(op) ** 2.0
        else:
           dens_without_N = cp.abs(op2) ** 2.0
        if beta_helper is None:
            beta_helper = cp.fft.ifftn(V_k_val * cp.fft.fftn(dens_without_N))
        herm_dip += phi_i * phi_j * beta_helper
     
    if cupy_used:
        hermite_dipol = cp.sum(cp.real(herm_dip)) * System.volume_element() 

    return hermite_dipol, beta_helper, gamma_helper


def En_TF(nr,l):
    return np.sqrt(2 * nr**2 + 2 * nr * l + 3 * nr + l)


def En(ind, ind_y_max, ind_z_max, a_y=1, a_z=1):
    ind_x, ind_y, ind_z = indices(ind, ind_y_max, ind_z_max)
    return (ind_x + 0.5) + a_y * (ind_y + 0.5) + a_z * (ind_z + 0.5)

    
def harmonic_eigenstate_3d(System, i, j, k, a_y, a_z, cupy_used=True):
    harmonic_eigenstate_2d = harmonic_eigenstate(System.x_mesh, i, a=1) * harmonic_eigenstate(System.y_mesh, j, a_y)
    harmonic_eigenstate_3d = harmonic_eigenstate_2d * harmonic_eigenstate(System.z_mesh, k, a_z)

    if cupy_used:
        harmonic_eigenstate_3d = cp.asarray(harmonic_eigenstate_3d)

    return harmonic_eigenstate_3d


def harmonic_eigenstate_3d_dask(x_mesh, y_mesh, z_mesh, i, j, k, a_y, a_z):
    harmonic_eigenstate_2d = harmonic_eigenstate(x_mesh, i, a=1) * harmonic_eigenstate(y_mesh, j, a_y)
    harmonic_eigenstate_3d = harmonic_eigenstate_2d * harmonic_eigenstate(z_mesh, k, a_z)

    return cp.asarray(harmonic_eigenstate_3d)


def hermite_transform(System, operator, comb1, comb2, a_y, a_z,
                     fourier_space: bool = False, dV: float = None, sandwich = True, lhy_terms=False, cupy_used=True):
    if (operator is None) or (comb1 is None) or (comb2 is None):
        return cp.array(0.0)

    herm_3d = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y, a_z, cupy_used=cupy_used)
    integrand = herm_3d * operator
    # integrand = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y, a_z, cupy_used=cuda_used) * operator
    if sandwich:
        integrand = integrand * herm_3d
        transform: float = System.sum_dV(integrand, fourier_space=fourier_space, dV=dV)
    else:
        if dV is None:
            transform: float = cp.sum(integrand) * System.volume_element()
        else:
            transform: float = cp.sum(integrand) * dV
            print(f"dV in transform: {dV}")
 
    return transform

def hermite_transform_with_lhy(System, operator, comb1, comb2, a_y, a_z,
                               fourier_space: bool = False, dV: float = None, sandwich = True,
                               cupy_used=True):
    if (operator is None) or (comb1 is None) or (comb2 is None):
        return cp.array(0.0)
    lhy_abc_list = []

    herm_3d = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y, a_z, cupy_used=cupy_used)
    integrand = herm_3d * operator
    # integrand = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2], a_y, a_z, cupy_used=cuda_used) * operator
    if sandwich:
        # integrand = integrand * harmonic_eigenstate_3d(System, comb2[0], comb2[1], comb2[2], a_y, a_z, cupy_used=cuda_used)
        integrand = integrand * herm_3d
        transform: float = System.sum_dV(integrand, fourier_space=fourier_space, dV=dV)
    else:
        if dV is None:
            transform: float = cp.sum(integrand) * System.volume_element()
        else:
            transform: float = cp.sum(integrand) * dV
            print(f"dV in transform: {dV}")

    if np.all(np.array(comb1) == 0) and np.all(np.array(comb2) == 0):
        print(f"System.lhy_factor: {System.lhy_factor}")
    if System.lhy_factor != 0:
        g_qf = functions.get_g_qf_bog(N=System.N_list[0],
                                      a_s=float(System.a_s_array[0, 0]),
                                      a_dd=float(System.a_dd_array[0, 0]))
        if np.all(np.array(comb1) == 0) and np.all(np.array(comb2) == 0):
            print(f"g_qf: {g_qf}")
        lhy_a = get_lhy_term_linear(System, index=0, ground_state=True, cupy_used=cupy_used)
        lhy_b = get_lhy_term_linear(System, index=1, ground_state=True, cupy_used=cupy_used)
        if ground_state:
            lhy_c = np.copy(lhy_b)
        else:
            lhy_c = get_lhy_term_linear(System, index=2, ground_state=True, cupy_used=cupy_used)

        lhy_abc_list = [lhy_a, lhy_b, lhy_c]
        with run_time(name=f"lhy all"):
            for lhy_key in lhy_abc_list:
                lhy_abc_list.append(g_qf * System.sum_dV(herm_3d * lhy_key * herm_3d, fourier_space=False))
 
        return transform, lhy_abc_list
    else:
        return transform


def hermite_transform_dask(dV, x_mesh, y_mesh, z_mesh, operator, comb1, comb2, a_y, a_z):
    if (operator is None) or (comb1 is None) or (comb2 is None):
        return cp.array(0.0)
    integrand = (harmonic_eigenstate_3d_dask(x_mesh, y_mesh, z_mesh, comb1[0], comb1[1], comb1[2], a_y, a_z) * operator \
                 * harmonic_eigenstate_3d_dask(x_mesh, y_mesh, z_mesh, comb2[0], comb2[1], comb2[2], a_y, a_z))
    transform: float = cp.sum(integrand) * dV

    return transform


def get_index_dict(nx, ny, nz):
    index_dict = {i * ny * nz + j * nz + k : [i, j, k]
                  for i in range(nx)
                  for j in range(ny)
                  for k in range(nz)}

    return index_dict

def get_ind_vec_dict(nx, ny, nz):
    # max_sum = (nx + ny + nz) // 2
    # condition = lambda i, j, k: (i + j + k) <= max_sum
    # condition = lambda i, j, k: (i/nx + j/ny + k/nz) <= 1
    # condition = lambda i, j, k: (i/nx) ** 2.0 + (j/ny) ** 2.0 + (k/nz) ** 2.0 <= 1.8
    condition = lambda i, j, k: (i/nx) ** 2.0 + (j/ny) ** 2.0 + (k/nz) ** 2.0 <= 3.0
    # condition = lambda i, j, k: nx + ny + nz <= nx + ny + nz + 1
    index_list_short = np.array([(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz) if condition(i, j, k)])

    return index_list_short

def get_bog_dict(index_dict, dim, maxi=0.00001):
    comb_list_list = []
    sum_list_list = []
    for j in range(dim):
        ind = np.ravel(np.argwhere(np.abs(bogoliubov_matrix[j, : dim]) > maxi))
        comb_list = [index_dict[ind[i]] for i in range(len(ind))]
        sum_list = list(map(sum, [index_dict[ind[i]] for i in range(len(ind))]))
        comb_list_list.append(comb_list)
        sum_list_list.append(sum_list)
        
    return comb_list_list, sum_list_list

def get_parity(comb1, comb2):
    assert len(comb1) == len(comb2)
    summed = np.array(comb1) + np.array(comb2)
    parity = len([i for i in summed if i % 2 == 0]) == len(comb1)

    return parity

def get_hermite_matrix(System, operator, dim, index_dict):
    hermite_matrix = cp.zeros((dim, dim))
    E_H0 = cp.zeros((dim, dim))
    triu_0, triu_1 = np.triu_indices(dim)

    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)
    for l, m in zip(triu_0, triu_1):
        comb1 = index_dict[l]
        comb2 = index_dict[m]
        if l == m:
            E_H0[l, m] = ((comb1[0] + 0.5)
                          + (System.w_y / System.w_x) * (comb1[1] + 0.5)
                          + (System.w_z / System.w_x) * (comb1[2] + 0.5))
        # with run_time(name=f"{l},{m} integrated"):
        if get_parity(comb1, comb2):
            hermite_matrix[l, m] = hermite_transform(System, operator, comb1, comb2, a_y, a_z)
        else:
            hermite_matrix[l, m] = 0.0

    return E_H0, hermite_matrix


def get_hermite_matrix_dask(System, operator, dim, index_dict, fast = True):
    save_RAM = True
    if save_RAM:
        hermite_matrix = np.zeros((dim, dim))
    else:
        hermite_matrix = cp.zeros((dim, dim))
    E_H0 = np.zeros((dim, dim))
    triu_0, triu_1 = np.triu_indices(dim)
    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)

    if fast:
        print(f"fast")
        dV = System.volume_element(fourier_space=False)
        [dV_dask, x_dask, y_dask, z_dask, operator_dask] = client.scatter([dV,
                                                                           System.x_mesh,
                                                                           System.y_mesh,
                                                                           System.z_mesh,
                                                                           operator])
    else:
        print(f"Not fast")
        [System_dask, operator_dask] = client.scatter([System, operator])

    futures = []
    for l, m in zip(triu_0, triu_1):
        comb1 = index_dict[l]
        comb2 = index_dict[m]
        if l == m:
            E_H0[l, m] = ((comb1[0] + 0.5)
                          + a_y * (comb1[1] + 0.5)
                          + a_z * (comb1[2] + 0.5))
        if not get_parity(comb1, comb2):
            comb1 = None
            comb2 = None
        if fast:
            futures.append(client.submit(hermite_transform_dask, dV_dask, x_dask, y_dask, z_dask,
                                         operator_dask, comb1, comb2, a_y, a_z))
        else:
            futures.append(client.submit(hermite_transform, System_dask, operator_dask,
                                         comb1, comb2, a_y, a_z))
    logging.info(f"gather")
    results = client.gather(futures)
    logging.info(f"gathered")
    client.close()
    logging.info(f"close")

    #  put results into correct array shape
    # results_arr = cp.array(results)

    print(f"hermite_matrix constructed by results")
    for i, (l, m) in enumerate(zip(triu_0, triu_1)):
        # hermite_matrix[l, m] = results_arr[i]
        hermite_matrix[l, m] = results[i]

    print(f"Save hermite_matrix as hermite_matrix to: {path_hermite_matrix}")
    with open(path_hermite_matrix, "wb") as g:
        np.savez_compressed(g, hermite_matrix=hermite_matrix)
    print(f"Succesfully saved")

    return E_H0, hermite_matrix


def get_hermite_matrix_linear(System, operator, dim, index_dict, cupy_used=True):
    hermite_matrix = np.zeros((dim, dim))
    E_H0 = np.zeros((dim, dim))

    hermite_lhy_abc_list = []
    dask = True
    # dask = False
    if dask:
        import dask.array as da
        hermite_matrix = da.from_array(hermite_matrix)
        E_H0 = da.from_array(E_H0)

    triu_0, triu_1 = np.triu_indices(dim)
    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)

    if cupy_used:
        operator = cp.asarray(operator)

    logging.info(psutil.virtual_memory())
    for l, m in zip(triu_0, triu_1):
        comb1 = index_dict[l]
        comb2 = index_dict[m]
        if l == m:
            # logging.info(f"hermite_transform: {l}, {m}")
            E_H0[l, m] = ((comb1[0] + 0.5)
                          + a_y * (comb1[1] + 0.5)
                          + a_z * (comb1[2] + 0.5))
        if not get_parity(comb1, comb2):
            comb1 = None
            comb2 = None
        if System.lhy_factor != 0:
            hermite_matrix_entry, hermite_lhy_abc_list = hermite_transform_with_lhy(System, operator, comb1, comb2, a_y, a_z, cupy_used=cupy_used)
        else:
            hermite_matrix_entry = hermite_transform_with_lhy(System, operator, comb1, comb2, a_y, a_z, cupy_used=cupy_used)
        hermite_matrix[l, m] = cp.asnumpy(hermite_matrix_entry)

    logging.info(psutil.virtual_memory())
    logging.info(f"hermite_matrix: {round(hermite_matrix.nbytes / 1024 / 1024,2)}MB")

    print(f"Save hermite_matrix as hermite_matrix to: {path_hermite_matrix}")
    if dask:
        hermite_matrix = np.array(hermite_matrix)
        with open(path_hermite_matrix, "wb") as g:
            np.savez_compressed(g, hermite_matrix=hermite_matrix)
    else:
        with open(path_hermite_matrix, "wb") as g:
            np.savez_compressed(g, hermite_matrix=hermite_matrix)
    print(f"Succesfully saved")

    return E_H0, hermite_matrix, hermite_lhy_abc_list

def get_hermite_dipol_entry(System, index_dict, dim, entry, ground_state=True, dask=False):
    with run_time(name=f"get_hermite_dipol_entry {entry}"):
        hermite_dipol_entry = np.zeros((dim, dim))
        beta_helper = None
        if dask:
            [System_dask] = client.scatter([System])
            futures = []
        j = 0
        for l in range(dim):
            gamma_helper = None
            for m in range(dim):
                j = j + 1
                comb1 = index_dict[l]
                comb2 = index_dict[m]
                # with run_time(name=f"hermite_dipol_entry {l}, {m}"):
                if dask:
                    with run_time(name=f"get_hermite_dipol_old {j}, ({l} {m})"):
                        futures.append(
                            client.submit(get_hermite_dipol_old,
                                          System_dask, comb1, comb2,
                                          ground_state=ground_state,
                                          entry=entry)
                            )
                else:
                    with run_time(name=f"get_hermite_dipol_old {j}, ({l} {m})"):
                        hermite_dipol_entry[l, m], beta_helper, gamma_helper = get_hermite_dipol_old(System, comb1, comb2,
                                                                                                    ground_state=ground_state,
                                                                                                    entry=entry,
                                                                                                    beta_helper=beta_helper,
                                                                                                    gamma_helper=gamma_helper
                                                                                                    )
        if dask:
            with run_time(name=f"hermite_dipol dask gather"):
                results = client.gather(futures)
            j = 0
            for l in range(dim):
                for m in range(dim):
                    j = j + 1
                    hermite_dipol_abcd_list[i][l, m] = results_arr[j]

    return hermite_dipol_entry


def get_hermite_matrix_dipol(System, index_dict, dim, ground_state=True):
    hermite_dipol_abcd_list = []
    with run_time(name=f"dipol all"):
        matrix_entries = ["A", "B", "C", "D"]
        for i, key in enumerate(matrix_entries):
            hermite_dipol_abcd_list.append(np.zeros((dim, dim)))
        with run_time(name=f"hermite_dipol"):
            for i, entry in enumerate(matrix_entries):
                with run_time(name=f"hermite_dipol {i}"):
                    hermite_dipol_abcd_list[i] = get_hermite_dipol_entry(System, index_dict, dim, entry, ground_state=ground_state)
                    logging.info(f"{i}:\n{hermite_dipol_abcd_list[i]}")

    return hermite_dipol_abcd_list

def get_hermite_matrix_dipol_dask(System, index_dict, dim, ground_state=True):
    client = Client() 

    hermite_dipol_abcd_list = []
    with run_time(name=f"dipol all dask"):
        matrix_entries = ["A", "B", "C", "D"]
        for i, _ in enumerate(matrix_entries):
            hermite_dipol_abcd_list.append(np.zeros((dim, dim)))
        [System_dask] = client.scatter([System])
        with run_time(name=f"hermite_dipol dask"):
            for i, entry in enumerate(matrix_entries):
                futures = []
                with run_time(name=f"hermite_dipol dask {i}"):
                    for l in range(dim):
                        for m in range(dim):
                            comb1 = index_dict[l]
                            comb2 = index_dict[m]
                            futures.append(
                                client.submit(get_hermite_dipol_old,
                                              System_dask, comb1, comb2,
                                              ground_state=ground_state,
                                              entry=entry)
                                )
                print(len(futures))
                with run_time(name=f"hermite_dipol dask gather"):
                    results = client.gather(futures)
                if cupy_used:
                    results_arr = cp.asnumpy(results)
                else:
                    results_arr = np.array(results)
                print(f"i: {i}")
                j = 0
                for l in range(dim):
                    for m in range(dim):
                        j = j + 1
                        hermite_dipol_abcd_list[i][l, m] = results_arr[j]

    return hermite_dipol_abcd_list

def get_lhy_term_linear(System, index, ground_state=True, cupy_used=True):
    lhy_abc_list = []
    density_list = System.get_density_list(jit=False, cupy_used=cupy_used)[0] / (System.N_list[0] ** 2)

    with run_time(name=f"lhy_abc_list"):
        if cupy_used:
            lhy_entry = cp.zeros((dim, dim))
        else:
            lhy_entry = np.zeros((dim, dim))

        if index == 0:
            lhy_entry = g_qf * 2.5 * density_raveled_by_N ** 1.5
        elif index == 1:
            lhy_entry = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled_by_N ** 2
        elif index == 2:
            if ground_state:
                print(f"ground_state: {ground_state}")
                lhy_entry = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled_by_N ** 2
            else:
                if cupy_used:
                    lhy_abc_list[2] = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * cp.conjugate(psi_val_raveled_by_N) ** 2
                else:
                    lhy_abc_list[2] = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * np.conjugate(psi_val_raveled_by_N) ** 2

    return lhy_abc_list

def get_lhy_terms(System, pos_v, dim, ground_state=True, cupy_used=True):
    lhy_entries = 3
    lhy_abc_list = []


    print(f"operator type: {density_list[0].dtype}")
    if ground_state:
        psi_val_raveled = operator_ravel(cp.array(cp.real(System.psi_val_list[0])), pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
    else:
        density_list = System.get_density_list(jit=False, cupy_used=cupy_used)
        if cupy_used:
            density_raveled = operator_ravel(cp.array(density_list[0]), pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
            psi_val_raveled = operator_ravel(cp.array(System.psi_val_list[0]), pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
        else:
            density_raveled = operator_ravel(density_list[0], pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
            psi_val_raveled = operator_ravel(System.psi_val_list[0], pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
        density_raveled_by_N = density_raveled / System.N_list[0]

    with run_time(name=f"lhy_abc_list"):
        for i in range(lhy_entries):
            # if cupy_used:
            #     lhy_abc_list.append(cp.zeros((dim, dim)))
            # else:
            lhy_abc_list.append(np.zeros((dim, dim)))

        if ground_state:
            print(f"ground_state: {ground_state}")
            lhy_abc_list[0] = cp.asnumpy(2.5 * psi_val_raveled ** 3.0)
            if cupy_used:
                # lhy_b = cp.asnumpy(cp.real(1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled ** 2.0))
                lhy_abc_list[1] = cp.asnumpy(1.5 * psi_val_raveled ** 3.0)
                lhy_abc_list[2] = lhy_abc_list[1]
            else:
                # lhy_b = np.real(1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled ** 2.0)
                lhy_abc_list[1] = np.real(1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled ** 2.0)
                lhy_abc_list[2] = lhy_abc_list[1]
        else:
            lhy_abc_list[0] = 2.5 * density_raveled_by_N ** 1.5
            if cupy_used:
                lhy_abc_list[1] = cp.asnumpy(1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled ** 2.0)
                lhy_abc_list[2] = 1.5 * (density_raveled_by_N ** 0.5) * cp.conjugate(psi_val_raveled) ** 2.0
            else:
                lhy_abc_list[1] = 1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled ** 2.0
                lhy_abc_list[2] = 1.5 * (density_raveled_by_N ** 0.5) * np.conjugate(psi_val_raveled) ** 2.0

    # logging.info(f"lhy_abc_list: {lhy_abc_list}")
    
    return lhy_abc_list

def cut_hermite_values_func(bog_helper, psi_raveled, min_cut_decimal_percent, quality_min, max_tries):
    dV = System.volume_element(fourier_space=False)

    if cupy_used:
        bog_helper = cp.array(bog_helper)
        psi_raveled = cp.array(psi_raveled)

    bog_helper_sum = cp.dot(bog_helper, psi_raveled) * dV

    logging.info(f"min bog_helper: {cp.min(bog_helper)}")
    logging.info(f"bog_helper_sum: {bog_helper_sum}")
    logging.info(f"bog_helper_sum.shape: {bog_helper_sum.shape}")
    quality_full = cp.sqrt(cp.sum(cp.abs(bog_helper_sum) ** 2.0))
    logging.info(f"quality_full: {quality_full}")
    dim_max = bog_helper_sum.shape[0] 
    ind_vec = cp.arange(0, dim_max, 1)

    for k in range(max_tries):
        min_cut_decimal_percent_new = min_cut_decimal_percent * 0.5 ** k
        cut_off = quality_full * min_cut_decimal_percent_new
        logging.info(f"min_cut_decimal_percent_new: {min_cut_decimal_percent_new}")

        cut_condition = cp.abs(bog_helper_sum) > cut_off
        ind_vec_cut = ind_vec[cut_condition]
        bog_helper_cut = bog_helper[cut_condition, :]

        bog_helper_sum_cut = cp.dot(bog_helper_cut, psi_raveled) * dV
        dim = bog_helper_sum_cut.shape[0]
        quality = cp.sqrt(cp.sum(cp.abs(bog_helper_sum_cut) ** 2.0))

        logging.info(f"quality: {quality}")
        logging.info(f"bog_helper_cut.shape: {bog_helper_cut.shape}")

        # done when maximal number of entries reached (given by polynomials dim ** 2 with dim = nx * ny * nz)
        if quality >= quality_min or (dim == dim_max):
            logging.info(f"Maximum number of entries for given n or high enough quality {quality}>={quality_min}")
            break

    if cupy_used:
        bog_helper = cp.asnumpy(bog_helper)
        bog_helper_cut = cp.asnumpy(bog_helper_cut)
        ind_vec_cut = cp.asnumpy(ind_vec_cut)

    # if k + 1 == max_tries:
    #     logging.info(f"Maximum number of tries to cut used ({max_tries}). No cut done!")
    #     logging.info(f"bog_helper.shape: {bog_helper.shape}")
    #     cut_condition = [True] * bog_helper.shape[0]
    #     return bog_helper, cut_condition
    # else:
    return bog_helper_cut, cut_condition, ind_vec_cut


def cut_hermite_values_func_2(bog_helper, bog_helper_operator_applied, min_cut_decimal_percent, quality_min, N):
    dV = System.volume_element(fourier_space=False)

    if cupy_used:
        bog_helper = cp.array(bog_helper)
        bog_helper_operator_applied = cp.array(bog_helper_operator_applied)
    #     bog_helper_sum = cp.einsum("ij,kj->i", bog_helper_operator_applied, bog_helper) * dV
    # else:
    #     bog_helper_sum = np.einsum("ij,kj->i", bog_helper_operator_applied, bog_helper) * dV

    dim_n = bog_helper.shape[0]
    dim_r = bog_helper.shape[1]
    bog_helper_sum = cp.zeros(dim_n)
    bog_helper_sum_cut = cp.zeros(dim_n)
    for i in range(dim_n):
        bog_helper_sum[i] = cp.dot(bog_helper[i, :], bog_helper_operator_applied[i, :]) * dV
    # bog_helper_sum = cp.sum(bog_helper_operator_applied, axis=-1) * dV

    # logging.info(f"bog_helper_lol: {bog_helper_lol}")
    # logging.info(f"bog_helper_lol.shape: {bog_helper_lol.shape}")
    # logging.info(f"bog_helper: {bog_helper}")
    # logging.info(f"bog_helper_operator_applied: {bog_helper_operator_applied}")
    logging.info(f"min bog_helper: {cp.min(bog_helper)}")
    logging.info(f"min bog_helper_operator_applied: {cp.min(bog_helper_operator_applied)}")
    logging.info(f"bog_helper_sum: {bog_helper_sum}")
    logging.info(f"bog_helper_sum.shape: {bog_helper_sum.shape}")
    quality_full = cp.abs(bog_helper_sum).sum() / N
    logging.info(f"quality_full: {quality_full}")
    logging.info(f"N: {N}")

    max_tries = 20
    for k in range(max_tries):
        min_cut_decimal_percent_new = min_cut_decimal_percent * 0.5 ** k
        sorted_cut_off = quality_full * min_cut_decimal_percent_new
        logging.info(f"min_cut_decimal_percent_new: {min_cut_decimal_percent_new}")
        # logging.info(f"sorted_cut_off: {sorted_cut_off}")
        cut_condition = cp.abs(bog_helper_sum) > sorted_cut_off
        # logging.info(f"cut_condition: {cut_condition}")
        # logging.info(f"cut_condition.shape: {cut_condition.shape}")

        bog_helper_sorted_cut = bog_helper[cut_condition, :]
        bog_helper_operator_applied_sorted_cut = bog_helper_operator_applied[cut_condition, :]

        # for i in range(bog_helper_sorted_cut.shape[0]):
        #     bog_helper_sum_cut[i] = cp.dot(bog_helper_sorted_cut[i, :], bog_helper_operator_applied_sorted_cut[i, :]) * dV
        bog_helper_sum_cut = cp.sum(bog_helper_operator_applied_sorted_cut, axis=-1) * dV

        quality = (cp.abs(bog_helper_sum_cut).sum() / N)
        logging.info(f"quality: {quality}")
        logging.info(f"bog_helper_operator_applied_sorted_cut.shape: {bog_helper_operator_applied_sorted_cut.shape}")

        # done when maximal number of entries reached (given by polynomials dim ** 2 with dim = nx * ny * nz)
        if quality >= quality_min or (bog_helper_sum_cut.shape[0] == bog_helper_sum.shape[0]):
            logging.info(f"Maximum number of entries for given n or high enough quality {quality}>={quality_min}")
            break

        # logging.info(f"bog_helper_sorted_cut: {bog_helper_sorted_cut}")
        # logging.info(f"bog_helper_sorted_cut.shape: {bog_helper_sorted_cut.shape}")

    # if cupy_used:
    #     index_sorted = cp.argsort(-bog_helper_sum, axis=-1)
    # else:
    #     index_sorted = np.argsort(-bog_helper_sum, axis=-1)
    # logging.info(f"index_sorted.shape: {index_sorted.shape}")
    # logging.info(f"index_sorted.dtype: {index_sorted.dtype}")
    # logging.info(f"cut_condition.dtype: {cut_condition.dtype}")
    # # logging.info(f"index_sorted: {index_sorted}")
    # index_sorted_cut = index_sorted[cut_condition]
    # logging.info(f"index_sorted_cut: {index_sorted_cut}")
    # logging.info(f"index_sorted_cut.shape: {index_sorted_cut.shape}")
    # bog_helper_sorted_cut = bog_helper[index_sorted_cut, :]
    # bog_helper_operator_applied_sorted_cut = bog_helper_operator_applied[index_sorted_cut, :]

    if cupy_used:
        bog_helper = cp.asnumpy(bog_helper)
        bog_helper_operator_applied = cp.asnumpy(bog_helper_operator_applied)
        bog_helper_sorted_cut = cp.asnumpy(bog_helper_sorted_cut)
        bog_helper_operator_applied_sorted_cut = cp.asnumpy(bog_helper_operator_applied_sorted_cut)

    # logging.info(f"bog_helper_sorted_cut: {bog_helper_sorted_cut}")
    # logging.info(f"bog_helper_operator_applied_sorted_cut: {bog_helper_operator_applied_sorted_cut}")
    # logging.info(f"bog_helper_sorted_cut.shape: {bog_helper_sorted_cut.shape}")
    # logging.info(f"bog_helper_operator_applied_sorted_cut.shape: {bog_helper_operator_applied_sorted_cut.shape}")

    if k + 1 == max_tries:
        logging.info(f"Maximum number of tries to cut used ({max_tries}). No cut done!")
        logging.info(f"bog_helper.shape: {bog_helper.shape}")
        logging.info(f"bog_helper_operator_applied.shape: {bog_helper_operator_applied.shape}")
        return bog_helper, bog_helper_operator_applied
    else:
        return bog_helper_sorted_cut, bog_helper_operator_applied_sorted_cut


def get_hermite_matrix_fft(System, operator, nx, ny, nz, mu):
    save_gpu_mem = False
    # save_gpu_mem = True

    dim = nx * ny * nz
    # dim = System.Res.x * System.Res.y * System.Res.z
    dim_r = System.Res.x * System.Res.y * System.Res.z
    hermite_lhy_abc_list = []
    pos_vec = np.arange(0, dim_r, 1)

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    ind_kx = np.abs(System.kx).argsort()
    ind_ky = np.abs(System.ky).argsort()
    ind_kz = np.abs(System.kz).argsort()
    ind_cut_kx = np.sort(ind_kx[:nx])
    ind_cut_ky = np.sort(ind_ky[:ny])
    ind_cut_kz = np.sort(ind_kz[:nz])
    # print(f"ind_cut: {ind_cut}")

    print(f"ind_kx.shape: {ind_kx.shape}")
    print(f"ind_kx: {ind_kx}")

    print(f"ind_cut_kx.shape: {ind_cut_kx.shape}")
    print(f"ind_cut_kx: {ind_cut_kx}")

    kx_cut = System.kx[ind_cut_kx]
    ky_cut = System.ky[ind_cut_ky]
    kz_cut = System.kz[ind_cut_kz]
    # print(f"kx_cut: {kx_cut}")

    print(f"System.kx: {System.kx}")
    print(f"kx_cut: {kx_cut}")
    print(f"System.kx[-ind_cut_kx]: {System.kx[-ind_cut_kx]}")

    save_gpu_mem = False
    if save_gpu_mem:
        kx_mesh_cut, ky_mesh_cut, kz_mesh_cut = np.meshgrid(kx_cut, ky_cut, kz_cut, indexing="ij", sparse=True)
        a = cp.asnumpy(0.5 * np.diag(np.ravel(kx_mesh_cut ** 2.0 + ky_mesh_cut ** 2.0 + kz_mesh_cut ** 2.0))) - np.diag(dim * [np.array(mu)])
    else:
        kx_mesh_cut, ky_mesh_cut, kz_mesh_cut = cp.meshgrid(cp.array(kx_cut), cp.array(ky_cut), cp.array(kz_cut), indexing="ij", sparse=True)
        # overwrite too save memory
        # kx_mesh_cut = 0.5 * cp.ravel(kx_mesh_cut ** 2.0 + ky_mesh_cut ** 2.0 + kz_mesh_cut ** 2.0)
        kx_mesh_cut = cp.asnumpy(0.5 * cp.ravel(kx_mesh_cut ** 2.0 + ky_mesh_cut ** 2.0 + kz_mesh_cut ** 2.0))
        a = np.diag(kx_mesh_cut) - np.diag(dim * [np.array(mu)])
        # a = cp.asnumpy(cp.diag(kx_mesh_cut)) - np.diag(dim * [np.array(mu)])
        # a = cp.asnumpy(0.5 * cp.diag(cp.ravel(kx_mesh_cut ** 2.0 + ky_mesh_cut ** 2.0 + kz_mesh_cut ** 2.0))) - np.diag(dim * [np.array(mu)])
    # save_gpu_mem = True

    kx_mesh_cut, ky_mesh_cut, kz_mesh_cut = None, None, None

    if save_gpu_mem:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if cupy_used:
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")


    ind_cut_kx_mesh, ind_cut_ky_mesh, ind_cut_kz_mesh = cp.meshgrid(cp.array(ind_cut_kx), cp.array(ind_cut_ky), cp.array(ind_cut_kz), indexing="ij", sparse=False)
    ind_cut_kx_mesh_ravel = cp.ravel(ind_cut_kx_mesh)
    print(f"ind_cut_kx_mesh_ravel.shape: {ind_cut_kx_mesh_ravel.shape}")
    ind_cut_kx_mesh = None
    ind_cut_ky_mesh_ravel = cp.ravel(ind_cut_ky_mesh)
    ind_cut_ky_mesh = None
    ind_cut_kz_mesh_ravel = cp.ravel(ind_cut_kz_mesh)
    ind_cut_kz_mesh = None

    if save_gpu_mem:
        q_cut_ind_kx = cp.asnumpy(ind_cut_kx_mesh_ravel) - cp.asnumpy(ind_cut_kx_mesh_ravel)[:, cp.newaxis]
        ind_cut_kx_mesh_ravel = None
        q_cut_ind_ky = cp.asnumpy(ind_cut_ky_mesh_ravel) - cp.asnumpy(ind_cut_ky_mesh_ravel)[:, cp.newaxis]
        ind_cut_ky_mesh_ravel = None
        q_cut_ind_kz = cp.asnumpy(ind_cut_kz_mesh_ravel) - cp.asnumpy(ind_cut_kz_mesh_ravel)[:, cp.newaxis]
        ind_cut_kz_mesh_ravel = None
    else:
        q_cut_ind_kx = ind_cut_kx_mesh_ravel - ind_cut_kx_mesh_ravel[:, cp.newaxis]
        ind_cut_kx_mesh_ravel = None
        q_cut_ind_ky = ind_cut_ky_mesh_ravel - ind_cut_ky_mesh_ravel[:, cp.newaxis]
        ind_cut_ky_mesh_ravel = None
        q_cut_ind_kz = ind_cut_kz_mesh_ravel - ind_cut_kz_mesh_ravel[:, cp.newaxis]
        ind_cut_kz_mesh_ravel = None

    # q_cut_ind_kx = ind_cut_kx_mesh_ravel - ind_cut_kx_mesh_ravel.reshape(dim, 1)
    # q_cut_ind_ky = ind_cut_ky_mesh_ravel - ind_cut_ky_mesh_ravel.reshape(dim, 1)
    # q_cut_ind_kz = ind_cut_kz_mesh_ravel - ind_cut_kz_mesh_ravel.reshape(dim, 1)
    print(f"q_cut_ind_kx.shape: {q_cut_ind_kx.shape}")
    print(f"q_cut_ind_ky.shape: {q_cut_ind_ky.shape}")
    print(f"q_cut_ind_kz.shape: {q_cut_ind_kz.shape}")

    if save_gpu_mem:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if cupy_used:
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    l_0 = np.sqrt(constants.hbar / (System.m_list[0] * System.w_x))
    logging.info(f"l_0: {l_0}")
    logging.info(f"q_cut_ind_kx.shape: {q_cut_ind_kx.shape}")

    if System.a_dd_array[0, 0] != 0:
        logging.info(f"Dipols used, beta")

        if save_gpu_mem:
            psi_val = cp.real(cp.array(System.psi_val_list[0]))
            beta = (System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0]
                   * np.real(System.V_k_val * cp.asnumpy(cp.fft.fftn(cp.abs(psi_val) ** 2.0, norm="forward")))[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]
                   )
            V_k_val = cp.array(System.V_k_val)
            check_2 = System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * cp.sum(cp.real(cp.fft.ifftn(cp.array(System.V_k_val) * cp.fft.fftn(cp.abs(psi_val) ** 2.0)))) / dim_r
        else:
            psi_val = cp.real(cp.array(System.psi_val_list[0]))
            V_k_val = cp.array(System.V_k_val)
            beta = (System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0]
                   * cp.real(V_k_val * cp.fft.fftn(cp.abs(psi_val) ** 2.0, norm="forward"))[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]
                   )
            check_2 = System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * cp.sum(cp.real(cp.fft.ifftn(cp.array(System.V_k_val) * cp.fft.fftn(cp.abs(psi_val) ** 2.0)))) / dim_r

        # if save_gpu_mem:
        #     psi_val = cp.real(cp.array(System.psi_val_list[0]))
        #     beta = (System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0]
        #            * np.real(System.V_k_val[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]
        #            * cp.asnumpy(cp.fft.fftn(cp.abs(psi_val) ** 2.0, norm="forward"))[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz])
        #            )
        #     V_k_val = cp.array(System.V_k_val)
        #     check_2 = System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * cp.sum(cp.real(cp.fft.ifftn(cp.array(System.V_k_val) * cp.fft.fftn(cp.abs(psi_val) ** 2.0)))) / dim_r
        # else:
        #     psi_val = cp.real(cp.array(System.psi_val_list[0]))
        #     V_k_val = cp.array(System.V_k_val)
        #     beta = (System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0]
        #            * cp.real(V_k_val[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]
        #            * cp.fft.fftn(cp.abs(psi_val) ** 2.0, norm="forward")[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz])
        #            )
        #     check_2 = System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * cp.sum(cp.real(cp.fft.ifftn(cp.array(System.V_k_val) * cp.fft.fftn(cp.abs(psi_val) ** 2.0)))) / dim_r

        psi_val = cp.asnumpy(psi_val)
        V_k_val = cp.asnumpy(V_k_val)

        logging.info(f"min V_k_val: {cp.min(V_k_val)}")
        logging.info(f"max V_k_val: {cp.max(V_k_val)}")

        check_1 = cp.sum(beta) / dim_r
        logging.info(f"check_1: {check_1}")
        logging.info(f"check_2: {check_2}")
        logging.info(f"check_2 / check_1: {check_2 / check_1}")
        a += cp.asnumpy(beta)
        logging.info(f"beta.shape: {beta.shape}")
        logging.info(f"beta.dtype: {beta.dtype}")
        logging.info(f"max beta: {cp.max(beta)}")
        logging.info(f"beta:\n{beta}")
        beta = None

    if save_gpu_mem:
        V_val_fft = cp.asnumpy(cp.real(cp.fft.fftn(cp.array(System.V_val), norm="forward")))
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    else:
        V_val_fft = cp.real(cp.fft.fftn(cp.array(System.V_val), norm="forward"))

    if cupy_used:
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    V_bog = V_val_fft[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]
    V_val_fft = None

    if not save_gpu_mem:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    a += cp.asnumpy(V_bog)
    V_bog = None


    if cupy_used:
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    dV = System.volume_element(fourier_space=False)
    logging.info(f"dV:\n{dV}")
    g = 4.0 * np.pi * System.a_s_array[0, 0]
    logging.info(f"g: {g}")

    op_fft = cp.real(cp.fft.fftn(operator, norm="forward"))

    if save_gpu_mem:
        op_fft = cp.asnumpy(op_fft)
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if cupy_used:
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    buffer_b = cp.asnumpy(g * op_fft[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz])
    b = np.copy(buffer_b)
    a += 2.0 * buffer_b

    buffer_b = None
    op_fft = None
    logging.info(f"a.shape: {a.shape}")
    logging.info(f"a: {a}")

    if cupy_used:
        psi_val = cp.real(cp.array(System.psi_val_list[0]))
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    if System.lhy_factor != 0:
        print(f"System.lhy_factor: {System.lhy_factor}")
        g_qf = functions.get_g_qf_bog(N=System.N_list[0],
                                      a_s=float(System.a_s_array[0, 0]),
                                      a_dd=float(System.a_dd_array[0, 0]))
        print(f"g_qf: {g_qf}")
        if reduced_version:
            hermite_lhy_base = g_qf * cp.asnumpy(cp.real(cp.fft.fftn(psi_val ** 3.0, norm="forward")))[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]
        else:
            sys.exit("Not implemented!")
        logging.info(f"hermite_lhy_base:\n{hermite_lhy_base}")
        a += 2.5 * hermite_lhy_base
        b += 1.5 * hermite_lhy_base

    # Check the norm in k space
    if cupy_used:
        psi_cut = cp.asnumpy(cp.sqrt(dim) * cp.ravel(cp.fft.fftn(psi_val, norm="forward").take(ind_cut_kx, axis=0).take(ind_cut_ky, axis=1).take(ind_cut_kz, axis=2)))
        if System.a_dd_array[0, 0] == 0:
            # psi_val not needed anymore
            psi_val = None
    else:
        psi_cut = np.sqrt(dim) * np.ravel(np.fft.fftn(psi_val, norm="forward").take(ind_cut_kx, axis=0).take(ind_cut_ky, axis=1).take(ind_cut_kz, axis=2))

    psi_cut_norm = np.dot(psi_cut, psi_cut)
    logging.info(f"psi_cut_norm: {psi_cut_norm}")
    # watch aout a and b do not have gamma here
    a_kspace = np.abs(np.dot(a, psi_cut))
    b_kspace = np.abs(np.dot(b, psi_cut))
    a_b_minus_kspace = np.abs(np.dot(a, psi_cut) + np.dot(b, psi_cut))
    a_b_plus_kspace = np.abs(np.dot(a, psi_cut) - np.dot(b, psi_cut))
    logging.info(f"a_b_minus_kspace: {a_b_minus_kspace}")
    logging.info(f"a_b_minus_kspace max: {np.max(a_b_minus_kspace)}")
    logging.info(f"a_b_plus_kspace: {a_b_plus_kspace}")
    logging.info(f"a_b_plus_kspace max: {np.max(a_b_plus_kspace)}")
    logging.info(f"a_kspace: {a_kspace}")
    logging.info(f"a_kspace max: {np.max(a_kspace)}")
    logging.info(f"b_kspace: {b_kspace}")
    logging.info(f"b_kspace max: {np.max(b_kspace)}")
    # mat2d(check_ground_state, label="check_ground_state:")

    test00 = (a - b) * (a + b)
    test01 = (a - b) @ (a + b)
    test02 = (a - b)
    test03 = (a + b)
    test04 = a
    test05 = b
    is_hermite06 = (a - b) @ (a + b) - (a + b) @ (a - b)
    is_hermite07 = (a - b) * (a + b) - (a + b) * (a - b)
    is_hermite00 = test00 - np.conjugate(test00).T
    is_hermite01 = test01 - np.conjugate(test01).T
    is_hermite02 = test02 - np.conjugate(test02).T
    is_hermite03 = test03 - np.conjugate(test03).T
    is_hermite04 = test04 - np.conjugate(test04).T
    is_hermite05 = test05 - np.conjugate(test05).T
    logging.info(f"is_hermite00: {is_hermite00}")
    logging.info(f"is_hermite00 min : {np.min(is_hermite00)}")
    logging.info(f"is_hermite00 max : {np.max(is_hermite00)}")
    logging.info(f"is_hermite01: {is_hermite01}")
    logging.info(f"is_hermite01 min : {np.min(is_hermite01)}")
    logging.info(f"is_hermite01 max : {np.max(is_hermite01)}")
    logging.info(f"is_hermite02: {is_hermite02}")
    logging.info(f"is_hermite02 min : {np.min(is_hermite02)}")
    logging.info(f"is_hermite02 max : {np.max(is_hermite02)}")
    logging.info(f"is_hermite03: {is_hermite03}")
    logging.info(f"is_hermite03 min : {np.min(is_hermite03)}")
    logging.info(f"is_hermite03 max : {np.max(is_hermite03)}")
    logging.info(f"is_hermite04: {is_hermite04}")
    logging.info(f"is_hermite04 min : {np.min(is_hermite04)}")
    logging.info(f"is_hermite04 max : {np.max(is_hermite04)}")
    logging.info(f"is_hermite05: {is_hermite05}")
    logging.info(f"is_hermite05 min : {np.min(is_hermite05)}")
    logging.info(f"is_hermite05 max : {np.max(is_hermite05)}")
    logging.info(f"is_hermite06: {is_hermite06}")
    logging.info(f"is_hermite06 min : {np.min(is_hermite06)}")
    logging.info(f"is_hermite06 max : {np.max(is_hermite06)}")
    logging.info(f"is_hermite07: {is_hermite07}")
    logging.info(f"is_hermite07 min : {np.min(is_hermite07)}")
    logging.info(f"is_hermite07 max : {np.max(is_hermite07)}")

    if System.a_dd_array[0, 0] != 0:
        logging.info(f"Dipols used")
        psi_val = cp.real(cp.array(System.psi_val_list[0]))
        psi_val_raveled = cp.ravel(cp.array(System.psi_val_list[0]))
        V_k_val = cp.array(System.V_k_val)

        if cupy_used:
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")


        # ind_kx_mesh, ind_ky_mesh, ind_kz_mesh = cp.meshgrid(cp.array(ind_kx), cp.array(ind_ky), cp.array(ind_kz), indexing="ij", sparse=False)
        # logging.info(f"ind_kx_mesh.shape: {ind_kx_mesh.shape}")

        ## Get norm in k space
        psi_fft = cp.fft.fftn(psi_val)
        ind_kx_mesh_to_0 = cp.array(np.delete(cp.asnumpy(ind_kx), cp.asnumpy(ind_cut_kx).astype(int))).reshape(System.Res.x - nx, 1, 1)
        ind_ky_mesh_to_0 = cp.array(np.delete(cp.asnumpy(ind_ky), cp.asnumpy(ind_cut_ky).astype(int))).reshape(1, System.Res.y - ny, 1)
        ind_kz_mesh_to_0 = cp.array(np.delete(cp.asnumpy(ind_kz), cp.asnumpy(ind_cut_kz).astype(int))).reshape(1, 1, System.Res.z - nz)
        logging.info(f"ind_kx_mesh_to_0 shape: {ind_kx_mesh_to_0.shape}")
        psi_fft[ind_kx_mesh_to_0, ind_ky_mesh_to_0, ind_kz_mesh_to_0] = 0.0
        psi_fft = cp.ravel(cp.real(cp.fft.ifftn(psi_fft.reshape(System.Res.x, System.Res.y, System.Res.z))))
        norm_in_k_space = cp.dot(psi_fft, cp.real(psi_val_raveled)) * dV
        logging.info(f"norm_in_k_space: {norm_in_k_space}")
        psi_fft = None

        ind_kx_mesh_to_0 = None
        ind_ky_mesh_to_0 = None
        ind_kz_mesh_to_0 = None

        if cupy_used:
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        gamma_helper = cp.real(cp.fft.fftn(psi_val * cp.fft.ifftn(V_k_val * cp.fft.fftn(psi_val)),
                                           norm="forward"
                                           )
                               )

        if save_gpu_mem:
            gamma_helper = cp.asnumpy(gamma_helper)
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        if cupy_used:
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")


        gamma = System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * gamma_helper[q_cut_ind_kx, q_cut_ind_ky, q_cut_ind_kz]

        logging.info(f"psi_val_raveled.shape: {psi_val_raveled.shape}")
        logging.info(f"V_k_val.shape: {V_k_val.shape}")
        logging.info(f"System.a_dd_array[0, 0]: {System.a_dd_array[0, 0]}")
        logging.info(f"System.N_list[0]: {System.N_list[0]}")
        logging.info(f"System.a_dd_factor: {System.a_dd_factor}")
        logging.info(f"System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0]: {System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0]}")

        logging.info(f"gamma.shape: {gamma.shape}")
        logging.info(f"gamma.dtype: {gamma.dtype}")
        logging.info(f"max gamma: {cp.max(gamma)}")
        logging.info(f"gamma:\n{gamma}")
        # a += cp.asnumpy(gamma + beta)
        a += cp.asnumpy(gamma)
        b += cp.asnumpy(gamma)
        gamma = None

    return a, b


def get_hermite_matrix_flat(System, operator, nx, ny, nz, ind_vec_dict=None, cut_hermite_orders=False,
                            cut_hermite_values=False):
    # dask = True
    dask = False
    if dask:
        logging.info(f"dask_flat")

    hermite_lhy_abc_list = []
    logging.info(psutil.virtual_memory())
    pos_max = System.Res.x * System.Res.y * System.Res.z
    logging.info(psutil.virtual_memory())
    pos_vec = np.arange(0, pos_max, 1)

    if ind_vec_dict is None:
        dim = int(nx * ny * nz)
        ind_vec = np.arange(0, dim, 1)
    else:
        # ind_vec_dict = get_index_arr(nx, ny, nz, max_sum=(nx + ny + nz) // 2)
        dim = len(ind_vec_dict)
        ind_vec = np.arange(0, dim, 1)
        ind_x = ind_vec_dict[:, 0]
        ind_y = ind_vec_dict[:, 1]
        ind_z = ind_vec_dict[:, 2]
        ind_x_n, _ = np.meshgrid(ind_x, pos_vec, indexing='ij', sparse=True)
        ind_y_n, _ = np.meshgrid(ind_y, pos_vec, indexing='ij', sparse=True)
        ind_z_n, _ = np.meshgrid(ind_z, pos_vec, indexing='ij', sparse=True)

    logging.info(psutil.virtual_memory())
    logging.info(f"pos_vec: {round(pos_vec.nbytes / 1024 / 1024,2)}MB")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    logging.info("Create meshgrid")
    logging.info(psutil.virtual_memory())

    dV = System.volume_element(fourier_space=False)

    if dask:
        logging.info(f"dask_flat")
        ind_v, pos_v = da.meshgrid(ind_vec, pos_vec, indexing='ij', sparse=True)
        logging.info(f"ind_v: {round(ind_v.nbytes / 1024 / 1024,2)}MB")
        logging.info(f"pos_v: {round(pos_v.nbytes / 1024 / 1024,2)}MB")
        logging.info(psutil.virtual_memory())

        hermite_matrix = da.zeros((dim, dim))
        operator = da.array(operator)
        x, y, z = position(System, pos_v) 
        x = da.array(x)
        y = da.array(y)
        z = da.array(z)
        logging.info(psutil.virtual_memory())
        logging.info(f"x: {round(x.nbytes / 1024 / 1024,2)}MB")
    else:
        ind_v, pos_v = np.meshgrid(ind_vec, pos_vec, indexing='ij', sparse=True)
        # logging.info(f"ind_vec: {ind_vec}")
        logging.info(f"ind_vec.shape: {ind_vec.shape}")
        logging.info(f"ind_v.shape: {ind_v.shape}")
        # logging.info(f"ind_v: {ind_v}")
        logging.info(f"pos_v: {pos_v}")
        logging.info(f"pos_v.shape: {pos_v.shape}")
        logging.info(f"ind_v: {round(ind_v.nbytes / 1024 / 1024,2)}MB")
        logging.info(f"pos_v: {round(pos_v.nbytes / 1024 / 1024,2)}MB")
        logging.info(psutil.virtual_memory())

        x, y, z = position(System, pos_v) 
        logging.info(psutil.virtual_memory())
        logging.info(f"x: {round(x.nbytes / 1024 / 1024,2)}MB")
        logging.info(f"x: {x}")
        logging.info(f"x.shape: {x.shape}")
        logging.info(f"y: {y}")
        logging.info(f"y.shape: {y.shape}")
        logging.info(f"z: {z}")
        logging.info(f"z.shape: {z.shape}")
        # logging.info(f"x == System.x_mesh: {np.all(x == System.x_mesh)}")

    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    try:
        logging.info(f"Trying to load bog_helper from: {path_bog_helper}")
        with open(path_bog_helper, "rb") as g:
            bog_helper_load = np.load(file=g)
            bog_helper = bog_helper_load["bog_helper"]
            bog_helper = cp.asnumpy(bog_helper)
    except Exception as e:
        logging.info(f"File not found. Calculating bog_helper")
        with run_time(name=f"bog_helper {nx} {ny} {nz}"):
            if ind_vec_dict is None:
                logging.info(f"Calculated bog_helper (HO_3D_old)")
                # bog_helper = HO_3D_old(System.x, System.y, System.z, ind_v, ny, nz, a_y=a_y, a_z=a_z, cupy_used=False)
                bog_helper = HO_3D_old(x, y, z, ind_v, ny, nz, a_y=a_y, a_z=a_z, cupy_used=False)
            else:
                logging.info(f"Calculated bog_helper (HO_3D)")
                bog_helper = HO_3D(x, y, z, ind_x_n, ind_y_n, ind_z_n, a_y=a_y, a_z=a_z, cupy_used=False)
        logging.info(f"Calculated bog_helper")

        if not dask:
            bog_helper = cp.asnumpy(bog_helper)
            logging.info(f"Converted bog_helper to numpy")

        if not dask:
            logging.info(f"Save bog_helper as bog_helper to: {path_bog_helper}")
            with open(path_bog_helper, "wb") as g:
                np.savez_compressed(g, bog_helper=bog_helper)
            logging.info(f"Succesfully saved bog_helper")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    # mat_check(bog_helper, "bog_helper:")
    # mat2d(bog_helper, "bog_helper:")

    logging.info(f"Calculate operator_raveled")
    try:
        # bring operator in same order as x
        with run_time(name=f"operator_raveled {nx} {ny} {nz}"):
            operator_raveled = operator_ravel(operator, pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
        operator = None
    except Exception as e:
        logging.info(f"Failed. Trying to use cp.asnumpy")
        with run_time(name=f"operator_raveled {nx} {ny} {nz}"):
            operator_raveled = operator_ravel(cp.asnumpy(operator), pos_v, System.Res.y, System.Res.z, cupy_used=False)
        operator = None

    logging.info(f"operator_raveled.shape: {operator_raveled.shape}")
    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    if not dask:
        operator_raveled = cp.asnumpy(operator_raveled)

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    # TODO: bog_helper 25GB for 32 ** 3, so bog_helper, operator_raveled, bog_helper_operator_applied = 75GB
    print(f"Calculate bog_helper_operator_applied")
    with run_time(name=f"bog_helper_operator_applied {nx} {ny} {nz}"):
        bog_helper_operator_applied = bog_helper * operator_raveled

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    logging.info(f"bog_helper.shape: {bog_helper.shape}")
    logging.info(f"bog_helper.dtype: {bog_helper.dtype}")
    logging.info(f"bog_helper_operator_applied.dtype: {bog_helper_operator_applied.dtype}")
    logging.info(f"bog_helper_operator_applied.shape: {bog_helper_operator_applied.shape}")
    # logging.info(f"bog_helper: {bog_helper}")
    # logging.info(f"bog_helper_operator_applied: {bog_helper_operator_applied}")

    if cut_hermite_values:
        logging.info(f"Cut bog_helper entries by value")
        min_cut_decimal_percent = 10 ** -9
        # min_cut_decimal_percent = 10 ** -1
        quality_min = 0.99999
        max_tries = 18

        if cupy_used:
            with run_time(name=f"psi_raveled {nx} {ny} {nz}"):
                psi_raveled = operator_ravel(cp.asnumpy(cp.real(System.psi_val_list[0])), pos_v, System.Res.y, System.Res.z, cupy_used=False)
            bog_helper, cut_condition, ind_vec = cut_hermite_values_func(bog_helper, psi_raveled, min_cut_decimal_percent, quality_min, max_tries)
            bog_helper_operator_applied = bog_helper_operator_applied[cp.asnumpy(cut_condition), :]
            ind_v, pos_v = np.meshgrid(ind_vec, pos_vec, indexing='ij', sparse=True)

            # bog_helper, bog_helper_operator_applied = cut_hermite_values_func_2(bog_helper, bog_helper_operator_applied, min_cut_decimal_percent, quality_min, System.N_list[0])
        dim = bog_helper.shape[0]

    print(f"Calculate bog_helper_swapped")
    try:
        if dask:
            bog_helper_swapped = da.swapaxes(bog_helper, 0, 1)
        else:
            bog_helper_swapped = cp.swapaxes(bog_helper, 0, 1)
    except Exception as e:
        print(f"Failed. Trying to use np.swapaxes.")
        bog_helper_swapped = np.swapaxes(bog_helper, 0, 1)

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    logging.info(f"Calculate hermite_matrix")
    with run_time(name=f"flat hermite_matrix dot {dim}"):
        try:
            if dask:
                logging.info(f"Calculate hermite_matrix with dask dot")
                hermite_matrix = da.dot(bog_helper_operator_applied, bog_helper_swapped) * dV
            else:
                logging.info(f"Calculate hermite_matrix with cupy dot")
                hermite_matrix = cp.dot(bog_helper_operator_applied, bog_helper_swapped) * dV
                # logging.info(f"hermite_matrix: {hermite_matrix}")
                # cp.set_printoptions(precision=cp.inf)
                # logging.info(f"diag hermite_matrix: {cp.diag(hermite_matrix)}")
        except Exception as e:
            logging.info(f"WARNING: GPU not used for calculation of integrals")
            logging.info(f"Failed. Trying to use cp.asnumpy")
            hermite_matrix = np.dot(cp.asnumpy(bog_helper_operator_applied), cp.asnumpy(bog_helper_swapped)) * dV
            # hermite_matrix = cp.array(hermite_matrix)

    # mat_check(cp.asnumpy(hermite_matrix), "hermite_matrix:")
    # mat2d(hermite_matrix, "hermite_matrix:")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    print(f"System.lhy_factor: {System.lhy_factor}")
    if System.lhy_factor != 0:
        try:
            logging.info(f"Trying to load hermite_lhy_abc_list from: {path_hermite_lhy_abc_list}")
            with open(path_hermite_lhy_abc_list, "rb") as g:
                hermite_lhy_abc_list_load = np.load(file=g)
                hermite_lhy_abc_list = hermite_lhy_abc_list_load["hermite_lhy_abc_list"]
        except Exception as e:
            logging.info(f"Failed to load hermite_lhy_abc_list")
            try:
                if reduced_version:
                    hermite_lhy_abc_list = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper,
                                                                    bog_helper_swapped, cupy_used=cupy_used)
                else:
                    hermite_lhy_abc_list = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper, bog_helper_swapped, cupy_used=cupy_used)
                bog_helper = None

                logging.info(f"Save hermite_lhy_abc_list as hermite_lhy_abc_list to: {path_hermite_lhy_abc_list}")
                with open(path_hermite_lhy_abc_list, "wb") as g:
                    if dask:
                        np.savez_compressed(g, hermite_lhy_abc_list=[np.array(hermite_lhy_i) for hermite_lhy_i in hermite_lhy_abc_list])
                    else:
                        np.savez_compressed(g, hermite_lhy_abc_list=hermite_lhy_abc_list)
                logging.info(f"Succesfully saved")

            except Exception as e:
                print(f"Failed. Trying to use cp.asnumpy")
                traceback.print_tb(e.__traceback__)
                hermite_lhy_abc_list = get_hermite_lhy_abc_list(System, pos_v, dim, cp.asnumpy(bog_helper), cp.asnumpy(bog_helper_swapped), cupy_used=False)
                bog_helper = None
    else:
        bog_helper = None

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    logging.info(f"Save hermite_matrix as hermite_matrix to: {path_hermite_matrix}")
    with open(path_hermite_matrix, "wb") as g:
        if dask:
            np.savez_compressed(g, hermite_matrix=np.array(hermite_matrix))
        else:
            np.savez_compressed(g, hermite_matrix=hermite_matrix)
    logging.info(f"Succesfully saved")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    put_0_after_cut = False
    if put_0_after_cut:
        dim = int(nx * ny * nz)
        # hermite_matrix_bigger = np.zeros((dim, dim))
        print(f"ind_vec: {ind_vec}")
        print(f"len ind_vec: {len(ind_vec)}")

        # indices_with_0 = [i for i in range(dim) if i not in ind_vec]
        # hermite_matrix = np.insert(hermite_matrix, indices_with_0, 0, axis=0)
        # hermite_matrix = np.insert(hermite_matrix, indices_with_0, 0, axis=1)

        for i in range(dim):
            if i not in ind_vec:
                # print(f"i: {i}")
                # if i < len(ind_vec):
                    # print(f"ind: {ind_vec[i]}")
                hermite_matrix = np.insert(hermite_matrix, i, 0, axis=0)
                hermite_matrix = np.insert(hermite_matrix, i, 0, axis=1)
        print(f"New hermite shape: {hermite_matrix.shape}")

        ind_vec = np.arange(0, dim, 1)
        # ind_v, pos_v = np.meshgrid(ind_vec, pos_vec, indexing='ij', sparse=True)
        E_H0 = np.diag(En(ind_vec, ny, nz, a_y=a_y, a_z=a_z))
    else:
        E_H0 = np.diag(En(ind_vec, ny, nz, a_y=a_y, a_z=a_z))

    logging.info(psutil.virtual_memory())

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    return E_H0, hermite_matrix, hermite_lhy_abc_list, ind_vec


def get_meshgrid_partwise(ind_vec, pos_vec, parts):
    ind_vec_splitted = np.array_split(ind_vec, parts)
    pos_vec_splitted = np.array_split(pos_vec, parts)
    logging.info(f"ind_vec_splitted: {ind_vec_splitted}")
    logging.info(f"len(ind_vec_splitted): {len(ind_vec_splitted)}")

    logging.info(f"ind_vec_splitted: {ind_vec_splitted}")
    logging.info(f"pos_vec_splitted: {pos_vec_splitted}")
    logging.info(f"len(pos_vec_splitted): {len(pos_vec_splitted)}")

    ind_v_list = []
    pos_v_list = []

    # for i, (ind, pos) in enumerate(zip(ind_vec_splitted, pos_vec_splitted)):
    for i in range(parts):
        for j in range(parts):
            print(f"i, j: {i}, {j}")
            logging.info(f"ind: {ind_vec_splitted[i]}")
            logging.info(f"pos: {pos_vec_splitted[j]}")
            logging.info(f"ind.shape: {ind_vec_splitted[i].shape}")
            logging.info(f"pos.shape: {pos_vec_splitted[i].shape}")
            ind_v, pos_v = np.meshgrid(ind_vec_splitted[i], pos_vec_splitted[j], indexing='ij')
            ind_v_list.append(ind_v)
            pos_v_list.append(pos_v)

    return ind_v_list, pos_v_list


def get_hermite_matrix_smart(System, operator, nx, ny, nz, parts):
    dim = int(nx * ny * nz)
    hermite_lhy_abc_list = []
    hermite_matrix_all = np.zeros((dim, dim))
    logging.info(psutil.virtual_memory())
    pos_max = System.Res.x * System.Res.y * System.Res.z
    logging.info(psutil.virtual_memory())
    pos_vec = np.arange(0, pos_max, 1)
    logging.info(psutil.virtual_memory())
    ind_vec = np.arange(0, dim, 1)
    logging.info(f"pos_vec: {round(pos_vec.nbytes / 1024 / 1024,2)}MB")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    logging.info("Create meshgrid")
    logging.info(psutil.virtual_memory())

    dV = System.volume_element(fourier_space=False)

    ind_vec_splitted = np.array_split(ind_vec, parts)
    pos_vec_splitted = np.array_split(pos_vec, parts)
    logging.info(f"ind_vec_splitted: {ind_vec_splitted}")
    logging.info(f"len(ind_vec_splitted): {len(ind_vec_splitted)}")

    logging.info(f"ind_vec_splitted: {ind_vec_splitted}")
    logging.info(f"pos_vec_splitted: {pos_vec_splitted}")
    logging.info(f"len(pos_vec_splitted): {len(pos_vec_splitted)}")

    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)
    
    E_H0 = np.diag(En(ind_vec, ny, nz, a_y=a_y, a_z=a_z))

    # use splitted arrays and meshgrid to be able to fit in GPU
    for i in range(parts):
        for j in range(parts):
            print(f"i, j: {i}, {j}")
            logging.info(f"ind_vec_splitted[{i}]: {ind_vec_splitted[i]}")
            logging.info(f"pos_vec_splitted[{j}]: {pos_vec_splitted[j]}")
            logging.info(f"ind.shape: {ind_vec_splitted[i].shape}")
            logging.info(f"pos.shape: {pos_vec_splitted[i].shape}")
            ind_v, pos_v = np.meshgrid(ind_vec_splitted[i], pos_vec_splitted[j], indexing='ij', sparse=True)

            logging.info(f"ind_v.shape: {ind_v.shape}")
            logging.info(f"pos_v.shape: {pos_v.shape}")
            logging.info(f"ind_v: {round(ind_v.nbytes / 1024 / 1024,2)}MB")
            logging.info(f"pos_v: {round(pos_v.nbytes / 1024 / 1024,2)}MB")
            logging.info(psutil.virtual_memory())

            x, y, z = position(System, pos_v) 
            logging.info(f"x.shape: {x.shape}")
            logging.info(psutil.virtual_memory())
            logging.info(f"x: {round(x.nbytes / 1024 / 1024,2)}MB")

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            logging.info(psutil.virtual_memory())

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(mempool.used_bytes() / (1024 ** 2))
                logging.info(mempool.total_bytes() / (1024 ** 2))

            try:
                logging.info(f"Trying to load bog_helper from: {path_bog_helper}")
                with open(path_bog_helper, "rb") as g:
                    bog_helper_load = np.load(file=g)
                    bog_helper = bog_helper_load["bog_helper"]
                    bog_helper = cp.asnumpy(bog_helper)
            except Exception as e:
                logging.info(f"File not found. Calculating bog_helper")
                with run_time(name=f"bog_helper {nx} {ny} {nz}"):
                    bog_helper = HO_3D(x, y, z, ind_v, ny, nz, a_y=a_y, a_z=a_z, cupy_used=False)

                logging.info(f"Calculated bog_helper")
                bog_helper = cp.asnumpy(bog_helper)
                logging.info(f"Converted bog_helper to numpy")

                logging.info(f"Save bog_helper as bog_helper to: {path_bog_helper}")
                with open(path_bog_helper, "wb") as g:
                    np.savez_compressed(g, bog_helper=bog_helper)
                logging.info(f"Succesfully saved bog_helper")

            mat_check(bog_helper, "bog_helper:")
            logging.info(f"bog_helper.shape: {bog_helper.shape}")

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            # mat_check(bog_helper, "bog_helper:")
            # mat2d(bog_helper, "bog_helper:")

            logging.info(f"Calculate operator_raveled")
            try:
                # bring operator in same order as x
                with run_time(name=f"operator_raveled {nx} {ny} {nz}"):
                    operator_raveled = operator_ravel(operator, pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
                # operator = None
            except Exception as e:
                logging.info(f"Failed. Trying to use cp.asnumpy")
                with run_time(name=f"operator_raveled {nx} {ny} {nz}"):
                    operator_raveled = operator_ravel(cp.asnumpy(operator), pos_v, System.Res.y, System.Res.z, cupy_used=False)
                operator = None

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            operator_raveled = cp.asnumpy(operator_raveled)

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            mat2d(bog_helper, "bog_helper")
            mat2d(operator_raveled, "operator_raveled")

            # TODO: bog_helper 25GB for 32 ** 3, so bog_helper, operator_raveled, bog_helper_operator_applied = 75GB
            print(f"Calculate bog_helper_operator_applied")
            with run_time(name=f"bog_helper_operator_applied {nx} {ny} {nz}"):
                bog_helper_operator_applied = bog_helper * operator_raveled

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            print(f"Calculate bog_helper_swapped")
            try:
                bog_helper_swapped = cp.swapaxes(bog_helper, 0, 1)
            except Exception as e:
                print(f"Failed. Trying to use np.swapaxes.")
                bog_helper_swapped = np.swapaxes(bog_helper, 0, 1)

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            print(f"System.lhy_factor: {System.lhy_factor}")
            if System.lhy_factor != 0:
                try:
                    hermite_lhy_abc_list = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper, bog_helper_swapped, cupy_used=cupy_used)
                    bog_helper = None
                except Exception as e:
                    print(f"Failed. Trying to use cp.asnumpy")
                    hermite_lhy_abc_list = get_hermite_lhy_abc_list(System, pos_v, dim, cp.asnumpy(bog_helper), cp.asnumpy(bog_helper_swapped), cupy_used=False)
                    bog_helper = None
            else:
                bog_helper = None

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            logging.info(f"Calculate hermite_matrix")
            with run_time(name=f"flat hermite_matrix dot {dim}"):
                try:
                    hermite_matrix = cp.dot(bog_helper_operator_applied, bog_helper_swapped) * dV
                except Exception as e:
                    logging.info(f"WARNING: GPU not used for calculation of integrals")
                    logging.info(f"Failed. Trying to use cp.asnumpy")
                    hermite_matrix = np.dot(cp.asnumpy(bog_helper_operator_applied), cp.asnumpy(bog_helper_swapped)) * dV
                    # hermite_matrix = cp.array(hermite_matrix)

            # mat_check(hermite_matrix, "hermite_matrix:")
            # mat2d(hermite_matrix, "hermite_matrix:")

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            logging.info(f"Save hermite_matrix as hermite_matrix to: {path_hermite_matrix}")
            with open(path_hermite_matrix, "wb") as g:
                np.savez_compressed(g, hermite_matrix=hermite_matrix)
        logging.info(f"Succesfully saved")

        if cupy_used:
            mempool = cp.get_default_memory_pool()
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        split_size = int(dim / parts)
        print(split_size)
        mat2d(hermite_matrix, "hermite_matrix_parts")
        print(f"{hermite_matrix.shape}")
        print(f"{i * split_size}: {(i + 1) * split_size}, {j * split_size}: {(j + 1) * split_size}")
        hermite_matrix_all[i * split_size: (i + 1) * split_size, j * split_size: (j + 1) * split_size] = hermite_matrix

    hermite_matrix_all = np.flip(hermite_matrix_all, axis=0)

    return E_H0, hermite_matrix_all, hermite_lhy_abc_list


def get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper, bog_helper_swapped=None, cupy_used=True, reduced_version=True):
    print(f"Get lhy terms")
    dV = System.volume_element(fourier_space=False)
    g_qf = functions.get_g_qf_bog(N=System.N_list[0],
                                  a_s=float(System.a_s_array[0, 0]),
                                  a_dd=float(System.a_dd_array[0, 0]))
    print(f"g_qf: {g_qf}")

    if reduced_version:
        psi_val_raveled = operator_ravel(cp.array(cp.real(System.psi_val_list[0])), pos_v, System.Res.y, System.Res.z, cupy_used=cupy_used)
        lhy_key = psi_val_raveled ** 3.0
        psi_val_raveled = None

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        # WARNING: overwrite lhy_key to save space
        try:
            bog_helper = cp.array(bog_helper)
            lhy_key = cp.array(lhy_key)
            lhy_key = cp.asnumpy(bog_helper * lhy_key)
            bog_helper = cp.asnumpy(bog_helper)
        except Exception as e:
            print(f"Failed. Trying to get lhy_ley with cp.asnumpy.")
            lhy_key = cp.asnumpy(bog_helper) * cp.asnumpy(lhy_key)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        if bog_helper_swapped is None:
            print(f"Calculate bog_helper_swapped")
            try:
                bog_helper = cp.array(bog_helper)
                bog_helper_swapped = cp.asnumpy(cp.swapaxes(bog_helper, 0, 1))
            except Exception as e:
                print(f"Failed. Trying to use np.swapaxes.")
                bog_helper_swapped = np.swapaxes(bog_helper, 0, 1)
            bog_helper = None

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        with run_time(name=f"get_hermite_lhy_abc_list reduced_version"):
            try:
                # lhy_key = cp.array(lhy_key)
                # bog_helper_swapped = cp.array(bog_helper_swapped)
                print(f"lhy_key.shape: {lhy_key.shape}")
                print(f"bog_helper_swapped.shape: {bog_helper_swapped.shape}")
                hermite_lhy_i = np.zeros((dim, dim))
                print(f"hermite_lhy_i.shape: {hermite_lhy_i.shape}")
                # hermite_lhy_i = cp.asnumpy(cp.dot(lhy_key, bog_helper_swapped))
                hermite_lhy_i = g_qf * np.dot(lhy_key, bog_helper_swapped) * dV
                bog_helper_swapped = None
                # hermite_lhy_i = g_qf * dV * hermite_lhy_i
                # hermite_lhy_i = cp.asnumpy(hermite_lhy_i)
            except Exception as e:
                print(f"Failed. Trying to use np.dot.")
                print(f"type(lhy_key): {type(lhy_key)}")
                print(f"type(bog_helper_swapped): {type(bog_helper_swapped)}")
                hermite_lhy_i = (System.N_list[0] ** 1.5) * g_qf * np.dot(lhy_key, bog_helper_swapped) * dV
            bog_helper_swapped = None
            
        return hermite_lhy_i
    else:
        hermite_lhy_abc_list = []

        if cupy_used:
            mempool = cp.get_default_memory_pool()
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        lhy_abc_list = get_lhy_terms(System, pos_v, dim, ground_state=True, cupy_used=cupy_used)

        if cupy_used:
            mempool = cp.get_default_memory_pool()
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        with run_time(name=f"lhy all"):
            for lhy_key in lhy_abc_list:
                if cupy_used:
                    try:
                        bog_helper_lhy_applied = cp.array(bog_helper) * cp.array(lhy_key)
                    except Exception as e:
                        print("WARNING numpy used to get bog_helper_lhy_applied")
                        bog_helper_lhy_applied = None
                        bog_helper_lhy_applied = bog_helper * lhy_key
                    lhy_key = None
                    bog_helper = None
                    # logging.info(f"bog_helper_lhy_applied: {bog_helper_lhy_applied}")
                    # logging.info(f"bog_helper_swapped: {bog_helper_swapped}")
                    try:
                        hermite_lhy_i = g_qf * cp.dot(cp.array(bog_helper_lhy_applied), cp.array(bog_helper_swapped)) * dV
                    except Exception as e:
                        print("Clear GPU")
                        hermite_lhy_i = None
                        print(f"type bog_helper_lhy_applied: {type(bog_helper_lhy_applied)}")
                        print(f"type bog_helper_swapped: {type(bog_helper_swapped)}")

                        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

                        try:
                            bog_helper_lhy_applied = cp.asnumpy(bog_helper_lhy_applied)
                        except:
                            print("Conversion to numpy failed")

                        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

                        try:
                            bog_helper_swapped = cp.asnumpy(bog_helper_swapped)
                        except:
                            print("Conversion to numpy failed")

                        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
                        pinned_mempool = cp.get_default_pinned_memory_pool()
                        pinned_mempool.free_all_blocks()

                        print("WARNING numpy used to get hermite_lhy_i")
                        print(f"type bog_helper_lhy_applied: {type(bog_helper_lhy_applied)}")
                        print(f"type bog_helper_swapped: {type(bog_helper_swapped)}")
                        hermite_lhy_i = g_qf * np.dot(bog_helper_lhy_applied, bog_helper_swapped) * dV
                    try:
                        hermite_lhy_i = cp.asnumpy(hermite_lhy_i)
                        bog_helper_swapped = cp.asnumpy(bog_helper_swapped)
                    except:
                        print("Conversion to numpy failed")
                    hermite_lhy_abc_list.append(hermite_lhy_i)
                    # hermite_lhy_abc_list.append(np.array(cp.dot(bog_helper_lhy_applied, bog_helper_swapped)) * dV)
                else:
                    bog_helper_lhy_applied = bog_helper * lhy_key
                    hermite_lhy_i = g_qf * np.dot(bog_helper_lhy_applied, bog_helper_swapped) * dV
                    hermite_lhy_abc_list.append(hermite_lhy_i)

        if cupy_used:
            bog_helper = None
            bog_helper_swapped = None
            lhy_abc_list = None
            mempool = cp.get_default_memory_pool()
            logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
            logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

        return hermite_lhy_abc_list

def check_eGPE(System, g):
    print(f"check_eGPE")
    if cupy_used:
        psi_0 = cp.array(System.psi_val_list[0])
        V_part = cp.array(System.V_val) * psi_0
    else:
        psi_0 = System.psi_val_list[0]
        V_part = System.V_val * psi_0
    H_kin_part = cp.fft.ifftn((0.5 * System.k_squared) * cp.fft.fftn(psi_0))
    # H_kin_test = cp.exp(-1.0 * (0.5 * System.k_squared) * System.dt)
    # dV = System.volume_element(fourier_space=False)
    # norm_psi_0_test = cp.sum(cp.abs(psi_0) ** 2) * dV
    norm_psi_0 = System.sum_dV(System.get_density_list()[0], fourier_space=False)
    print(f"norm_psi_0: {norm_psi_0}")
    # print(f"H_kin: {System.H_kin_list[0]}")
    # print(f"H_kin_test: {H_kin_test.shape}")
    # print(f"H_kin ok?: {cp.all(cp.array(H_kin_test) == cp.array(System.H_kin_list[0]))}")

    g_qf = functions.get_g_qf_bog(N=System.N_list[0],
                                  a_s=float(System.a_s_array[0, 0]),
                                  a_dd=float(System.a_dd_array[0, 0]))

    g_part = g * System.N_list[0] * cp.abs(psi_0) ** 2.0 * psi_0
    g_qf_part = g_qf * cp.abs(psi_0) ** 3.0 * psi_0
    a_dd_part = System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * cp.real(cp.fft.ifftn(cp.array(System.V_k_val) * cp.fft.fftn(cp.abs(psi_0) ** 2.0))) * psi_0

    mu_part = System.mu_arr[0] * psi_0
    
    all_parts = H_kin_part + V_part + g_part + a_dd_part + g_qf_part - mu_part
    mu_mesh = (H_kin_part + V_part + g_part) / psi_0
    limit = 10**-15
    all_0 = mat_check(all_parts, name="all_parts", limit=limit)
    mat_check(mu_part, name="mu_part", limit=limit)
    mat_check(g_part, name="g_part", limit=limit)
    mat_check(V_part, name="V_part", limit=limit)
    mat_check(H_kin_part, name="H_kin_part", limit=limit)
    mat_check(a_dd_part, name="a_dd_part", limit=limit)
    mat_check(g_qf_part, name="g_qf_part", limit=limit)
    
    print(f"limit: {limit}")
    # print(f"all_parts[:, int(System.Res.y / 2), int(System.Res.z / 2)].tolist(): "
            # f"{cp.real(all_0[:, int(System.Res.y / 2), int(System.Res.z / 2)]).tolist()}")
    # print(f"\nSystem.psi_val_list[0]x:\n {System.psi_val_list[0][:, int(System.Res.y / 2), int(System.Res.z / 2)].tolist()}")
    # print(f"\nSystem.psi_val_list[0]y:\n {System.psi_val_list[0][int(System.Res.x / 2), :, int(System.Res.z / 2)].tolist()}")
    # print(f"\nSystem.psi_val_list[0]z:\n {System.psi_val_list[0][int(System.Res.x / 2), int(System.Res.y / 2), :].tolist()}")

    # print("En_TF")
    # for i in range(0, 8):
    #     for j in range(0, 8):
    #         print(f"{i}, {j}: {En_TF(i, j)}")

    return all_parts


def mat_check(arr, name, limit = 10 ** -5):
    print(f"{name}:")
    arr_limited = cp.where(cp.abs(arr) < limit, 0, cp.real(arr))

    arr_imag = np.imag(cp.asnumpy(arr))
    arr_real = np.real(cp.asnumpy(arr))
    imag_min = np.min(arr_imag)
    imag_max = np.max(arr_imag)
    real_min = np.min(arr_real)
    real_max = np.max(arr_real)

    number_0 = cp.count_nonzero(arr_limited==0)
    print(f"Min/Max imag: {imag_min} and {imag_max} at {minimum_position(arr_imag)} and {maximum_position(arr_imag)}")
    print(f"Min/Max real: {real_min} and {real_max} at {minimum_position(arr_real)} and {maximum_position(arr_real)}")
    print(f"Number of zeros: {number_0} of {arr.size}")
    
    return arr_limited


def get_bogoliubov_matrix(System, operator, nx, ny, nz, index_dict, cupy_used, mode="flat",
                          dipol=False, l_0=None, ground_state=True, cut_hermite_orders=False,
                          cut_hermite_values=False, reduced_version=False):
    # contact_interaction_vec, dipol_term_vec, mu_lhy_list = System.get_dipol_U_dd_mu_lhy()

    # np cp conversion
    # contact_interaction_vec = cp.array(contact_interaction_vec)
    # dipol_term_vec = cp.array(dipol_term_vec)
    System.V_val = cp.array(System.V_val)
    System.k_squared = cp.array(System.k_squared)
   
    g = 4.0 * np.pi * System.a_s_array[0, 0]
    # mu = functions.mu_3d(g * System.N_list[0])
    mu = System.mu_arr[0]
    # mu = System.mu_arr[0] - 8.32
    # mu = float(g * np.max(operator))
    # mu = 3.833 # no_dipol_no_lhy_w_paper
    # print(f"mu by hand: {mu}, g: {g}")
    print(f"get_bogoliubov_matrix mu: {mu}")
    print(f"g: {g}")
 
    should_0 = check_eGPE(System, g) 
    limit = 10**-7
    print(f"limit: {limit}")
    mat_check(should_0, name="should_0", limit=limit)
    # print(f"should_0: {should_0}")

    if cut_hermite_values:
        # will be cut by cut_hermite_by_value()
        ind_vec_dict = None
        dim = int(nx * ny * nz)
    elif cut_hermite_orders:
        # gets cut by hermite order
        ind_vec_dict = get_ind_vec_dict(nx, ny, nz)
        dim = len(ind_vec_dict)
    else:
        ind_vec_dict = None
        dim = int(nx * ny * nz)

    if path_hermite_matrix.is_file():
        print(f"Trying to load hermite_matrix from: {path_hermite_matrix}")
        with open(path_hermite_matrix, "rb") as g:
            hermite_matrix_load = np.load(file=g, allow_pickle=False)
            hermite_matrix = hermite_matrix_load["hermite_matrix"]
        print(f"Succesfully loaded hermite_matrix from: {path_hermite_matrix}")

        try:
            logging.info(f"Trying to load hermite_lhy_abc_list from: {path_hermite_lhy_abc_list}")
            with open(path_hermite_lhy_abc_list, "rb") as g:
                hermite_lhy_abc_list_load = np.load(file=g)
                hermite_lhy_abc = hermite_lhy_abc_list_load["hermite_lhy_abc_list"]
        except Exception as e:
            logging.info(f"Failed loading hermite_lhy_abc_list")
            logging.info(f"Trying to load bog_helper from: {path_bog_helper}")
            with open(path_bog_helper, "rb") as g:
                bog_helper_load = np.load(file=g)
                bog_helper = bog_helper_load["bog_helper"]
            print(f"Succesfully loaded bog_helper!")
            dim = bog_helper.shape[0]
            # if cut used for bog_helper, other ind_vec needed (load from file)
            pos_max = System.Res.x * System.Res.y * System.Res.z
            if cupy_used:
                ind_vec = cp.arange(0, dim, 1)
                pos_vec = cp.arange(0, pos_max, 1)
                ind_v, pos_v = cp.meshgrid(ind_vec, pos_vec, indexing='ij', sparse=True)
            else:
                ind_vec = np.arange(0, dim, 1)
                pos_vec = np.arange(0, pos_max, 1)
                ind_v, pos_v = np.meshgrid(ind_vec, pos_vec, indexing='ij', sparse=True)

            if cupy_used:
                mempool = cp.get_default_memory_pool()
                logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            if System.lhy_factor != 0:
                print(f"System.lhy_factor: {System.lhy_factor}")
                if reduced_version:
                    hermite_lhy_base = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper,
                                                                bog_helper_swapped=None,
                                                                cupy_used=cupy_used,
                                                                reduced_version=reduced_version)
                else:
                    hermite_lhy_abc = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper, bog_helper_swapped,
                                                               cupy_used=cupy_used)

                if cupy_used:
                    mempool = cp.get_default_memory_pool()
                    logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                    logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

            # bog_helper = None
            # bog_helper_swapped = None

        a_y = (System.w_y / System.w_x)
        a_z = (System.w_z / System.w_x)
        E_H0 = np.diag(En(ind_vec, ny, nz, a_y=a_y, a_z=a_z))
        try: 
            E_H0 = cp.asnumpy(E_H0)
        except Exception as e:
            print(f"Conversion of E_H0 to numpy failed.")
        print(f"E_H0: {E_H0.shape}, {type(E_H0)}")

        loaded = True
    else:
        loaded = False
        print(f"Calculating hermite_matrix")
        if mode == "dask":
            logging.info("get_hermite_matrix_dask")
            with run_time(name=f"{mode} {nx} {ny} {nz}"):
                hermite_matrix_triu, E_H0 = get_hermite_matrix_dask(System, operator, dim, index_dict,
                                                                    fast=True)
            hermite_matrix = functions.symmetric_mat(hermite_matrix_triu)

            print(f"System.lhy_factor: {System.lhy_factor}")
            if System.lhy_factor != 0:
                hermite_lhy_abc = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper, bog_helper_swapped)

        elif mode == "cupy":
            logging.info("get_hermite_matrix_cupy")
            with run_time(name=f"{mode} {nx} {ny} {nz}"):
                hermite_matrix_triu, E_H0 = get_hermite_matrix(System, operator, dim, index_dict)
            hermite_matrix = functions.symmetric_mat(hermite_matrix_triu)

            print(f"System.lhy_factor: {System.lhy_factor}")
            if System.lhy_factor != 0:
                hermite_lhy_abc = get_hermite_lhy_abc_list(System, pos_v, dim, bog_helper, bog_helper_swapped)

        elif mode == "flat":
            logging.info("get_hermite_matrix_flat")
            with run_time(name=f"{mode} {nx} {ny} {nz}"):
                E_H0, hermite_matrix, hermite_lhy_abc, ind_vec = get_hermite_matrix_flat(System, operator, nx, ny, nz,
                                                                                         ind_vec_dict=ind_vec_dict,
                                                                                         cut_hermite_orders=cut_hermite_orders,
                                                                                         cut_hermite_values=cut_hermite_values)
            dim = len(ind_vec)

        elif mode == "fft":
            logging.info("get_hermite_matrix_fft")
            with run_time(name=f"{mode} {nx} {ny} {nz}"):
                # hermite_matrix, E_H0 = get_hermite_matrix_fft(System, operator, nx, ny, nz)
                a, b = get_hermite_matrix_fft(System, operator, nx, ny, nz, mu)
            dim = int(nx * ny * nz)
            ind_vec = np.arange(0, dim, 1)

        elif mode == "smart":
            logging.info("get_hermite_matrix_smart")
            with run_time(name=f"{mode} {nx} {ny} {nz}"):
                E_H0, hermite_matrix, hermite_lhy_abc = get_hermite_matrix_smart(System, operator, nx, ny, nz, parts=2)

        elif mode == "linear":
            logging.info("get_hermite_matrix_linear")
            with run_time(name=f"{mode} {nx} {ny} {nz}"):
                E_H0, hermite_matrix_triu, hermite_lhy_abc = get_hermite_matrix_linear(System, operator, dim, index_dict, cupy_used=True)
            hermite_matrix = functions.symmetric_mat(hermite_matrix_triu)

        else:
            sys.exit("Mode not implemented. Choose between dask, cupy, flat.")

    # logging.info(f"hermite_matrix:\n{hermite_matrix}")
    # mat2d(hermite_matrix, "hermite_matrix_parts")

    if mode == "fft":
        logging.info(f"a.shape: {a.shape}")
    else:
        logging.info(f"hermite_matrix.shape: {hermite_matrix.shape}")

    logging.info("get_hermite_matrix_flat done")
    # mat_check(E_H0, "E_H0:")
    # mat2d(E_H0, "E_H0:")

    g = 4.0 * np.pi * System.a_s_array[0, 0]
    # g = System.a_s_factor * System.a_s_array[0, 0]

    # IMPORTANT: diagonalization will be done in CPU (no cupy method),
    # also matrix is huge (80GB for 32, 32, 32 gird and polynomials)
    if loaded:
        cupy_used = False

    print(f"g: {g}")

    if not mode == "fft":
        mat2d(hermite_matrix, label="hermite_matrix:")

        b = g * hermite_matrix
        a = E_H0 + 2.0 * b - np.diag(dim * [np.array(mu)])
        print(f"E_H0: {E_H0.shape}, {type(E_H0)}")
        print(f"E_H0: {E_H0}")

    print(f"dim: {dim}, {type(dim)}")
    print(f"b: {b.shape}, {type(b)}")
    print(f"mu: {mu.shape}, {type(mu)}")
    print(f"b: {b}")

    if cupy_used:
        E_H0 = None

    # if cupy_used:
    #     mempool = cp.get_default_memory_pool()
    #     print(mempool.used_bytes() / (1024 ** 2))
    #     print(mempool.total_bytes() / (1024 ** 2))

    # psi_0 = cp.zeros(dim)
    # operator_0 = cp.real(System.psi_val_list[0])
    # for l in range(dim):
    #     comb1 = index_dict[l]
    #     psi_0[l] = hermite_transform(System, operator_0, comb1, comb1, a_y, a_z, sandwich=False)
    # fix_mu = cp.dot(psi_0.T, cp.dot(a - b, psi_0)) / cp.dot(psi_0.T, psi_0)
    # print(f"fix_mu: {fix_mu}")
    
    # if mode == "flat":
    #     if cupy_used:
    #         mempool = cp.get_default_memory_pool()
    #         print(mempool.used_bytes() / (1024 ** 2))
    #         print(mempool.total_bytes() / (1024 ** 2))
    #         print(pinned_mempool.n_free_blocks())    
    #         matrix = cp.zeros((2*dim, 2*dim), dtype=cp.complex_)
    #         matrix[0:dim, 0:dim] = a
    #         matrix[0:dim, dim:] = -b
    #         matrix[dim:, 0:dim] = b
    #         matrix[dim:, dim:] = -a
    #     else:
    #         matrix = np.block([[a, -b],
    #                            [b, -a]])
    # elif mode == "dask":
    #     matrix = np.block([[a, -b],
    #                        [b, -a]])


    if reduced_version:
        logging.info(f"reduced_version")
        # if (System.lhy_factor != 0):
        #     print(f"lhy used, reduced_version")
        #     a += 2.5 * hermite_lhy_base
        #     b += 1.5 * hermite_lhy_base
        # if dipol and System.a_dd_array[0, 0] != 0 and reduced_version and not mode == "fft":
        #     logging.info(f"Groundstate assumed for dipol interaction, reduced_version")
        #     a += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "A", ground_state=ground_state)
        #     b += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "B", ground_state=ground_state)

        if not mode == "fft":
            if dipol and System.a_dd_array[0, 0] != 0:
                a += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "A", ground_state=ground_state)
                b -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "B", ground_state=ground_state)
            if (System.lhy_factor != 0) and not mode == "fft":
                a += hermite_lhy_abc[0]
                b -= hermite_lhy_abc[1]
            
        matrix = (a - b) @ (a + b)
    else:
        matrix = np.block([[a, -b],
                           [b, -a]])

    mat2d(matrix, label="matrix block:")

    if cupy_used:
        mempool = cp.get_default_memory_pool()
        logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
        logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")

    print(f"lhy_factor: {System.lhy_factor}")
    if (System.lhy_factor != 0) and not mode == "fft":
        # logging.info(f"matrix.dtype: {matrix.dtype}")
        # logging.info(f"matrix.shape: {matrix.shape}")
        # logging.info(f"matrix[0:dim, 0:dim]: {matrix[0:dim, 0:dim].shape}")
        # logging.info(f"hermite_lhy_abc[0].dtype: {hermite_lhy_abc[0].dtype}")
        # logging.info(f"hermite_lhy_abc[0].shape: {hermite_lhy_abc[0].shape}")
        # logging.info(f"len hermite_lhy_abc: {len(hermite_lhy_abc)}")
        # N ** 1.5 is already in g_qf
        if not reduced_version:
            matrix[0:dim, 0:dim] += hermite_lhy_abc[0]
            matrix[0:dim, dim:] -= hermite_lhy_abc[1]
            matrix[dim:, 0:dim] += hermite_lhy_abc[2]
            matrix[dim:, dim:] -= hermite_lhy_abc[0]
        # logging.info(f"hermite_lhy_abc[0]:\n{+hermite_lhy_abc[0]}")
        # logging.info(f"hermite_lhy_abc[1]:\n{-hermite_lhy_abc[1]}")
        # logging.info(f"hermite_lhy_abc[2]:\n{+hermite_lhy_abc[2]}")

    if dipol and System.a_dd_array[0, 0] != 0 and not reduced_version and not mode == "fft":
        logging.info(f"Dipols used")
        logging.info(f"System.N_list[0]: {System.N_list[0]}")
        logging.info(f"System.a_dd_factor: {System.a_dd_factor}")
        logging.info(f"System.a_dd_array[0, 0]: {System.a_dd_array[0, 0]}")
        if args.dask_dipol:
            logging.info(f"Dask used for dipols")
            hermite_dipol_abcd_list = []
            with run_time(name=f"get_hermite_matrix_dipol_dask"):
                hermite_dipol_abcd_list = get_hermite_matrix_dipol_dask(System,
                                                                        index_dict, dim,
                                                                        ground_state=ground_state)
            matrix[0:dim, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * hermite_dipol_abcd_list[0]
            matrix[0:dim, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * hermite_dipol_abcd_list[1]
            matrix[dim:, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * hermite_dipol_abcd_list[2]
            matrix[dim:, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * hermite_dipol_abcd_list[3]
        else:
            # matrix[0:dim, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol_entry(System, index_dict, dim, "A", ground_state=ground_state)
            # matrix[0:dim, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol_entry(System, index_dict, dim, "B", ground_state=ground_state)
            # matrix[dim:, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol_entry(System, index_dict, dim, "C", ground_state=ground_state)
            # matrix[dim:, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol_entry(System, index_dict, dim, "D", ground_state=ground_state)
            if ind_vec_dict is None:
                matrix[0:dim, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "A", ground_state=ground_state)
                matrix[0:dim, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "B", ground_state=ground_state)
                matrix[dim:, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "C", ground_state=ground_state)
                matrix[dim:, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, index_dict, dim, "D", ground_state=ground_state)
            else:
                matrix[0:dim, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, ind_vec_dict, dim, "A", ground_state=ground_state)
                matrix[0:dim, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, ind_vec_dict, dim, "B", ground_state=ground_state)
                matrix[dim:, 0:dim] += System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, ind_vec_dict, dim, "C", ground_state=ground_state)
                matrix[dim:, dim:] -= System.N_list[0] * System.a_dd_factor * System.a_dd_array[0, 0] * get_hermite_dipol(System, ind_vec_dict, dim, "D", ground_state=ground_state)

    else:
        logging.info(f"No dipols or reduced_version")
        logging.info(f"dipol: {dipol}")
        logging.info(f"a_dd_array[0, 0]: {System.a_dd_array[0, 0]}")

    # mat2d(matrix, label="matrix:")

    return matrix, ind_vec

def hermite_laplace(System, i, a_y, a_z):
    k = 2
    factor = 2 ** k * (factorial(i) / np.math.factorial(i - k))
    herm_laplace = factor * harmonic_eigenstate_3d(System, i - k, a_y, a_z)
    
    return herm_laplace

def check_hermite_orthonormal(System, nx, ny, nz, index_dict):
    print("Check if hermite basis is orthonormal")
    dim = int(nx * ny * nz)
    h_orth = cp.zeros(dim)


    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)

    for l in range(dim):
        comb1 = index_dict[l]
        h_orth[l] = hermite_transform(System, cp.ones_like(cp.real(System.psi_val_list[0])),
                                      comb1, comb1, a_y, a_z, sandwich=True, fourier_space=False)
   
    hermite_norm = cp.sum(h_orth) / dim
    mat2d(h_orth, "h_orth:")

    # print(f"h_orth: {mat2d(h_orth}")
    print(f"h_orth min, max: {cp.min(h_orth)}, {cp.max(h_orth)}")
    print(f"hermite_norm: {hermite_norm}")

    return index_dict


def check_psi(System, nx, ny, nz, ind_vec, index_dict):
    print("check psi")
    dim = len(ind_vec)
    psi_0 = cp.zeros(dim)
    a_y = (System.w_y / System.w_x)
    a_z = (System.w_z / System.w_x)

    operator = cp.real(System.psi_val_list[0])
    # operator = cp.array(System.psi_val_list[0])

    x = System.x_mesh[:, System.Res.y // 2, System.Res.z // 2]
    # y = System.y_mesh[System.Res.x // 2, :, System.Res.z // 2]
    # z = System.z_mesh[System.Res.x // 2, System.Res.y // 2, :]

    # gauss_0 = np.pi ** -0.25 * np.exp(-x ** 2.0 / 2.0)
    # test_1 = System.sum_dV(gauss_0 * gauss_0, fourier_space=False, dV=System.dx)
    # test_2 = cp.sum(gauss_0 * gauss_0 * System.dx) 
    # print(f"test_1: {test_1}")
    # print(f"test_2: {test_2}")

    # print(f"ind_vec: {ind_vec}")
    # print(f"len(ind_vec): {len(ind_vec)}")
    # print(f"index_dict: {index_dict}")
    # print(f"len(index_dict): {len(index_dict)}")
    for i, l in enumerate(ind_vec):
        comb1 = index_dict[l]
        psi_0[i] = hermite_transform(System, operator, comb1, comb1, a_y, a_z,
                                     sandwich=False, fourier_space=False)

    print(f"System.dV: {System.dV}")
    # print(f"dV_by_dxyz: {System.dx * System.dy * System.dz}")

    norm = cp.sqrt(cp.dot(psi_0, psi_0))
    print(f"Check sol, norm psi_0: {norm}")

    norm_0 = System.sum_dV(cp.abs(operator) ** 2.0, fourier_space=False, dV=System.dV) 
    # norm_1 = cp.sum(cp.abs(psi_0) ** 2.0) 
    # norm_2 = cp.sum(cp.abs(operator) ** 2.0 * System.dV) 
    print(f"norm_0: {norm_0}")
    # print(f"norm_1: {norm_1}")
    # print(f"norm_2: {norm_3}")

    return psi_0


def check_sol(System, nx, ny, nz, bog_mat, ind_vec, index_dict, reduced_version=False):
    # dim = bog_mat.shape[0] // 2
    psi_0 = check_psi(System, nx, ny, nz, ind_vec, index_dict)
    
    # if cupy_used:
    #     bog_mat = cp.array(bog_mat)
    #     psi_0_2dim = cp.array(psi_0_2dim)
    # result = cp.dot(bog_mat, psi_0_2dim)
    # result = np.einsum("ij,j->i", bog_mat, psi_0_2dim)
    if reduced_version:
        psi_0 = cp.asnumpy(psi_0)
        result = np.real(np.einsum("ij,j->i", bog_mat, psi_0))
    else:
        psi_0_2dim = cp.hstack((psi_0, psi_0))
        psi_0_2dim = cp.asnumpy(psi_0_2dim)
        result = np.real(np.einsum("ij,j->i", bog_mat, psi_0_2dim))
    
    # result_0 = np.where(np.abs(result) > 0.0001, result, 0)
    # mat2d(psi_0_2dim, "psi_0_2dim:")
    mat2d(result, "result:")
    print(f"result min, max: {cp.min(cp.real(result))}, {cp.max(cp.real(result))}")
    # fix_mu = result[0] / psi_0[0]
    # print(f"fix_mu: {fix_mu}")
    
    return result


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])

    os.environ["SUPERSOLIDS_GPU_INDEX"] = str(args.gpu_index)
    os.environ["SUPERSOLIDS_GPU_OFF"] = str(args.gpu_off)
    __GPU_OFF_ENV__, __GPU_INDEX_ENV__ = get_version.get_env_variables(gpu_index_str=args.gpu_index)
    cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np,
                                                                   gpu_off=__GPU_OFF_ENV__,
                                                                   gpu_index=__GPU_INDEX_ENV__)

    from supersolids.Schroedinger import Schroedinger
    from supersolids.SchroedingerMixture import SchroedingerMixture
    from supersolids.SchroedingerMixtureNumpy import SchroedingerMixtureNumpy
    from supersolids.tools.get_System_at_npz import get_System_at_npz
    pytorch = args.pytorch
    arnoldi = args.arnoldi

    if pytorch:
        import torch
        mode_eig = "pytorch"
    if arnoldi:
        mode_eig = "arnoldi"
    if pytorch and arnoldi:
        sys.exit("Choose pytorch or arnoldi!")
    if not pytorch and not arnoldi:
        print("Consider using arnoldi or pytorch to speed up. Numpy used for eigenvalues.")
        mode_eig = "numpy"

    # args.reduced_version = False
    # args.reduced_version = False
    reduced_version = args.reduced_version

    # home = "/bigwork/dscheier"
    # # experiment_suffix = "gpu_02_16_dip_1comp"
    # # experiment_suffix = "gpu_02_17_paper_1comp"

    # # experiment_suffix = "gpu_02_06_no_V_1comp"
    # # args.dir_name = "movie050"

    # # experiment_suffix = "gpu_02_20_lhy_1comp"
    # # experiment_suffix = "gpu_02_22_lhy_1comp"
    # # experiment_suffix = "gpu_02_22_no_dipol_no_lhy_1comp"

    # # experiment_suffix = "gpu_02_22_no_dipol_no_lhy_1comp_w100"
    # # args.dir_name = "movie001"

    # # experiment_suffix = "gpu_02_22_no_dipol_no_lhy_1comp_w_paper"
    # # args.dir_name = "movie001"
    # # args.dir_name = "movie010"
    # # args.dir_name = "movie030"

    # # experiment_suffix = "gpu_02_23_no_dipol_no_lhy_1comp_w_paper"
    # # args.dir_name = "movie004"

    # experiment_suffix = "gpu_02_27_no_dipol_no_lhy_1comp_w_paper"
    # args.dir_name = "movie017"
    # # args.dir_name = "movie012"
    # # args.dir_name = "movie016"

    # args.dir_path = Path(f"{home}/results/begin_{experiment_suffix}/")
    # args.filename_schroedinger = "schroedinger.pkl"
    # args.filename_steps = "step_"
    # args.steps_format = "%07d"
    # args.frame = None
    # # args.mode = "dask"
    # args.mode = "flat"
    # # args.dipol = True
    # args.dipol = False
    # args.l_0 = None
    # args.ground_state = True

    # n = 2
    # args.nx = n
    # args.ny = n
    # args.nz = n
    # # args.nx = 18
    # # args.ny = 11
    # # args.nz = 11
    # # args.recalculate = False
    # args.recalculate = True
    # args.print_num_eigenvalues = 100

    # args.graphs_dirname = "graphs"
    # args.label = ""

    ######## END OF USER INPUT #####################################################################
    with run_time(name="Full BdG"):
        try:
            dir_path = Path(args.dir_path).expanduser()
        except Exception:
            dir_path = args.dir_path

        path_graphs = Path(dir_path, args.graphs_dirname)
        path_log = Path(dir_path, "log")
        params_string = f"{args.dir_name}_{args.label}{mode_eig}_{args.nx}_{args.ny}_{args.nz}_{args.mode}_{args.stepper_x}_{args.stepper_y}_{args.stepper_z}"
        path_log_file = Path(path_log, f"BdG_{params_string}.log")
        path_result = Path(path_graphs, f"BdG_{params_string}.npz")
        path_result_ev = Path(path_graphs, f"BdG_{params_string}_ev_print.npz")
        path_result_txt = Path(path_graphs, f"BdG_{params_string}.txt")
        path_bogoliubov = Path(path_graphs, f"Matrix_BdG_{params_string}.npz")
        path_bogoliubov_sparse = Path(path_graphs, f"Sparse_Matrix_BdG_{params_string}.npz")
        path_bogoliubov_csr_txt = Path(path_graphs, f"Sparse_Matrix_BdG_{params_string}.txt")
        path_bogoliubov_csr_row = Path(path_graphs, f"Sparse_Matrix_BdG_row_{params_string}.txt")
        path_bogoliubov_csr_col = Path(path_graphs, f"Sparse_Matrix_BdG_col_{params_string}.txt")
        path_bogoliubov_csr_val = Path(path_graphs, f"Sparse_Matrix_BdG_val_{params_string}.txt")
        path_bog_helper = Path(path_graphs, f"BdG_bog_helper_{params_string}.npz")
        path_hermite_matrix = Path(path_graphs, f"BdG_hermite_matrix_{params_string}.npz")
        path_hermite_lhy_abc_list = Path(path_graphs, f"BdG_hermite_lhy_abc_list_{params_string}.npz")

        if not path_graphs.is_dir():
            path_graphs.mkdir(parents=True)

        if not path_log.is_dir():
            path_log.mkdir(parents=True)

        logging.basicConfig(filename=path_log_file, encoding='utf-8', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        if args.frame is None:
            _, last_index, _, _ = get_path.get_path(Path(dir_path, args.dir_name),
                                                    search_prefix=args.filename_steps,
                                                    counting_format=args.steps_format,
                                                    file_pattern=".npz")
            frame = last_index
        else:
            frame = args.frame

        System = get_System_at_npz(dir_path=dir_path,
                                   dir_name=f"{args.dir_name}",
                                   filename_schroedinger=args.filename_schroedinger,
                                   filename_steps=args.filename_steps,
                                   steps_format=args.steps_format,
                                   frame=frame,
                                   )
        print(f"mu: {System.mu_arr}")

        System.stack_shift = 0.0
        # x = System.x_mesh[:, System.Res.y // 2, System.Res.z // 2]
        # y = System.y_mesh[System.Res.x // 2, :, System.Res.z // 2]
        # z = System.z_mesh[System.Res.x // 2, System.Res.y // 2, :]
        # print(x)
        # print(y)
        # print(z)

        System.dV = System.volume_element()
        print(f"System.dV: {System.dV}")

        if not cupy_used:
            # convert to numpy
            # System_np = System.copy_with_all_numpy()
            System_np = SchroedingerMixtureNumpy(System)
            numpy_path = Path(dir_path, args.dir_name, "schroedinger_numpy.pkl")
            with open(numpy_path, "wb") as f:
                dill.dump(obj=System_np, file=f)

        if args.l_0 is None:
            # x harmonic oscillator length
            l_0 = np.sqrt(constants.hbar / (System.m_list[0] * System.w_x))
        else:
            l_0 = args.l_0

        if args.mode == "lin_op":
            # nx = args.nx
            # ny = args.ny
            # nz = args.nz

            stepper_x = args.stepper_x
            stepper_y = args.stepper_y
            stepper_z = args.stepper_z
            nx = args.nx // stepper_x
            ny = args.ny // stepper_y
            nz = args.nz // stepper_z
            # Res = Resolution(*(System.Res.to_array() // stepper))
            Res = Resolution(System.Res.x // stepper_x, System.Res.y // stepper_y, System.Res.z // stepper_z)
            # System.x, System.dx, System.kx, System.dkx = functions.get_grid_helper(self.Res, self.Box, 0)
            # System.y, System.dy, System.ky, System.dky = functions.get_grid_helper(self.Res, self.Box, 1)
            # System.z, System.dz, System.kz, System.dkz = functions.get_grid_helper(self.Res, self.Box, 2)
            _, _, kx, _ = functions.get_grid_helper(Res, System.Box, 0)
            _, _, ky, _ = functions.get_grid_helper(Res, System.Box, 1)
            _, _, kz, _ = functions.get_grid_helper(Res, System.Box, 2)
            kx_mesh, ky_mesh, kz_mesh = cp.meshgrid(cp.array(kx), cp.array(ky), cp.array(kz), indexing="ij", sparse=True)
            V_3d_ddi = functools.partial(functions.dipol_dipol_interaction,
                                         r_cut=0.98 * max(System.Box.lengths()),
                                         use_cut_off=True,
                                         )
            V_k_val = V_3d_ddi(kx_mesh, ky_mesh, kz_mesh)
            # cupy_used = False
            if not cupy_used:
                V_k_val = cp.asnumpy(V_k_val)
                kx_mesh = cp.asnumpy(kx_mesh)
                ky_mesh = cp.asnumpy(ky_mesh)
                kz_mesh = cp.asnumpy(kz_mesh)

            if System.lhy_factor != 0.0:
                g_qf = functions.get_g_qf_bog(N=System.N_list[0],
                                              a_s=float(System.a_s_array[0, 0]),
                                              a_dd=float(System.a_dd_array[0, 0]))
            else:
                g_qf = 0.0
            print(f"g_qf: {g_qf}")
            print(f"System.N_list[0]: {System.N_list[0]}")
            print(f"System.a_s_array[0, 0]: {System.a_s_array[0, 0]}")
            print(f"System.a_dd_array[0, 0]: {System.a_dd_array[0, 0]}")
            print(f"l_0: {l_0}")

            # Full loaded System
            if cupy_used:
                psi_real = cp.real(System.psi_val_list[0][::stepper_x, ::stepper_y, ::stepper_z])
                V_val_cut = cp.array(System.V_val[::stepper_x, ::stepper_y, ::stepper_z])
                # V_k_val = cp.array(V_k_val)
            else:
                psi_real = np.real(System.psi_val_list[0][::stepper_x, ::stepper_y, ::stepper_z])
                V_val_cut = System.V_val[::stepper_x, ::stepper_y, ::stepper_z]
            BogOperatorObjMax = BogOperator(psi_real,
                                            System.lhy_factor * g_qf,
                                            System.N_list[0] * System.a_s_array[0, 0],
                                            System.N_list[0] * System.a_dd_array[0, 0],
                                            System.a_s_factor,
                                            System.a_dd_factor,
                                            V_val_cut,
                                            V_k_val,
                                            kx_mesh, ky_mesh, kz_mesh,
                                            System.mu_arr[0],
                                            nx, ny, nz,
                                            eigen_max=0.0,
                                            cupy_used=cupy_used,
                                            )

            dV = System.volume_element() * stepper_x * stepper_y * stepper_z
            # dV = System.volume_element()
            # H_sandwich = BogOperatorObjMax.test_H(dV)
            # H_sandwich_v = BogOperatorObjMax.test_v(dV)
            # H_sandwich_kin = BogOperatorObjMax.test_kin(dV)
            norm_psi = BogOperatorObjMax.norm_psi(dV)
            # print(f"H_sandwich_kin: {H_sandwich_kin}")
            # print(f"H_sandwich_v: {H_sandwich_v}")
            # print(f"H_sandwich: {H_sandwich}")
            print(f"norm_psi: {norm_psi}")
            print(f"mu: {System.mu_arr[0]}")
            print(f"Run eigs for max eigen value with stepper: [{stepper_x}, {stepper_y}, {stepper_z}] and Res {Res.to_array()}")
            # print(f"Run eigs for max eigen value:")
            with run_time(name="eig max"):
                eigen_max_list = eigs(BogOperatorObjMax, k=1, which="LM", tol=10**-8, return_eigenvectors=False)
                eigen_max = np.real(eigen_max_list[0])
                print(f"eigen_max: {eigen_max}")

            eigen_max_factor = 0.6
            eigen_max_adjusted = eigen_max_factor * eigen_max
            print(f"eigen_max_factor: {eigen_max_factor}")
            print(f"eigen_max_adjusted: {eigen_max_adjusted}")
            
            if cupy_used:
                mu_Bog = cp.real(cp.sum(psi_real * BogOperatorObjMax.gpe_ham(psi_real, mu_factor=0.0))) * dV
            else:
                mu_Bog = np.real(np.sum(psi_real * BogOperatorObjMax.gpe_ham(psi_real, mu_factor=0.0))) * dV
            print(f"mu_Bog: {mu_Bog}")

            BogOperatorObj = BogOperator(psi_real,
                                         System.lhy_factor * g_qf,
                                         System.N_list[0] * System.a_s_array[0, 0],
                                         System.N_list[0] * System.a_dd_array[0, 0],
                                         System.a_s_factor,
                                         System.a_dd_factor,
                                         V_val_cut,
                                         V_k_val,
                                         kx_mesh, ky_mesh, kz_mesh,
                                         mu_Bog,
                                         # System.mu_arr[0],
                                         nx, ny, nz,
                                         eigen_max=eigen_max_adjusted,
                                         cupy_used=cupy_used,
                                         )

            # print(f"Run eigs with stepper: {stepper} and Res {Res.to_array()}")
            print(f"Run eigs for max eigen value:")
            with run_time(name="eig"):
                eigen_values = eigs(BogOperatorObj, k=args.arnoldi_num_eigs, which="LM", tol=10**-5, return_eigenvectors=False)
                eigen_values = np.sort(np.real(eigen_values))
                eigen_vectors = None
            # print(f"sqrt eigen_values:\n{eigen_values}")
            ev_print = np.sqrt(eigen_values + eigen_max_adjusted)
            print(f"sqrt eigen_values:\n{ev_print}")

            # sys.exit(0)
        else:
            index_dict = get_index_dict(args.nx, args.ny, args.nz)
            # TODO: uncomment
            # check_hermite_orthonormal(System, args.nx, args.ny, args.nz, index_dict)

            if path_result.exists() and not args.recalculate:
                print(f"Try loading solution as val, vec from: {path_result}")
                try:
                    with open(path_bogoliubov, "rb") as g:
                        bog = np.load(file=g)
                        bogoliubov_matrix = bog["bog"]
                        try:
                            ind_vec = bog["ind_vec"]
                        except Exception as e:
                            dim = bogoliubov_matrix.shape[0] // 2
                            ind_vec = np.arange(0, dim, 1)

                        mat2d(bogoliubov_matrix[0], "bog:")
                        # print(cp.real(bogoliubov_matrix))
                        # print(cp.imag(bogoliubov_matrix))
                        bog_real = np.where(np.real(bogoliubov_matrix) < 0, 0, np.real(bogoliubov_matrix))
                        bog_real = np.where(bog_real < 10 ** -10, 0, bog_real)
                        bog_imag = np.where(np.imag(bogoliubov_matrix) < 0, 0, np.imag(bogoliubov_matrix))
                        bog_imag = np.where(bog_imag < 10 ** -10, 0, bog_imag)
                        mat2d(bog_real, "real:\n", formatter={'float': '{:0.1f}'.format}, linewidth=550)
                        print(f"imag:\n{bog_imag}")
                        print(f"imag?:\n{np.all(bog_imag == 0)}")
                except Exception:
                    print(f"No File for the Bogoliubov matrix!")

                checked = check_sol(System, args.nx, args.ny, args.nz, bogoliubov_matrix, ind_vec, index_dict, reduced_version=reduced_version)

                with open(path_result, "rb") as f:
                    sol = np.load(file=f, allow_pickle=True)
                    val = sol["val"]
                    vec = sol["vec"]
                ev_sorted = np.sort(np.where(val < 0, 100000, val))
                ev_print = np.real(ev_sorted)[:args.print_num_eigenvalues]
                print(f"ev_print:\n{ev_print}")

                print(f"Loaded solution as val, vec from: {path_result}")
            else:
                density_list = System.get_density_list(jit=False, cupy_used=cupy_used)
                operator = density_list[0]
                print(f"max(operator): {np.max(operator)}")
                print(f"max(operator) / N: {np.max(operator) / System.N_list[0]}")

                # h_pol_0 = harmonic_eigenstate(System.x_mesh, 0, a=1)[:, System.Res.y // 2, System.Res.z // 2]
                # print(f"h_pol_0: {h_pol_0.tolist()}")
                # print(f"System.x: {System.x_mesh[:, System.Res.y // 2, System.Res.z // 2].tolist()}")

                if path_bogoliubov.is_file():
                    print(f"Try loading bogoliubov_matrix from: {path_bogoliubov}")
                    with open(path_bogoliubov, "rb") as g:
                        bogoliubov_matrix_load = np.load(file=g)
                        bogoliubov_matrix = bogoliubov_matrix_load["bog"]
                    print(f"Succesfully loaded bogoliubov_matrix from: {path_bogoliubov}")
                    cupy_used = False
                    loaded = True
                else:
                    loaded = False
                    if args.mode == "dask":
                        client = Client() 

                    print(f"Calculating bogoliubov_matrix")
                    with run_time(name="get_bogoliubov_matrix"):
                        bogoliubov_matrix, ind_vec = get_bogoliubov_matrix(System, operator,
                                                                           args.nx, args.ny, args.nz,
                                                                           index_dict,
                                                                           cupy_used,
                                                                           mode=args.mode, dipol=args.dipol,
                                                                           l_0=l_0, ground_state=args.ground_state,
                                                                           cut_hermite_orders=args.cut_hermite_orders,
                                                                           cut_hermite_values=args.cut_hermite_values,
                                                                           reduced_version=args.reduced_version)
                if cupy_used:
                    try:
                        bogoliubov_matrix = cp.asnumpy(bogoliubov_matrix)
                    except Exception as e:
                        traceback.print_tb(e.__traceback__)
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                    logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    pinned_mempool.free_all_blocks()
                    logging.info(f"{mempool.used_bytes() / (1024 ** 2)}MB")
                    logging.info(f"{mempool.total_bytes() / (1024 ** 2)}MB")
                    logging.info(f"pinned_mempool.n_free_blocks(): {pinned_mempool.n_free_blocks()}")

                if loaded:
                    print(f"Save bogoliubov matrix to: {path_bogoliubov}")
                    with open(path_bogoliubov, "wb") as g:
                        # np.savez_compressed(g, bog=bogoliubov_matrix, ind_vec=ind_vec)
                        np.savez_compressed(g, bog=bogoliubov_matrix)
                    print(f"Succesfully saved")

                try:
                    # bogoliubov_matrix_sparse = csr_matrix(np.real(bogoliubov_matrix))
                    # args.csr_cut_off_0 = 10 ** -8 * np.max(bogoliubov_matrix)
                    # args.csr_cut_off_0 = 10 ** -8
                    # bogoliubov_matrix_cut = np.where(np.abs(bogoliubov_matrix) < cut_off or (0.0 > bogoliubov_matrix > -csr_cut_off_0), 0.0, bogoliubov_matrix)
                    print(f"Cut_off_0: {args.csr_cut_off_0}")
                    print(f"min(abs(bog)): {np.min(np.abs(bogoliubov_matrix))}")
                    cut_condition_sparse = np.abs(bogoliubov_matrix) < args.csr_cut_off_0
                    number_cutted = np.sum(cut_condition_sparse)
                    print(f"Number of elements cutted to 0: {number_cutted} of {bogoliubov_matrix.size}, so {number_cutted/bogoliubov_matrix.size}")
                    bogoliubov_matrix_cut = np.where(cut_condition_sparse, 0.0, bogoliubov_matrix)
                    bogoliubov_matrix_sparse = csr_matrix(np.real(bogoliubov_matrix_cut))
                    bogoliubov_matrix_csr_row = bogoliubov_matrix_sparse.indptr
                    bogoliubov_matrix_csr_col = bogoliubov_matrix_sparse.indices
                    bogoliubov_matrix_csr_val = np.real(bogoliubov_matrix_sparse.data)
                    print(f"Succesfully converted to CSR")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)

                try:
                    print(f"Save bogoliubov matrix in CSR format: {path_bogoliubov_sparse}")
                    save_npz(path_bogoliubov_sparse, bogoliubov_matrix_sparse)
                    print(f"Succesfully saved")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)

                bogoliubov_matrix_sparse.maxprint = np.inf
                # bogoliubov_matrix_sparse.maxprint = np.inf
                # bogoliubov_matrix_csr_row.maxprint = np.inf
                # bogoliubov_matrix_csr_col.maxprint = np.inf
                # bogoliubov_matrix_csr_val.maxprint = np.inf

                try:
                    print(f"Save bogoliubov matrix in CSR format to txt: {path_bogoliubov_csr_txt}")
                    with open(path_bogoliubov_csr_txt, "w") as g:
                        print(bogoliubov_matrix_sparse, file=g)
                    print(f"Succesfully saved")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)

                try:
                    print(f"Save bogoliubov_matrix_csr_row matrix in CSR format to txt: {path_bogoliubov_csr_row}")
                    np.savetxt(path_bogoliubov_csr_row, bogoliubov_matrix_csr_row, delimiter='\n', fmt="%d")
                    # with open(path_bogoliubov_csr_row, "w") as g:
                    #     print(bogoliubov_matrix_csr_row, file=g)
                    print(f"Succesfully saved")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)

                try:
                    print(f"Save bogoliubov_matrix_csr_col matrix in CSR format to txt: {path_bogoliubov_csr_col}")
                    np.savetxt(path_bogoliubov_csr_col, bogoliubov_matrix_csr_col, delimiter='\n', fmt="%d")
                    # with open(path_bogoliubov_csr_col, "w") as g:
                    #     print(bogoliubov_matrix_csr_col, file=g)
                    print(f"Succesfully saved")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)

                try:
                    print(f"Save bogoliubov_matrix_csr_val matrix in CSR format to txt: {path_bogoliubov_csr_val}")
                    np.savetxt(path_bogoliubov_csr_val, bogoliubov_matrix_csr_val, delimiter='\n')
                    # with open(path_bogoliubov_csr_val, "w") as g:
                    #     print(bogoliubov_matrix_csr_val, file=g)
                    print(f"Succesfully saved")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)

                try:
                    print(f"Loading sparse bogoliubov matrix in CSR format: {path_bogoliubov_sparse}")
                    bogoliubov_matrix_sparse = load_npz(path_bogoliubov_sparse)
                    print(f"Succesfully loaded")
                except Exception as e:
                    print(f"Failed!")
                    traceback.print_tb(e.__traceback__)
                # print(f"sparse:\n{bogoliubov_matrix_sparse}")
                print(f"sparse:\n{bogoliubov_matrix_sparse.shape}")

                if args.get_eigenvalues:
                    print(f"Calculating eigenvalues")
                    with run_time(name="eig"):
                        if pytorch:
                            print("Using pytorch")
                            bogoliubov_matrix = torch.from_numpy(bogoliubov_matrix)
                            eigen_values, eigen_vectors = torch.linalg.eig(bogoliubov_matrix)
                            # bogoliubov_matrix_cut = torch.from_numpy(bogoliubov_matrix_cut)
                            # eigen_values, eigen_vectors = torch.linalg.eig(bogoliubov_matrix_cut)
                            eigen_values = torch.Tensor.numpy(eigen_values)
                            eigen_vectors = torch.Tensor.numpy(eigen_vectors)
                        elif arnoldi:
                            print("Using scipy (arnoldi)")
                            logging.info(f"bogoliubov_matrix: {round(bogoliubov_matrix.nbytes / 1024 ** 2, 2)}MB")
                            print(args.arnoldi_num_eigs)
                            # eigen_values = eigs(bogoliubov_matrix, k=args.arnoldi_num_eigs, which="SM", tol=10**-7, return_eigenvectors=False)
                            # eigen_values = eigs(bogoliubov_matrix_cut, k=args.arnoldi_num_eigs, which="SM", tol=10**-7, return_eigenvectors=False)

                            # gpu_diag = True
                            gpu_diag = False
                            if gpu_diag:
                                # bogoliubov_matrix_cp = cp.array(bogoliubov_matrix)
                                # eigsh has no SM
                                # eigen_values = cp_eigsh(bogoliubov_matrix_cp, k=args.arnoldi_num_eigs, which="SA", tol=10**-7, return_eigenvectors=False)
                                # eigen_values = eigsh(bogoliubov_matrix_cp, k=args.arnoldi_num_eigs, which="SA", tol=10**-7, return_eigenvectors=False)
                                is_hermite = bogoliubov_matrix - np.conjugate(bogoliubov_matrix).T
                                logging.info(f"is_hermite: {is_hermite}")
                                logging.info(f"is_hermite min: {np.min(is_hermite)}")
                                logging.info(f"is_hermite max: {np.max(is_hermite)}")

                                eigen_values = eigsh(bogoliubov_matrix, k=args.arnoldi_num_eigs, which="SM", tol=10**-7, return_eigenvectors=False)
                            else:
                                eigen_values = eigs(bogoliubov_matrix, k=args.arnoldi_num_eigs, which="SM", tol=10**-7, return_eigenvectors=False)
                            eigen_vectors = None
                        else:
                            print("Using numpy")
                            eigen_values, eigen_vectors = np.linalg.eig(bogoliubov_matrix)
                            # eigen_values, eigen_vectors = np.linalg.eig(bogoliubov_matrix_cut)

                    if loaded:
                        checked = check_sol(System, args.nx, args.ny, args.nz, bogoliubov_matrix, ind_vec, index_dict, reduced_version=reduced_version)

                    ev_sorted = np.sort(np.where(eigen_values < 0, np.inf, eigen_values))
                    ev_print = np.real(ev_sorted)[:args.print_num_eigenvalues]
                    if reduced_version:
                        ev_print = np.sqrt(ev_print)
                        print(f"Cut_off_0: {args.csr_cut_off_0}")
                        print(f"sqrt ev_print:\n{ev_print}")
                    else:
                        print(f"Cut_off_0: {args.csr_cut_off_0}")
                        print(f"ev_print:\n{ev_print}")

        print(f"Save solution as val, vec to: {path_result}")
        with open(path_result, "wb") as g:
            np.savez_compressed(g, val=eigen_values, vec=eigen_vectors)
        print(f"Succesfully saved")

        print(f"Save solution as ev_print to: {path_result_ev}")
        with open(path_result_ev, "wb") as g:
            np.savez_compressed(g, val=ev_print, vec=None)
        print(f"Succesfully saved")

        try:
            print(f"Save solution as ev_print as txt: {path_result_txt}")
            np.savetxt(path_result_txt, ev_print, delimiter='\n')
            print(f"Succesfully saved")
        except Exception as e:
            print("Failed")

        try:
            client.close()
        except Exception as e:
            print("No dask client to close!")

