#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np
import dask.array as da
from dask.distributed import Client

from scipy.special import eval_hermite, factorial
from scipy.special import hermite
from scipy.sparse.linalg import eigs

from supersolids.helper import constants, functions, get_path, get_version
from supersolids.helper.run_time import run_time

__GPU_OFF_ENV__, __GPU_INDEX_ENV__ = get_version.get_env_variables()
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np,
                                                               gpu_off=__GPU_OFF_ENV__,
                                                               gpu_index=__GPU_INDEX_ENV__)

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.tools.get_System_at_npz import get_System_at_npz


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
    parser.add_argument("-l_0", metavar="l_0", type=float, default=None,
                        help="Help constant for dimensionless formulation of equations.")
    parser.add_argument("-label", type=str, default="", help="Label to name result dirnames.")

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


def mat2d(mat, label="", precision=4, formatter={'float': '{:0.3f}'.format}, linewidth=250):

    with cp.printoptions(precision=precision, suppress=True, formatter=formatter,
                         linewidth=linewidth):
        if cupy_used:
            print(f"{label}\n{np.matrix(cp.asnumpy(mat))}") 
        else:
            print(f"{label}\n{np.matrix(mat)}") 



def harmonic_eigenstate(x, n):
    prefactor = np.sqrt(1 / (factorial(n) * 2 ** n )) * np.pi ** -0.25
    herm = hermite(n)(x)
    result = prefactor * np.exp(-x ** 2 / 2) * herm

    return result


def HO_1D(ind, x, a=1):
    return (1./np.sqrt(2**ind * factorial(ind) * np.sqrt(np.pi)) * a**0.25
            * np.exp(-a * x**2/2) * eval_hermite(ind, np.sqrt(a) * x))


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


def operator_ravel(operator, pos, nymax, nzmax):
    pos_x, pos_y, pos_z = indices(pos, nymax, nzmax)
    operator_list = []
    index_list = []
    for ix, iy, iz in zip(pos_x[0, :], pos_y[0, :], pos_z[0, :]):
        operator_list.append(operator[ix, iy, iz])
        index_list.append((ix, iy, iz))
       
    operator_ordered = cp.array(operator_list)

    return operator_ordered


def position_revert(x, y, z, xmax, ymax, zmax, nxmax, nymax, nzmax):
    dx, dy, dz = 2 * xmax/nxmax, 2 * ymax/nymax, 2 * zmax/nzmax
    pos_x = (x + xmax) / dx
    pos_y = (y + ymax) / dy
    pos_z = (z + zmax) / dz

    pos = np.vectorize(int)(pos_z + pos_x * nymax * nzmax + pos_y * nzmax)

    return pos


def HO_3D(x, y, z, ind, ind_y_max, ind_z_max, ay=1, az=1):
    ax = 1
    ind_x, ind_y, ind_z = indices(ind, ind_y_max, ind_z_max)
    # herm_3d = HO_1D(ind_x, x, ax) * HO_1D(ind_y, y, ay) * HO_1D(ind_z, z, az)
    herm_3d = HO_1D(ind_x, x, ax) * HO_1D(ind_y, y, ax) * HO_1D(ind_z, z, ax)
    if cupy_used:
        herm_3d = cp.array(herm_3d)

    return herm_3d


def get_hermite_dipol(System, comb1, comb2, ground_state=True, entry="A"):
    phi_i = harmonic_eigenstate_3d_dask(System.x_mesh, System.y_mesh, System.z_mesh,
                                        comb1[0], comb1[1], comb1[2])
    phi_j = harmonic_eigenstate_3d_dask(System.x_mesh, System.y_mesh, System.z_mesh,
                                        comb2[0], comb2[1], comb2[2])

    if ground_state:
        op = System.psi_val_list[0]
        op_in_fft = op
    else:
        if entry == "A":
            op = System.psi_val_list[0]
            op_in_fft = cp.conjugate(System.psi_val_list[0])
        elif entry == "B":
            op = System.psi_val_list[0]
            op_in_fft = System.psi_val_list[0]
        elif entry == "C":
            op = cp.conjugate(System.psi_val_list[0])
            op_in_fft = cp.conjugate(System.psi_val_list[0])
        elif entry == "D":
            op = cp.conjugate(System.psi_val_list[0])
            op_in_fft = System.psi_val_list[0]
        else:
            sys.exit("No such entry. Choose between A, B, C, D.")

    op = op / System.N_list[0]
    op_in_fft = op_in_fft / System.N_list[0]

    if cupy_used:
       op  = cp.array(op)
       op_in_fft  = cp.array(op_in_fft)
       V_k_val = cp.array(System.V_k_val)

    herm_dip = phi_j * cp.fft.ifftn(V_k_val *
                                    cp.fft.fftn(phi_i * op_in_fft)
                                    ) * op
     
    hermite_dipol = System.sum_dV(cp.real(herm_dip), fourier_space=False) 

    return hermite_dipol


def En_TF(nr,l):
    return np.sqrt(2 * nr**2 + 2 * nr * l + 3 * nr + l)


def En(ind, ind_y_max, ind_z_max, ay=1, az=1):
    ind_x, ind_y, ind_z = indices(ind, ind_y_max, ind_z_max)
    return (ind_x + 0.5) + (ay ** -2) * (ind_y + 0.5) + (az ** -2) * (ind_z + 0.5)

    
def harmonic_eigenstate_3d(System, i, j, k):
    harmonic_eigenstate_2d = harmonic_eigenstate(System.x_mesh, i) * harmonic_eigenstate(System.y_mesh, j)
    harmonic_eigenstate_3d = harmonic_eigenstate_2d * harmonic_eigenstate(System.z_mesh, k)

    return cp.asarray(harmonic_eigenstate_3d)


def harmonic_eigenstate_3d_dask(x_mesh, y_mesh, z_mesh, i, j, k):
    harmonic_eigenstate_2d = harmonic_eigenstate(x_mesh, i) * harmonic_eigenstate(y_mesh, j)
    harmonic_eigenstate_3d = harmonic_eigenstate_2d * harmonic_eigenstate(z_mesh, k)

    return cp.asarray(harmonic_eigenstate_3d)


def hermite_transform(System, operator, comb1, comb2,
                      fourier_space: bool = False, dV: float = None, sandwich = True):
    if (operator is None) or (comb1 is None) or (comb2 is None):
        return cp.array(0.0)
    integrand = harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2]) * operator
    if sandwich:
        integrand = integrand * harmonic_eigenstate_3d(System, comb2[0], comb2[1], comb2[2])

    transform: float = System.sum_dV(integrand, fourier_space=fourier_space, dV=dV)
    
    return transform

def hermite_transform_dask(dV, x_mesh, y_mesh, z_mesh, operator, comb1, comb2):
    if (operator is None) or (comb1 is None) or (comb2 is None):
        return cp.array(0.0)
    integrand = (harmonic_eigenstate_3d_dask(x_mesh, y_mesh, z_mesh, comb1[0], comb1[1], comb1[2]) * operator \
                 * harmonic_eigenstate_3d_dask(x_mesh, y_mesh, z_mesh, comb2[0], comb2[1], comb2[2]))
    transform: float = cp.sum(integrand) * dV

    return transform


def get_index_dict(nx, ny, nz):
    index_dict = {i * ny * nz + j * nz + k : [i, j, k]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)}

    return index_dict

def get_bog_dict(index_dict, dim, maxi=0.00001):
    comb_list_list = []
    sum_list_list = []
    for j in range(dim):
        ind = np.ravel(np.argwhere(np.abs(bogoliubov_matrix[j, : dim]) > maxi))
        comb_list = [index_dict[ind[i]] for i in range(len(ind))]
        sum_list = list(map(sum , [index_dict[ind[i]] for i in range(len(ind))]))
        comb_list_list.append(comb_list)
        sum_list_list.append(sum_list)
        
    return comb_list_list, sum_list_list

def get_parity(comb1, comb2):
    assert len(comb1) == len(comb2)
    summed = np.array(comb1) + np.array(comb2)
    parity = len([i for i in summed if i % 2 == 0]) == len(comb1)

    return parity

def get_hermit_matrix(index_dict, System, operator, dim):
    hermite_matrix = cp.zeros((dim, dim))
    E_H0 = cp.zeros((dim, dim))
    triu_0, triu_1 = np.triu_indices(dim)
    for l, m in zip(triu_0, triu_1):
        comb1 = index_dict[l]
        comb2 = index_dict[m]
        if l == m:
            E_H0[l, m] = ((comb1[0] + 0.5)
                          + (System.w_y / System.w_x) * (comb1[1] + 0.5)
                          + (System.w_z / System.w_x) * (comb1[2] + 0.5))
        # with run_time(name=f"{l},{m} integrated"):
        if get_parity(comb1, comb2):
            hermite_matrix[l, m] = hermite_transform(System, operator, comb1, comb2)
        else:
            hermite_matrix[l, m] = 0.0

    return hermite_matrix, E_H0


def get_hermit_matrix_dask(index_dict, System, operator, dim, fast = True):
    hermite_matrix = cp.zeros((dim, dim))
    E_H0 = cp.zeros((dim, dim))
    triu_0, triu_1 = np.triu_indices(dim)
    if fast:
        dV = System.volume_element(fourier_space=False)
        [dV_dask, x_dask, y_dask, z_dask, operator_dask] = client.scatter([dV,
                                                                           System.x_mesh,
                                                                           System.y_mesh,
                                                                           System.z_mesh,
                                                                           operator])
    else:
        [System_dask, operator_dask] = client.scatter([System, operator])

    futures = []
    for l, m in zip(triu_0, triu_1):
        comb1 = index_dict[l]
        comb2 = index_dict[m]
        if l == m:
            E_H0[l, m] = ((comb1[0] + 0.5)
                          + (System.w_y / System.w_x) * (comb1[1] + 0.5)
                          + (System.w_z / System.w_x) * (comb1[2] + 0.5))
        if not get_parity(comb1, comb2):
            comb1 = None
            comb2 = None
        if fast:
            futures.append(client.submit(hermite_transform_dask, dV_dask, x_dask, y_dask, z_dask,
                                         operator_dask, comb1, comb2))
        else:
            futures.append(client.submit(hermite_transform, System_dask, operator_dask, comb1, comb2))
    results = client.gather(futures)

    #  put results into correct array shape
    results_arr = cp.array(results)
    for i, (l, m) in enumerate(zip(triu_0, triu_1)):
        hermite_matrix[l, m] = results_arr[i]

    return hermite_matrix, E_H0

def get_hermit_matrix_flat(index_dict, System, operator, nx, ny, nz, dipol=False, ground_state=True):
    dim = int(nx * ny * nz)
    pos_max = System.Res.x * System.Res.y * System.Res.z
    pos_vec = np.arange(0, pos_max, 1)
    ind_vec = np.arange(0, int(nx * ny * nz), 1)
    ind_v, pos_v = np.meshgrid(ind_vec, pos_vec, indexing='ij')
    dV = System.volume_element(fourier_space=False)
    x, y, z = position(System, pos_v) 

    ax, ay, az = 1, np.sqrt(System.w_x / System.w_y), np.sqrt(System.w_x / System.w_z)
    E_H0 = cp.diag(En(ind_vec, ny, nz, ay=ay, az=az))
    bog_helper = HO_3D(x, y, z, ind_v, ny, nz, ay=ay, az=az)

    if dipol:
        bog_helper_swapped = cp.swapaxes(bog_helper, 0, 1)

    # bring operator in same order as x
    operator_raveled = operator_ravel(operator, pos_v, System.Res.y, System.Res.z)
    with run_time(name=f"flat hermite_matrix {nx} {ny} {nz}"):
        if dipol:
            hermite_matrix = cp.dot(bog_helper * operator_raveled, bog_helper_swapped) * dV
        else:
            hermite_matrix = cp.dot(bog_helper * operator_raveled, cp.swapaxes(bog_helper, 0, 1)) * dV

    hermite_dipol_abcd_list = []
    hermite_lhy_abc_list = []
    lhy_abc_list = []
    if dipol:
        with run_time(name=f"dipol all {nx} {ny} {nz}"):
            matrix_entries = ["A", "B", "C", "D"]
            lhy_entries = 3
            for i, key in enumerate(matrix_entries):
                hermite_dipol_abcd_list.append(cp.zeros((dim, dim)))
            triu_0, triu_1 = np.triu_indices(dim)
            with run_time(name=f"hermite_dipol {nx} {ny} {nz}"):
                for i, entry in enumerate(matrix_entries):
                    with run_time(name=f"hermite_dipol {i}"):
                        for l, m in zip(triu_0, triu_1):
                            comb1 = index_dict[l]
                            comb2 = index_dict[m]
                            hermite_dipol_abcd_list[i][l, m] = get_hermite_dipol(
                                                                  System, comb1, comb2,
                                                                  ground_state=ground_state,
                                                                  entry=entry)
                        print(f"{i}:\n{hermite_dipol_abcd_list[i]}")
                hermite_dipol_abcd_list = map(functions.symmetric_mat, hermite_dipol_abcd_list)

            g_qf = functions.get_g_qf_bog(N=System.N_list[0],
                                          a_s=float(System.a_s_array[0, 0]),
                                          a_dd=float(System.a_dd_array[0, 0]))
            print(f"g_qf: {g_qf}")

            density_list = System.get_density_list(jit=False, cupy_used=cupy_used)

            density_raveled = operator_ravel(density_list[0], pos_v, System.Res.y, System.Res.z)
            psi_val_raveled = operator_ravel(System.psi_val_list[0], pos_v, System.Res.y, System.Res.z)
            density_raveled_by_N = density_raveled / (System.N_list[0] ** 2)
            psi_val_raveled_by_N = psi_val_raveled / System.N_list[0]

            with run_time(name=f"lhy_abc_list"):
                for i in range(lhy_entries):
                    lhy_abc_list.append(cp.zeros((dim, dim)))

                lhy_abc_list[0] = g_qf * 2.5 * density_raveled_by_N ** 1.5
                if ground_state:
                    print(f"ground_state: {ground_state}")
                    lhy_b = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled_by_N ** 2
                    lhy_abc_list[1] = lhy_b
                    lhy_abc_list[2] = lhy_b
                else:
                    lhy_abc_list[1] = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * psi_val_raveled_by_N ** 2
                    lhy_abc_list[2] = g_qf * 1.5 * (density_raveled_by_N ** 0.5) * cp.conjugate(psi_val_raveled_by_N) ** 2

            with run_time(name=f"lhy all {nx} {ny} {nz}"):
                for lhy_key in lhy_abc_list:
                    hermite_lhy_abc_list.append(cp.dot(bog_helper * lhy_key, bog_helper_swapped) * dV)

    return hermite_matrix, E_H0, hermite_dipol_abcd_list, hermite_lhy_abc_list


def get_bogoliuv_matrix(System, operator, nx, ny, nz, mode="dask", dipol=False, l_0=None,
                        ground_state=True):
    # contact_interaction_vec, dipol_term_vec, _ = System.get_dipol_U_dd_mu_lhy()

    # contact_interaction_vec = cp.array(contact_interaction_vec)
    # dipol_term_vec = cp.array(dipol_term_vec)
    System.V_val = cp.array(System.V_val)
    System.k_squared = cp.array(System.k_squared)

    # np cp conversion
    # for i, (contact_interaction, dipol_term, mu_lhy) in enumerate(zip(
    #         list(contact_interaction_vec), list(dipol_term_vec), mu_lhy_list)):
    #     term: np.ndarray = System.get_H_pot_exponent_terms(dipol_term,
    #                                                        contact_interaction,
    #                                                        mu_lhy
    #                                                        )

    grid_shape = System.x_mesh.shape
    dim = int(nx * ny * nz)

    index_dict = get_index_dict(nx, ny, nz)

    if mode == "dask":
        with run_time(name=f"{mode} {nx} {ny} {nz}"):
            hermite_matrix_triu, E_H0, bog_helper = get_hermit_matrix_dask(index_dict, System,
                                                                           operator, dim, fast=True)
        hermite_matrix = functions.symmetric_mat(hermite_matrix_triu)
    elif mode == "cupy":
        with run_time(name=f"{mode} {nx} {ny} {nz}"):
            hermite_matrix_triu, E_H0, hermite_dipol, hermite_lhy = get_hermit_matrix(index_dict, System, operator, dim)
        hermite_matrix = functions.symmetric_mat(hermite_matrix_triu)
    elif mode == "flat":
        with run_time(name=f"{mode} {nx} {ny} {nz}"):
            hermite_matrix, E_H0, hermite_dipol_abcd_list, hermite_lhy_abc_list = get_hermit_matrix_flat(index_dict, System, operator, nx, ny, nz,
                                                                                                         dipol=dipol, ground_state=ground_state)
    else:
        sys.exit("Mode not implemented. Choose between dask, cupy, flat.")

    g = 4.0 * np.pi * System.a_s_array[0, 0]
    # mu = functions.mu_3d(g * System.N_list[0])
    mu = System.mu_arr[0]
    # mu = float(g * np.max(operator))
    # mu = 3.833 # no_dipol_no_lhy_w_paper
    # print(f"mu by hand: {mu}, g: {g}")

    b = g * hermite_matrix
    a = E_H0 + 2.0 * b - cp.diag(dim * [mu])
    # a = np.diag(En(ind_vec, ny, nz) - mu) + 2.0 * b

    matrix = cp.zeros((2*dim, 2*dim), dtype=cp.complex_)

    if dipol:
        hermite_dipol_a_triu, hermite_dipol_b_triu, hermite_dipol_c_triu, hermite_dipol_d_triu = hermite_dipol_abcd_list
        hermite_lhy_a, hermite_lhy_b, hermite_lhy_c = hermite_lhy_abc_list

        matrix[0:dim, 0:dim] = a + hermite_dipol_a_triu + hermite_lhy_a
        matrix[0:dim, dim:] = -b - hermite_dipol_b_triu - hermite_lhy_b
        matrix[dim:, 0:dim] = b + hermite_dipol_c_triu + hermite_lhy_c
        matrix[dim:, dim:] = -a - hermite_dipol_d_triu - hermite_lhy_a

    else:
        matrix[0:dim, 0:dim] = a
        matrix[0:dim, dim:] = -b
        matrix[dim:, 0:dim] = b
        matrix[dim:, dim:] = -a

    mat2d(matrix, "bog:")
    
    return matrix

    
def hermite_laplace(System, i):
    k = 2
    factor = 2 ** k * (factorial(i) / np.math.factorial(i - k))
    herm_laplace = factor * harmonic_eigenstate_3d(System, i - k)
    
    return herm_laplace

def check_sol(System, nx, ny, nz, bog_mat):
    dim = int(nx * ny * nz)
    psi_0 = cp.zeros(dim)

    operator = cp.real(System.psi_val_list[0])

    index_dict = get_index_dict(nx, ny, nz)

    for l in range(dim):
        comb1 = index_dict[l]
        psi_0[l] = hermite_transform(System, operator, comb1, comb1, sandwich=False)
   
    psi_0_2dim = cp.hstack((psi_0, psi_0))
    norm = cp.dot(psi_0, psi_0)
    print(f"Norm psi_0: {norm}")
    
    # if cupy_used:
    #     bog_mat = cp.array(bog_mat)
    #     psi_0_2dim = cp.array(psi_0_2dim)
    # result = cp.dot(bog_mat, psi_0_2dim)
    psi_0_2dim = cp.asnumpy(psi_0_2dim)
    result = np.einsum("ij,j->i", bog_mat, psi_0_2dim)
    
    # result_0 = np.where(np.abs(result) > 0.0001, result, 0)
    # mat2d(psi_0_2dim, "psi_0_2dim:")
    mat2d(result, "result:")
    
    return result


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])

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

    # experiment_suffix = "gpu_02_22_no_dipol_no_lhy_1comp_w_paper"
    # # args.dir_name = "movie001"
    # # args.dir_name = "movie010"
    # args.dir_name = "movie030"

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

    # # n = 7
    # # args.nx = n
    # # args.ny = n
    # # args.nz = n
    # args.nx = 12
    # args.ny = 5
    # args.nz = 5
    # # args.recalculate = False
    # args.recalculate = True
    # args.print_num_eigenvalues = 100

    # args.graphs_dirname = "graphs"
    # args.label = ""

    ######## END OF USER INPUT #####################################################################

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path
    path_graphs = Path(dir_path, args.graphs_dirname)
    path_result = Path(path_graphs,
                       f"BdG_{args.dir_name}_{args.label}{args.nx}_{args.ny}_{args.nz}_{args.mode}.npz")
    path_bogoliubov = Path(path_graphs,
                           f"Matrix_BdG_{args.dir_name}_{args.label}{args.nx}_{args.ny}_{args.nz}"
                           + f"_{args.mode}.npz")

    if not path_graphs.is_dir():
        path_graphs.mkdir(parents=True)

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
    print(f"{System}")
    print(f"mu: {System.mu_arr}")
    System.stack_shift = 0.0

    if args.l_0 is None:
        # x harmonic oscillator length
        l_0 = np.sqrt(constants.hbar / (System.m_list[0] * System.w_x))
    else:
        l_0 = args.l_0

    if path_result.exists() and not args.recalculate:
        try:
            with open(path_bogoliubov, "rb") as g:
                bog = np.load(file=g)
                bogoliubov_matrix = bog["bog"]
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

        checked = check_sol(System, args.nx, args.ny, args.nz, bogoliubov_matrix)

        with open(path_result, "rb") as f:
            sol = np.load(file=f)
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

        if args.mode == "dask":
            client = Client() 
        bogoliubov_matrix = get_bogoliuv_matrix(System, operator, args.nx, args.ny, args.nz,
                                                mode=args.mode, dipol=args.dipol, l_0=l_0,
                                                ground_state=args.ground_state)
        if cupy_used:
            bogoliubov_matrix = cp.asnumpy(bogoliubov_matrix)

        with run_time(name="eig"):
            eigen_values, eigen_vectors = np.linalg.eig(bogoliubov_matrix)
            # eigen_values, eigen_vectors = eigs(bogoliubov_matrix, k=10, which="SM")

        checked = check_sol(System, args.nx, args.ny, args.nz, bogoliubov_matrix)

        ev_sorted = np.sort(np.where(eigen_values < 0, 100000, eigen_values))
        ev_print = np.real(ev_sorted)[:args.print_num_eigenvalues]
        print(f"ev_print:\n{ev_print}")

        print(f"Save solution as val, vec to: {path_result}")
        with open(path_result, "wb") as g:
            np.savez_compressed(g, val=eigen_values, vec=eigen_vectors)
        print(f"Succesfully saved")

        print(f"Save bogoliubov matrix to: {path_bogoliubov}")
        with open(path_bogoliubov, "wb") as g:
            np.savez_compressed(g, bog=bogoliubov_matrix)
        print(f"Succesfully saved")
