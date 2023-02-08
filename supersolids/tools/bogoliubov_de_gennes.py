#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import hermite
from scipy.sparse.linalg import eigs
from supersolids.helper import functions, get_path, get_version
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
    parser.add_argument("-graphs_dirname", type=str, default="graphs",
                        help="Name of directory for the results."
    parser.add_argument("--recalculate", default=False, action="store_true",
                        help="Ignores saved results for the parameters, "
                             "then recalculates and overwrites the old results.")
    parser.add_argument("-print_num_eigenvalues", type=int, default=20,
                        help="Number of eigenvalues printed.")
    parser.add_argument("-nx", type=int, default=4,
                        help="Number of Hermite polynomials used for x axis.")
    parser.add_argument("-ny", type=int, default=4,
                        help="Number of Hermite polynomials used for y axis.")
    parser.add_argument("-nz", type=int, default=4,
                        help="Number of Hermite polynomials used for z axis.")

    args = parser.parse_args(args_array)
    print(f"args: {args}")

    return args


def harmonic_eigenstate(x, n):
    prefactor = np.sqrt(1 / (np.math.factorial(n) * 2 ** n )) * np.pi ** -0.25
    herm = hermite(n)(x)
    result = prefactor * np.exp(-x ** 2 / 2) * herm

    return result

def get_index_dict(nx, ny, nz):
    dict = {i * ny * nz + j * nz + k : [i, j, k]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)}

    return dict

def get_bog_dict(dim, maxi=0.00001):
    comb_list_list = []
    sum_list_list = []
    for j in range(dim):
        ind = np.ravel(np.argwhere(np.abs(bogoliubov_matrix[j, : nx * ny * nz]) > maxi))
        comb_list = [dict[ind[i]] for i in range(len(ind))]
        sum_list = list(map(sum , [dict[ind[i]] for i in range(len(ind))]))
        comb_list_list.append(comb_list)
        sum_list_list.append(sum_list)
        
    return comb_list_list, sum_list_list

def get_parity(comb1, comb2):
    assert len(comb1) == len(comb2)
    summed = np.array(comb1) + np.array(comb2)
    parity = len([i for i in summed if i % 2 == 0]) == len(comb1)

    return parity


def get_bogoliuv_matrix(System, operator, nx, ny, nz):
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
    # buffer = cp.zeros((nx, ny, nz, *grid_shape))
    hermite_matrix = cp.zeros((dim, dim))
    E_H0 = cp.zeros((dim, dim))

    dict = get_index_dict(nx, ny, nz)
    triu_0, triu_1 = np.triu_indices(dim)
    with run_time(name="all"):
        for l, m in zip(triu_0, triu_1):
            comb1 = dict[l]
            comb2 = dict[m]
            E_H0[l, l] = ((comb1[0] + 0.5)
                          + (System.w_y / System.w_x) * (comb1[1] + 0.5)
                          + (System.w_z / System.w_x) * (comb1[2] + 0.5))
            with run_time(name=f"{l},{m} integrated"):
                if get_parity(comb1, comb2):
                    hermite_matrix[l, m] = hermite_transform(System, operator, comb1, comb2)
                else:
                    hermite_matrix[l, m] = 0.0
                g = 4.0 * np.pi * System.a_s_array[0, 0]

    hermite_matrix = functions.symmetric_mat(hermite_matrix)

    g = 4.0 * np.pi * System.a_s_array[0, 0]
    # mu = functions.mu_3d(g * System.N_list[0])
    mu = System.mu_arr[0]
    # mu = 14.408167497573881

    b = g * hermite_matrix
    a = E_H0 + 2.0 * b - cp.diag(dim * [mu])

    matrix = cp.zeros((2*dim, 2*dim))
    matrix[0:dim, 0:dim] = a
    matrix[0:dim, dim:] = b
    matrix[dim:, 0:dim] = -b
    matrix[dim:, dim:] = -a
    
    return matrix

    
def hermite_laplace(System, i):
    k = 2
    factor = 2 ** k * (np.math.factorial(i) / np.math.factorial(i - k))
    herm_laplace = factor * harmonic_eigenstate_3d(System, i - k)
    
    return herm_laplace


def hermite_transform(System, operator, comb1, comb2,
                      fourier_space: bool = False, dV: float = None):
    integrand = (harmonic_eigenstate_3d(System, comb1[0], comb1[1], comb1[2]) * operator \
                 * harmonic_eigenstate_3d(System, comb2[0], comb2[1], comb2[2]))
    transform: float = System.sum_dV(integrand, fourier_space=fourier_space, dV=dV)
    
    return transform
    
def harmonic_eigenstate_3d(System, i, j, k):
    harmonic_eigenstate_2d = harmonic_eigenstate(System.x_mesh, i) * harmonic_eigenstate(System.y_mesh, j)
    harmonic_eigenstate_3d = harmonic_eigenstate_2d * harmonic_eigenstate(System.z_mesh, k)

    return cp.asarray(harmonic_eigenstate_3d)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    args = flags(sys.argv[1:])

    # home = "/bigwork/dscheier"
    # experiment_suffix = "gpu_02_06_no_V_1comp"
    # args.dir_path = Path(f"{home}/results/begin_{experiment_suffix}/")
    # args.dir_name = "movie027"
    # args.filename_schroedinger = "schroedinger.pkl"
    # args.filename_steps = "step_"
    # args.steps_format = "%07d"
    # args.frame = None

    # args.nx = 16
    # args.ny = 16
    # args.nz = 16
    # args.recalculate = False
    # args.print_num_eigenvalues = 20

    # args.graphs_dirname = "graphs"

    ######## END OF USER INPUT #####################################################################

    try:
        dir_path = Path(args.dir_path).expanduser()
    except Exception:
        dir_path = args.dir_path
    path_graphs = Path(dir_path, args.graphs_dirname)
    path_result = Path(path_graphs, f"BdG_{args.dir_name}_{args.nx}_{args.ny}_{args.nz}.npz")
    path_bogoliubov = Path(path_graphs,
                           f"Matrix_BdG_{args.dir_name}_{args.nx}_{args.ny}_{args.nz}.npz")

    if not path_result.is_dir():
        path_result.mkdir(parents=True)
    if not path_bogoliubov.is_dir():
        path_bogoliubov.mkdir(parents=True)

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
    System.stack_shift = 0.0


    if path_result.exists() and not args.recalculate:
        try:
            with open(path_bogoliubov, "rb") as g:
                bog = np.load(file=g)
                bogoliubov_matrix = bog["bog"]
        except Exception:
            print(f"No File for the Bogoliubov matrix!")

        with open(path_result, "rb") as f:
            sol = np.load(file=f)
            val = sol["val"]
            vec = sol["vec"]
        vals_sorted = np.sort(val)
        vals_over0 = vals_sorted[vals_sorted > 0]
        print(vals_over0[:args.print_num_eigenvalues])
        vals_sorted2 = np.sort(np.abs(np.real(val)))
        print(vals_sorted2[:args.print_num_eigenvalues])
        print(f"Loaded solution as val, vec from: {path_result}")
    else:
        density_list = System.get_density_list(jit=False, cupy_used=cupy_used)
        operator = density_list[0]

        bogoliubov_matrix = get_bogoliuv_matrix(System, operator, args.nx, args.ny, args.nz)
        if cupy_used:
            bogoliubov_matrix = cp.asnumpy(bogoliubov_matrix)

        with run_time(name="eig"):
            eigen_values, eigen_vectors = np.linalg.eig(bogoliubov_matrix)
            # eigen_values, eigen_vectors = eigs(bogoliubov_matrix, k=10, which="SM")

        vals_over0 = np.sort(eigen_values)
        print(vals_over0[vals_over0 >= 0])
        vals_sorted2 = np.sort(np.abs(np.real(eigen_values)))
        print(vals_sorted2)

        print(f"Save solution as val, vec to: {path_result}")
        with open(path_result, "wb") as g:
            np.savez_compressed(g, val=eigen_values, vec=eigen_vectors)
        print(f"Succesfully saved")

        print(f"Save bogoliubov matrix to: {path_bogoliubov}")
        with open(path_bogoliubov, "wb") as g:
            np.savez_compressed(g, bog=bogoliubov_matrix)
        print(f"Succesfully saved")
