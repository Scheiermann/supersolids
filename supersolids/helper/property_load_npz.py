#!/usr/bin/env python

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def property_load_npz(path_property_npz: Path):
    with open(path_property_npz, "rb") as f:
        t = np.load(file=f)["t"]
        property_all = np.load(file=f)["property_all"]

    return t, property_all


if __name__ == "__main__":
    experiment_suffix = "ramp_fixed_0_test_24_8"
    # experiment_suffix = "pretilt0.05to100_test"
    path_anchor_input = Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/")

    png_suffix = "_lambda_test"

    filename = "E_paper_framestart_0"

    dir_name = "movie"
    counting_format = "%03d"
    movie_start = 1
    movie_end = 2

    # start = 0
    start = 1
    end = -1
    step = 1
    func = lambda x, y: (x[1:], np.abs(np.diff(y)) / np.abs(y[1:]))

    for i, movie_number in enumerate(range(movie_start, movie_end + 1)):
        movie_name = f"{dir_name}{counting_format % movie_number}"

        path_property_npz = Path(path_anchor_input, movie_name, f"{filename}.npz")
        t, property_all = property_load_npz(path_property_npz)
        x_range, y_range = t[start:end:step], property_all[start:end:step]
        x = x_range.ravel()
        y = y_range.ravel()
        plt.yscale('symlog')
        plt.plot(t, property_all.T.ravel(), "x-")
        # plt.plot(x, y_range, "x-")
        plt.xlabel(rf"t with dt=")
        plt.ylabel(f"{filename}")
        plt.grid()
        plt.title(f"lambda:")
        path_output = Path(path_property_npz.parent, path_property_npz.stem + f"{png_suffix}.png")
        print(f"Save to {path_output}")
        func_x, func_y = func(x, y)
        plt.savefig(path_output)