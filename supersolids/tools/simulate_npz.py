#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D in single-core.

"""
import pickle
from pathlib import Path

import numpy as np

from supersolids.Animation.Animation import Animation

from supersolids.Schroedinger import Schroedinger
from supersolids.tools.simulate_case import simulate_case

# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    movie_number = 9
    frame = 15570
    max_timesteps = 101
    movie_format: str = "%03d"
    filename_schroedinger = f"schroedinger.pkl"
    filename_steps = f"step_"
    steps_format: str = "%06d"

    dir_path = Path.home().joinpath("supersolids", "results")

    Anim: Animation = Animation(plot_psi_sol=False,
                                plot_V=False,
                                alpha_psi=0.8,
                                alpha_psi_sol=0.5,
                                alpha_V=0.3,
                                filename="anim.mp4",
                                )

    input_path = Path(dir_path, "movie" + movie_format % movie_number)

    schroedinger_path = Path(input_path, filename_schroedinger)
    psi_val_path = Path(input_path, filename_steps + steps_format % frame + ".npz")
    try:
        print("Load schroedinger")
        with open(schroedinger_path, "rb") as f:
            # WARNING: this is just the input Schroedinger at t=0
            System = pickle.load(file=f)

        print(f"File at {Path(input_path, filename_schroedinger)} loaded.")
        try:
            # get the psi_val of Schroedinger at other timesteps (t!=0)
            with open(psi_val_path, "rb") as f:
                System.psi_val = np.load(file=f)["psi_val"]

            System.max_timesteps = max_timesteps
            SystemResult: Schroedinger = simulate_case(
                System=System,
                Anim=Anim,
                accuracy=10 ** -12,
                delete_input=False,
                dir_path=dir_path,
                offscreen=True,
                x_lim=(-2.0, 2.0),  # from here just matplotlib
                y_lim=(-2.0, 2.0),
                z_lim=(0, 0.5),
            )

        except FileNotFoundError:
            print(f"File at {psi_val_path} not found.")

    except FileNotFoundError:
        print(f"File at {schroedinger_path} not found.")

