#!/usr/bin/env python

from pathlib import Path

import dill

from supersolids.helper.get_supersolids_version import get_supersolids_version


def save_script(script_count_old, input_path, script_name, obj_to_save,
                txt_version: bool = False,
                script_format="%04d"):
    if txt_version:
        with open(Path(input_path, script_name + ".txt"), "a") as script_file:
            script_file.write(f"supersolids: {get_supersolids_version()}\n\n")
            script_file.write(f"{vars(obj_to_save)}\n")
            script_file.write(f"--------------------------------------------------\n")

    script_count = script_count_old + 1
    script_path_pkl = Path(input_path, script_name + f"_{script_format % script_count}.pkl")
    try:
        with open(script_path_pkl, "wb") as f:
            dill.dump(obj=obj_to_save, file=f)
            print(f"Namespace of script saved under: {script_path_pkl}")
    except Exception:
        print(f"Saving script under {script_path_pkl} did NOT work!")
