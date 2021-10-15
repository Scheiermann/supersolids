#!/usr/bin/env python

from pathlib import Path
from typing import Tuple


def get_path(dir_path: Path,
             dir_name: str = "movie",
             counting_format: str = "%03d",
             file_pattern: str = "") -> Tuple[Path, int, str, str]:
    """
    Looks up all directories with matching dir_name
    and counting format in dir_path.
    Gets the highest number and returns a path with dir_name counted one up
    (prevents colliding with old data).

    :param dir_path: Path where to look for old directories (movie data)
    :param dir_name: General name of the directories without the counter
    :param counting_format: Format of counter of the directories

    :return: Path for the new directory (not colliding with old data)
    """

    # "movie" and "%03d" strings are hardcoded
    # in mayavi movie_maker _update_subdir
    if file_pattern:
        # for files
        existing = sorted([x for x in dir_path.glob(dir_name + "*") if x.is_file()])
        last_index = get_step_index(existing[-1],
                                    filename_prefix=dir_name,
                                    file_pattern=file_pattern)
    else:
        # for dirs
        existing = sorted([x for x in dir_path.glob(dir_name + "*") if x.is_dir()])

        try:
            last_index: int = int(existing[-1].name.split(dir_name)[1])
        except IndexError as e:
            last_index = 0
            print(f"No old data found. Setting last_index={last_index}.")

    input_path = Path(dir_path, dir_name + counting_format % last_index)

    return input_path, last_index, dir_name, counting_format


def get_step_index_from_list(dir_paths, filename_prefix="step_", file_pattern=".npz"):
    try:
        if isinstance(dir_paths, list):
            # dir_paths has just one path
            dir_path = dir_paths[-1]
        else:
            # dir_paths has just one path
            dir_path = dir_paths

        last_index = get_step_index(dir_path,
                                    filename_prefix=filename_prefix,
                                    file_pattern=file_pattern)

    except Exception as e:
        last_index = 0
        print(f"Could not extract last_index: {e}. Setting last_index={last_index}.")

    return last_index


def get_step_index(dir_path, filename_prefix="step_", file_pattern=".npz"):
    try:
        last_str_part = dir_path.name.split(file_pattern)[0]
        last_index: int = int(last_str_part.split(filename_prefix)[1])
    except Exception as e:
        last_index = 0
        print(f"Could not extract last_index from {dir_path} "
              f"and filename_prefix {filename_prefix}: {e}\n"
              f"Setting last_index={last_index}.")

    return last_index
