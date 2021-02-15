#!/usr/bin/env python
from pathlib import Path
from typing import Tuple


def get_path(dir_path: Path,
             dir_name: str = "movie",
             counting_format: str = "%03d") -> Tuple[Path, int, str, str]:
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
    existing = sorted([x for x in dir_path.glob(dir_name + "*") if x.is_dir()])
    try:
        last_index: int = int(existing[-1].name.split(dir_name)[1])
    except IndexError as e:
        assert last_index is not None, (
            "Extracting last index from dir_path failed")

    input_path = Path(dir_path, dir_name + counting_format % last_index)

    return input_path, last_index, dir_name, counting_format

