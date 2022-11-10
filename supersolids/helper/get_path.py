#!/usr/bin/env python

import fnmatch

from pathlib import Path
from typing import Tuple, List, Optional
from stat import S_ISDIR, S_ISREG


def get_path(dir_path: Path,
             search_prefix: str = "movie",
             counting_format: Optional[str] = "%03d",
             file_pattern: str = "",
             take_last: int = 1,
             host=None) -> Tuple[Optional[Path], Optional[int], str, str]:
    """
    Looks up all directories with matching dir_name
    and counting format in dir_path.
    Gets the highest number and returns a path with dir_name counted one up
    (prevents colliding with old data).

    :param dir_path: Path where to look for old directories (movie data)

    :param search_prefix: General name of the directories without the counter

    :param counting_format: Format of counter of the directories

    :param file_pattern : File extension to search for e.g. ".npz".

    :param take_last : Index of list of path counted from behind.

    :return: Path for the new directory (not colliding with old data)

    """

    # "movie" and "%03d" strings are hardcoded
    # in mayavi movie_maker _update_subdir
    if dir_path is None:
        input_path: Optional[Path] = None
        last_index: Optional[int] = None
    else:
        if file_pattern:
            # for files
            if host:
                sftp = host.sftp()
                files = [x.filename for x in sftp.listdir_attr(path=str(dir_path)) if S_ISREG(x.st_mode)]
                existing = sorted(fnmatch.filter(files, search_prefix + "*"))
                existing = [Path(dir_path, file) for file in existing]
            else:
                existing = sorted([x for x in dir_path.glob(search_prefix + "*") if x.is_file()])
            if take_last <= len(existing):
                last_index = get_step_index(existing[-take_last],
                                            filename_prefix=search_prefix,
                                            file_pattern=file_pattern)
            else:
                last_index = None
        else:
            # for dirs
            if host:
                sftp = host.sftp()
                files = [x.filename for x in sftp.listdir_attr(path=str(dir_path)) if S_ISDIR(x.st_mode)]
                existing = sorted(fnmatch.filter(files, search_prefix + "*"))
                existing = [Path(dir_path, file) for file in existing]
            else:
                existing = sorted([x for x in dir_path.glob(search_prefix + "*") if x.is_dir()])
            try:
                last_index = int(existing[-take_last].name.split(search_prefix)[1])
            except IndexError as e:
                last_index = 0
                print(f"No old data found. Setting last_index={last_index}.")

        if file_pattern:
            try:
                if counting_format is None:
                    input_path = Path(dir_path, search_prefix + file_pattern)
                else:
                    input_path = Path(dir_path,
                                      search_prefix + counting_format % last_index + file_pattern)
            except Exception:
                input_path = None
        else:
            try:
                if counting_format is None:
                    input_path = Path(dir_path, search_prefix)
                else:
                    input_path = Path(dir_path, search_prefix + counting_format % last_index)
            except Exception:
                input_path = None

    return input_path, last_index, search_prefix, counting_format


def get_step_index_from_list(dir_paths: List[Path], filename_prefix: str = "step_",
                             file_pattern: str = ".npz") -> int:
    try:
        if isinstance(dir_paths, list):
            # dir_paths has just one path
            dir_path: Path = dir_paths[-1]
        else:
            # dir_paths has just one path
            dir_path = dir_paths

        last_index: int = get_step_index(dir_path,
                                         filename_prefix=filename_prefix,
                                         file_pattern=file_pattern)

    except Exception as e:
        last_index = 0
        print(f"Could not extract last_index: {e}. Setting last_index={last_index}.")

    return last_index


def get_step_index(dir_path: Path, filename_prefix: str = "step_",
                   file_pattern: str = ".npz") -> int:
    try:
        last_str_part = dir_path.name.split(file_pattern)[0]
        last_index: int = int(last_str_part.split(filename_prefix)[1])
    except Exception as e:
        last_index = 0
        print(f"Could not extract last_index from {dir_path} "
              f"and filename_prefix {filename_prefix}: {e}\n"
              f"Setting last_index={last_index}.")

    return last_index


def get_last_png_in_last_anim(path_movie, dir_name_png, counting_format_png, movie_take_last,
                              filename_pattern, filename_format, filename_extension,
                              frame_format=None):
    # gets last movie with animations of each movie
    path_last_movie_png, _, _, _ = get_path(
        path_movie,
        search_prefix=f"{dir_name_png}",
        counting_format=counting_format_png,
        file_pattern="",
        take_last=movie_take_last,
        )
    if path_last_movie_png is not None:
        # gets last anim.png of each last movie with animations
        path_last_png, _, _, _ = get_path(
            path_last_movie_png,
            search_prefix=filename_pattern,
            counting_format=filename_format,
            file_pattern=filename_extension,
            )
        if path_last_movie_png is not None:
            # try other frame_format instead of filename_format
            if not path_last_png.exists():
                if frame_format:
                    path_last_png, _, _, _ = get_path(
                        path_last_movie_png,
                        search_prefix=filename_pattern,
                        counting_format=frame_format,
                        file_pattern=filename_extension,
                        )
        else:
            print(f"No last png in {path_last_movie_png} found.")
            
    else:
        path_last_png = None

    return path_last_png
