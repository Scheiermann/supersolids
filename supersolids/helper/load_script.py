#!/usr/bin/env python
from shutil import copy
from pathlib import Path
from typing import Optional, List


def reload_files(dir_path: Path, dir_name_load: str,
                 result_path: Path, script_name: str,
                 script_number_regex: str = '*',
                 script_extensions: Optional[List[str]] = None):
    if script_extensions is None:
        script_extensions = [".pkl", ".txt"]

    path_load = Path(dir_path, dir_name_load)
    files_per_extension_list = []
    for filename_extension in script_extensions:
        files_per_extension_list.append(
            sorted([x for x
                    in path_load.glob(script_name + script_number_regex + filename_extension)
                    if x.is_file()])
            )

    copy_from(files_per_extension_list, result_path)

    return files_per_extension_list


def copy_from(files_per_extension_list, result_path):
    # copy old loaded files to new result_path
    for files_per_extension in files_per_extension_list:
        for file_fixed_extension in files_per_extension:
            copy(file_fixed_extension,
                 Path(result_path, file_fixed_extension.name)
                 )

