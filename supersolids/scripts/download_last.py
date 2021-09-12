#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

import fnmatch
from pathlib import Path

from fabric import Connection


if __name__ == "__main__":
    path_anchor_input = Path("/bigwork/dscheier/supersolids/supersolids/results/begin_alpha/")
    path_anchor_output = Path("/run/media/dsche/ITP Transfer/begin_alpha/")

    # take_last = 30
    take_last = None

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 107
    movie_end = 426

    filename_steps = "step_"
    steps_format = "%07d"
    filename_pattern = ".npz"
    filename_number_regex = '*'
    # filename_number_regex = '*0000'

    filename_singles = ["schroedinger.pkl", "distort.txt"]
    download_steps = True

    for i in range(movie_start, movie_end + 1):
        path_in = Path(path_anchor_input, movie_string + f"{counting_format % i}")
        path_out = Path(path_anchor_output, movie_string + f"{counting_format % i}")

        print(f"\npath_in: {path_in}")
        with Connection('itpx') as c:
            # Create a results dir, if there is none
            if not path_out.is_dir():
                path_out.mkdir(parents=True)

            if filename_singles:
                print(path_out)
                files_single_already_there = sorted([
                    x for x in filename_singles if Path(path_out, x).is_file()
                    ])

                print(files_single_already_there)
                filenames_singles_new = [x for x in filename_singles if (x not in files_single_already_there)]
                print(f"Possible files to download: {filenames_singles_new}")
                for filename_single in filenames_singles_new:
                    try:
                        file_single = fnmatch.filter(c.sftp().listdir(path=str(path_in)),
                                                     filename_single)

                        if len(file_single) > 0:
                            print(f"Downloading {filename_single}")
                            result_schroedinger = c.get(str(Path(path_in, filename_single)),
                                                        local=str(Path(path_out, filename_single)))
                        else:
                            print(f"{Path(path_in, filename_single)} not found. Skipping.")
                            continue

                    except Exception:
                        print(f"Some error getting {Path(path_in, filename_single)}. Skipping.")
                        continue

            if download_steps:
                try:
                    files_all = sorted(fnmatch.filter(c.sftp().listdir(path=str(path_in)),
                                                      filename_steps
                                                      + filename_number_regex
                                                      + filename_pattern))
                except FileNotFoundError:
                    print(f"{path_in} not found. Skipping.")
                    continue
                if take_last is None:
                    files = files_all
                else:
                    files = files_all[-take_last:]

                files_already_there = sorted([x.name for x
                                              in path_out.glob(filename_steps
                                                               + filename_number_regex
                                                               + filename_pattern)
                                              if x.is_file()])

                files_new = [x for x in files if (x not in files_already_there)]
                print("Downloading:")
                print(files_new)

                for file in files_new:
                    result = c.get(str(Path(path_in, file)), local=str(Path(path_out, file)))
