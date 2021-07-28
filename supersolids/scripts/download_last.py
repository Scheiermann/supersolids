#!/usr/bin/env python
import fnmatch
from pathlib import Path

from fabric import Connection


if __name__ == "__main__":
    path_anchor_input = Path("/bigwork/dscheier/supersolids/results/")
    path_anchor_output = Path("/run/media/dsche/ITP Transfer/")

    take_last = 30

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 107
    movie_end = 426

    filename_steps = "step_"
    steps_format = "%07d"
    filename_pattern = ".npz"

    filename_singles = ["schroedinger.pkl", "distort.txt"]
    download_steps = True

    for i in range(movie_start, movie_end + 1):
        path_in = Path(path_anchor_input, movie_string + f"{counting_format % i}")
        path_out = Path(path_anchor_output, movie_string + f"{counting_format % i}")

        print(f"path_in: {path_in}")
        with Connection('itpx') as c:
            # Create a results dir, if there is none
            if not path_out.is_dir():
                path_out.mkdir(parents=True)

            if filename_singles:
                for filename_single in filename_singles:
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
                                                      filename_steps + '*' + filename_pattern))
                except FileNotFoundError:
                    print(f"{path_in} not found. Skipping.")
                    continue
                files = files_all[-take_last:]

                for file in files:
                    result = c.get(str(Path(path_in, file)), local=str(Path(path_out, file)))
