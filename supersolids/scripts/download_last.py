#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

import fnmatch
from pathlib import Path

from fabric import Connection


def download(host, path_in, path_out, filename_singles, download_steps, take_last):
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
                file_single = fnmatch.filter(host.sftp().listdir(path=str(path_in)),
                                             filename_single)

                if len(file_single) > 0:
                    print(f"Downloading {filename_single}")
                    result_schroedinger = host.get(str(Path(path_in, filename_single)),
                                                local=str(Path(path_out, filename_single)))
                else:
                    print(f"{Path(path_in, filename_single)} not found. Skipping.")
                    continue

            except Exception:
                print(f"Some error getting {Path(path_in, filename_single)}. Skipping.")
                continue

    if download_steps:
        for filename_steps, steps_format, filename_pattern, filename_number_regex in zip(
                filename_steps_list, steps_format_list,
                filename_pattern_list, filename_number_regex_list):

            try:
                files_all = sorted(fnmatch.filter(host.sftp().listdir(path=str(path_in)),
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
                result = host.get(str(Path(path_in, file)), local=str(Path(path_out, file)))


if __name__ == "__main__":
    ssh_hostname = 'transfer'
    password = None

    path_anchor_input = Path("/bigwork/nhbbsche/results/begin_gpu_big/")
    path_anchor_output = Path("/bigwork/dscheier/results/begin_gpu_big/")
    # path_anchor_output = Path("/run/media/dsche/scr2/begin_gpu/")
    # path_anchor_output = Path("/run/media/dsche/scr2/begin_gpu_big/")


    # take_last = 3
    take_last = None

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 1
    movie_end = 6

    mixture = True

    if mixture:
        filename_steps_list = ["script_", "schroedinger_",
                               "mixture_step_", "SchroedingerMixtureSummary_"]
    else:
        filename_steps_list = ["script_", "schroedinger_",
                               "step_", "SchroedingerSummary_"]
    steps_format_list = ["%04d", "%04d", "%07d", "%07d"]
    filename_pattern_list = [".pkl", ".pkl", ".npz", ".pkl"]
    filename_number_regex_list = ['*', '*', '*', '*']
    # filename_number_regex = '*0000'

    filename_singles = ["schroedinger.pkl", "script.txt"]
    download_steps = True

    for i in range(movie_start, movie_end + 1):
        path_in = Path(path_anchor_input, movie_string + f"{counting_format % i}")
        path_out = Path(path_anchor_output, movie_string + f"{counting_format % i}")

        print(f"\npath_in: {path_in}")
        if password is None:
            with Connection(ssh_hostname) as host:
                download(host, path_in, path_out, filename_singles, download_steps, take_last)
        else:
            with Connection(ssh_hostname, connect_kwargs={'password': f"{password}"}) as host:
                download(host, path_in, path_out, filename_singles, download_steps, take_last)