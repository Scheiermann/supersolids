#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Convert python dictionary to str, which can be used to provide flags to commandline
"""


def dic2str(dic, single_quote_wrapped=True):
    dic_str = str(dic).replace("\'", "\"")
    if single_quote_wrapped:
        dic_str = f"'{dic_str}'"

    return dic_str
