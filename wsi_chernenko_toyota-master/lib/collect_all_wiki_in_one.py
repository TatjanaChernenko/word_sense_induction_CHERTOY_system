#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Tatjana Chernenko, Utaemon Toyota
@usage: python3 collect_all_wiki_in_one.py input_directory_path output_txt_file_path
@course: Formale Semantik WS 2017/18
@description: Collects all Wikipedia Dump Texts into one large text file.
"""

from pathlib import Path
import sys

rootdir_glob = "/proj/toyota/plain2/"
target_file = "/proj/toyota/all_plain_text2.txt"

def collect_all_files_in_one(input_path = rootdir_glob, output_path = target_file):
    rootdir = Path(input_path)
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]
    for file in file_list:
        with open(str(file),"r") as input:
            with open(output_path, "a") as output:
                output.write(input.read())

if __name__ == "__main__":
    if len(sys.argv) == 1:
        collect_all_files_in_one()
    elif (len(sys.argv)) == 3:
        collect_all_files_in_one(sys.argv[1],sys.argv[2])
    else:
        print("@usage: python3 input_directory_path output_txt_file_path")