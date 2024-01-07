#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Tatjana Chernenko, Utaemon Toyota
@usage: python3 preprocess_wikitext.py input_directory_path output_dir_file_path
@course: Formale Semantik WS 2017/18
@description: For each extractet Wikipedia text the programm will preprocess the text and removing e.g. all comments in parentheses
    and creating new files with each sentence in one line.
"""

import re
from pathlib import Path
import os
import errno
import sys

rootdir_glob = "/home/utaemon/Semantik_Projekt/results/"
plain_path = "/home/utaemon/Semantik_Projekt/plain/"

def remove_quotes(text):
    new_text = ""
    for token in text.split():
        # --- removes all unnecessary quotes
        temp_punct = ""
        if re.findall("(\?|:|!|\.|,|;)$", token):
            temp_punct = token[-1]
            token = token[:-1]
        if re.findall("''+",token):
            token = re.sub("''+","'",token)
        while token.startswith("'") or token.endswith("'") or token.startswith('"') or token.endswith('"'):
            if token.startswith("'"):
                token = token[1:]
            elif token.endswith("'"):
                token = token[:-1]
            elif token.startswith('"'):
                token = token[1:]
            elif token.endswith('"'):
                token = token[:-1]
        new_text += token + temp_punct + " "
    return new_text

def eliminate_brackets(text, patternstart, patternend):
    count_brackets = 0
    openbracket = patternstart
    closingbracket = patternend
    new_str = ""
    for token in text.split():
        if re.findall(openbracket, token) or re.findall(closingbracket, token):
            new_token = ""
            for char in token:
                if re.findall(openbracket, char):
                    count_brackets += 1
                elif re.findall(closingbracket, char):
                    count_brackets -= 1
                elif count_brackets == 0:
                    new_token += char
            new_str += new_token + " "
        elif count_brackets != 0:
            continue
        elif count_brackets == 0:
            new_str += token + " "
    return new_str

def get_plain_text(file_path):
    with open(str(file_path)) as file:
        text = ""
        get_title = False
        for line in file:
            if line.startswith("<doc"):
                get_title = True
                continue
            elif get_title == True:
                get_title = False
                continue
            elif line.startswith("</doc>"):
                continue
            else:
                text += line + " "
        text = remove_quotes(text)
        text = eliminate_brackets(text, "\(", "\)")
        text = re.sub("\.\.+","", text)
        text = re.sub(r'\s+(\?|:|!|\.|,|;)', r'\1', text)
        text = re.sub(r"\s\s+"," ", text)
        return text

def split_lines(function):
    pattern = r"(\?|:|!|\.|;)\s"
    text = re.split(pattern, function)
    new_text = ""
    sentence = True
    for elm in text:
        if sentence == True:
            new_text += elm
            sentence = False
        else:
            new_text += elm + "\n"
            sentence = True
    return new_text


#--------------write file
def write_plain_file(input_path = rootdir_glob, target_path=plain_path):
    file_list = [f for f in Path(input_path).glob('**/*') if f.is_file()]
    for file in file_list:
        print (file)
        file_li = str(file).split("/")
        dir_name = file_li[-2]
        file_name = file_li[-1]
        new_file_path = target_path + dir_name + "/" + file_name
        #https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
        if not os.path.exists(os.path.dirname(new_file_path)):
            try:
                os.makedirs(os.path.dirname(new_file_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        plain_text = get_plain_text(file)
        plain_text = split_lines(plain_text)
        with open(new_file_path, "w") as file:
            file.write(plain_text + "\n")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        write_plain_file()
    elif (len(sys.argv)) == 3:
        write_plain_file(sys.argv[1],sys.argv[2])
    else:
        print("@usage: python3 input_directory_path output_txt_file_path")
