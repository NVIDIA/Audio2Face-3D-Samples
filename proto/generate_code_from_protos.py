#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import subprocess
import sys

import yaml

PYTHON_OPTIONS = ["--python_out=", "--grpc_python_out="]
PYTHON_CMD = "python3 -m grpc_tools.protoc"

ARG_CONFIG_FILE = 'config_file'
ARG_PARALLEL = 'parallel'


def make_parser():
    # Create the argument parser.
    parser = argparse.ArgumentParser(description='Generate code from proto files '
                                                 'for a specified programming language.')
    # Add arguments.
    parser.add_argument(ARG_CONFIG_FILE, type=str,
                        help='Path to the configuration file listing files to process.')
    parser.add_argument("-p","--parallel", dest=ARG_PARALLEL, action='store_true',
                        help="execute proto gen in parallel")
    # Parse the arguments.
    args = parser.parse_args()
    return vars(args)


def process_one_file(src_file, src_folders, dst_folder, python_cmd, python_options):
    """
    Generate python gRPC code for the given proto file.
    """
    cmd = f"{python_cmd} "
    for option in python_options:
        cmd += f"{option}{dst_folder} "
    for folder in src_folders:
        cmd += f"-I {folder} "
    cmd += f"{src_file} "
    print(f"$ {cmd}")

    p = subprocess.Popen(cmd, shell=True)
    return p


def generate_files(src_file_list, dst_folder, src_folders, is_parallel_enabled):
    """
    Generates python gRPC code from proto files.
    """
    # Remove the folder if it already exits and create a new one.
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)

    error= False
    list_process = []
    for file in src_file_list:
        # Process each proto file in the given list and collect overall status.
        print(f"\n###\nProcessing {file}...")
        p = process_one_file(file, src_folders, dst_folder, PYTHON_CMD, PYTHON_OPTIONS)
        list_process.append(p)
        if not is_parallel_enabled:
            p.wait()
            error |= p.returncode != 0
            if error:
                break
    # Wait for the parallel processing of the proto files to end.
    if is_parallel_enabled:
        for p in list_process:
            p.wait()
            error |= p.returncode != 0

    if error:
        print("ERROR: not files were generated!")
        return False
    print("SUCCESS: all files generated!")
    return True


def get_files_folders_from_yaml(yaml_path):
    """
    Gets a list of all proto files and their folders as specified in the yaml
    configuration.

    Expected yaml format is:
        folder1:
            - file1
            - file2
    """
    with open(yaml_path, "r") as f:
        yml_dt = yaml.safe_load(f.read())

    list_src_folders = []
    list_all_files = []
    for folder, list_file in yml_dt["files"].items():
        list_src_folders.append(folder)
        for file in list_file:
            path = os.path.join(folder, file)
            list_all_files.append(path)

    return list_all_files, list_src_folders


def generate_init_py(root_folder):
    """
    Recursively generate __init__.py in all folders of the specified root_folder.
    """
    for dirpath, _, _ in os.walk(root_folder):
        # Check if __init__.py exists in the current directory
        init_py_path = os.path.join(dirpath, "__init__.py")
        if not os.path.exists(init_py_path):
            # If it doesn't exist, create it
            open(init_py_path, 'a').close()
            print(f"Created __init__.py in: {dirpath}")
        # No need to manually dive into subdirectories, os.walk does that automatically


def main():
    # Parse command line arguments.
    args = make_parser()
    is_parallel_enabled = args[ARG_PARALLEL] is True if ARG_PARALLEL in args else False
    cfg_f = args[ARG_CONFIG_FILE]
    # Get the list of proto files from the yaml config file.
    if not os.path.isfile(cfg_f):
        raise FileNotFoundError(f"Missing file {cfg_f}")
    cfg_file = os.path.abspath(os.path.expanduser(cfg_f))
    file_list, src_folders = get_files_folders_from_yaml(cfg_file)

    print("Generating python code from protos...")
    res = generate_files(file_list, "generated", src_folders, is_parallel_enabled)
    print("Done.")
    if not res:
        sys.exit(1)

    # For python, generate __init__.py file at top the level and in each
    # submodule.
    os.chdir("generated")
    files_python = [elm for elm in  os.listdir(".") if os.path.isfile(elm)]
    for elm in files_python:
        list_elm = elm.split(".")
        path = "/".join(list_elm[:-2])
        file = ".".join(list_elm[-2:])
        fpath = os.path.join(path, file)
        if elm != fpath:
            os.makedirs(path, exist_ok=True)
            shutil.copy2(elm, fpath)
            os.remove(elm)
    generate_init_py(".")


if __name__ == '__main__':
    main()
