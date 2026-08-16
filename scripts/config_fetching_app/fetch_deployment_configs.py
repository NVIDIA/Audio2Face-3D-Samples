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

import grpc
from grpc import insecure_channel

from nvidia_ace.services.a2x_export_config import v1_pb2
from nvidia_ace.services.a2x_export_config.v1_pb2_grpc import A2XExportConfigServiceStub


OUTPUT_BASE_FOLDER = "output_yaml"

def capture_a2x_config_client(url, config_type=v1_pb2.ConfigsTypeRequest.YAML):

    channel = insecure_channel(url)
    # Create the client stub
    client = A2XExportConfigServiceStub(channel)

    # Prepare the request
    request = v1_pb2.ConfigsTypeRequest(config_type=config_type)
    list_cfg= []
    try:
        # Make the RPC call (streaming response)
        response_stream = client.GetConfigs(request)

        # Process the streaming response
        for response in response_stream:
            list_cfg.append((response.name, response.content))

    except grpc.RpcError as e:
        print(f"gRPC Error: {e}")
        return None
    return list_cfg


def get_yaml_configs(url):
    list_cfg_yaml = capture_a2x_config_client(url, v1_pb2.ConfigsTypeRequest.YAML)
    if list_cfg_yaml is None:
        print("Failed to retrieve configs; exiting.")
        return

    curr_index = 1
    curr_name = f"{OUTPUT_BASE_FOLDER}_{curr_index:06}"
    while os.path.exists(curr_name):
        curr_index += 1
        curr_name = f"{OUTPUT_BASE_FOLDER}_{curr_index:06}"

    os.makedirs(curr_name)
    for key, val in list_cfg_yaml:
        path_write=os.path.join(curr_name, key)
        with open(path_write, "w") as f:
            f.write(val)
        print(f"Writing config file to {path_write}")

def main():
    parser = argparse.ArgumentParser(description="Fetch YAML config from A2F-3D")
    parser.add_argument("url", type=str, help="The URL to process")

    args = parser.parse_args()
    get_yaml_configs(args.url)

if __name__ == "__main__":
    main()
