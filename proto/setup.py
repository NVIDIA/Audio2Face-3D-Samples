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

from setuptools import setup, find_packages

setup(
    name='nvidia-ace',
    version='1.2.0',
    packages=find_packages(),
    install_requires=[
        'PyYAML==6.0.1',
        'grpclib==0.4.7',
        'googleapis-common-protos==1.60.0',
        'protobuf==4.25.8',
        'protobuf-gen==0.0.4',
    ],
    # Metadata
    author='nvidia',
)
