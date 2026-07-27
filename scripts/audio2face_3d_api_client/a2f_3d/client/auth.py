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

import grpc
from pathlib import Path
import os
from typing import List, Optional, Tuple, Union

def create_channel(ssl_cert: Optional[Union[str, os.PathLike]] = None,
        uri= "grpc.nvcf.nvidia.com:443", use_ssl: bool = False, metadata: Optional[List[Tuple[str, str]]] = None) -> grpc.Channel:
    def metadata_callback(context, callback):
        callback(metadata, None)
        
    if ssl_cert is not None or use_ssl:
        root_certificates = None
        if ssl_cert is not None:
            ssl_cert = Path(ssl_cert).expanduser()
            with open(ssl_cert, 'rb') as f:
                root_certificates = f.read()
        creds = grpc.ssl_channel_credentials(root_certificates)
        if metadata:
            auth_creds = grpc.metadata_call_credentials(metadata_callback)
            creds = grpc.composite_channel_credentials(creds, auth_creds)
        channel = grpc.aio.secure_channel(uri, creds)
    else:
        channel = grpc.aio.insecure_channel(uri)
    return channel

class Auth:
    def __init__(
        self,
        ssl_cert: Optional[Union[str, os.PathLike]] = None,
        use_ssl: bool = False,
        uri: str = "localhost:50052",
        metadata_args: List[List[str]] = None,
    ) -> None:
        """
        A class responsible for establishing connection with a server and providing security metadata.

        Args:
            ssl_cert (:obj:`Union[str, os.PathLike]`, `optional`): a path to SSL certificate file. If :param:`use_ssl`
                is :obj:`False` and :param:`ssl_cert` is not :obj:`None`, then SSL is used.
            use_ssl (:obj:`bool`, defaults to :obj:`False`): whether to use SSL. If :param:`ssl_cert` is :obj:`None`,
                then SSL is still used but with default credentials.
            uri (:obj:`str`, defaults to :obj:`"localhost:50051"`): a Riva URI.
        """
        self.ssl_cert: Optional[Path] = None if ssl_cert is None else Path(ssl_cert).expanduser()
        self.uri: str = uri
        self.use_ssl: bool = use_ssl
        self.metadata = []
        if metadata_args:
            for meta in metadata_args:
                if len(meta) != 2:
                    raise ValueError(f"Metadata should have 2 parameters in \"key\" \"value\" pair. Receieved {len(meta)} parameters.")
                self.metadata.append(tuple(meta))
        self.channel: grpc.Channel = create_channel(self.ssl_cert, uri=self.uri, use_ssl=self.use_ssl, metadata=self.metadata)

    def get_auth_metadata(self) -> List[Tuple[str, str]]:
        """
        Will become useful when API key and OAUTH tokens will be enabled.

        Metadata for authorizing requests. Should be passed to stub methods.

        Returns:
            :obj:`List[Tuple[str, str]]`: an empty list.
        """
        metadata = []
        return metadata