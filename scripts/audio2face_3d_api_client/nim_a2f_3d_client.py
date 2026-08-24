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

import argparse, asyncio
import a2f_3d.client.auth
import a2f_3d.client.service
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                        description="Sample python application to send audio and receive animation data and emotion data through the Audio2Face-3D API.",
                        epilog="NVIDIA CORPORATION.  All rights reserved.")
    parser.add_argument("file", help="PCM-16 bits single channel audio file in WAV container to be sent to the Audio2Face-3D service")
    parser.add_argument("config", help="Configuration file for inference models")
    parser.add_argument("--apikey", type=str, required=True, help="NGC API Key to invoke the API function")
    parser.add_argument("--function-id", type=str, required=True, default="", help="Function ID to invoke the API function")
    return parser.parse_args()

async def main():
    args = parse_args()

    metadata_args = [("function-id", args.function_id), ("authorization", "Bearer " + args.apikey)]
    # Open gRPC channel and get Audio2Face-3D stub
    channel = a2f_3d.client.auth.create_channel(uri="grpc.nvcf.nvidia.com:443", use_ssl=True, metadata=metadata_args)
            
    stub = A2FControllerServiceStub(channel)

    stream = stub.ProcessAudioStream()
    write = asyncio.create_task(a2f_3d.client.service.write_to_stream(stream, args.config, args.file))
    read = asyncio.create_task(a2f_3d.client.service.read_from_stream(stream))

    try:
        await asyncio.gather(write, read)
    finally:
        write.cancel()
        read.cancel()
        await asyncio.gather(write, read, return_exceptions=True)
        await channel.close()

if __name__ == "__main__":
    asyncio.run(main())
