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
import asyncio
from sys import stderr
import os
import time
import warnings

import numpy as np
import grpc
import scipy
import yaml
import pandas as pd

from nvidia_ace.health_pb2_grpc import HealthStub
from nvidia_ace.health_pb2 import HealthCheckRequest, HealthCheckResponse

from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
from nvidia_ace.a2f.v1_pb2 import AudioWithEmotion, EmotionPostProcessingParameters, \
    FaceParameters, BlendShapeParameters, EmotionParameters
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode
from nvidia_ace.emotion_aggregate.v1_pb2 import EmotionAggregate


BITS_PER_BYTE = 8
# Bit depth of the audio file, only 16 bit PCM audio is currently supported.
REQUIRED_BITS_PER_SAMPLE = 16
# Channel count, only mono audio is currently supported.
REQUIRED_CHANNEL_COUNT = 1
# Audio format, only PCM is supported.
REQUIRED_AUDIO_FORMAT = AudioHeader.AUDIO_FORMAT_PCM
# Prefix of output folder
OUTPUT_BASE_FOLDER = "output"


def get_audio_bit_format(audio_header: AudioHeader):
    """
    Reads the audio_header parameters and returns the write type to interpret
    the audio data sent back by the server.
    """
    if audio_header.audio_format == AudioHeader.AUDIO_FORMAT_PCM:
        # We only support 16 bits PCM.
        if audio_header.bits_per_sample == 16:
            return np.int16
    return None


def save_audio_data_to_file(outdir: str, audio_header: AudioHeader, audio_buffer: bytes):
    """
    Reads the AudioHeader and output the content of the audio buffer into a wav
    file.
    """
    # Type of the audio data to output.
    dtype = get_audio_bit_format(audio_header)
    if dtype is None:
        print("Error while downloading data, unknown format for audio output", file=stderr)
        return

    audio_data_to_save = np.frombuffer(audio_buffer, dtype=dtype)
    # Write the audio data output as a wav file.
    scipy.io.wavfile.write(f"{outdir}/out.wav", audio_header.samples_per_second, audio_data_to_save)


def parse_emotion_data(animation_data, emotion_key_frames):
    """
    Fills the emotion key frames dictionnary using the data found in the emotion_aggregate metadata.

    Each emotion aggregate contains the following values:
    - input_emotions: Emotions that are manually inputed by the user.
    - a2e_output: The output of the emotion inference on the audio out of Audio2Emotion.
    - a2f_smoothed_output: The smoothed and post-processed emotions output, used for the actual
        blendshape generation.

    They are grouped into `emotion key frames` which are a timestamp as well as emotion parameters.
    """
    emotion_aggregate: EmotionAggregate = EmotionAggregate()
    # Metadata is an Any type, try to unpack it into an EmotionAggregate object
    if (animation_data.metadata["emotion_aggregate"] and
        animation_data.metadata["emotion_aggregate"].Unpack(emotion_aggregate)):
        for emotion_with_timecode in emotion_aggregate.a2e_output:
            emotion_key_frames["a2e_output"].append({
                "time_code": emotion_with_timecode.time_code,
                "emotion_values": dict(emotion_with_timecode.emotion),
            })
        for emotion_with_timecode in emotion_aggregate.input_emotions:
            emotion_key_frames["input"].append({
                "time_code": emotion_with_timecode.time_code,
                "emotion_values": dict(emotion_with_timecode.emotion),
            })
        for emotion_with_timecode in emotion_aggregate.a2f_smoothed_output:
            emotion_key_frames["a2f_smoothed_output"].append({
                "time_code": emotion_with_timecode.time_code,
                "emotion_values": dict(emotion_with_timecode.emotion),
            })


def compute_output_directory_name():
  index = 1
  dir_name = f"{OUTPUT_BASE_FOLDER}_{index:06}"
  while os.path.exists(dir_name):
      index += 1
      dir_name = f"{OUTPUT_BASE_FOLDER}_{index:06}"
  return dir_name


async def read_from_stream(stream, should_print_to_files, print_fps):
    # List of blendshapes names recovered from the model data in the AnimationDataStreamHeader
    bs_names = []
    # List of animation key frames, meaning a time code and the values of the blendshapes
    animation_key_frames = []
    # Audio buffer that contains the result
    audio_buffer = b''
    # Audio header to store metadata for audio saving
    audio_header: AudioHeader = None
    # Emotions 'key frames' data from input, a2e output and final a2f-3d smoothed output.
    emotion_key_frames = {
        "input": [],
        "a2f_smoothed_output": []
    }
    # Prep variables to compute latency and fps of the request.
    packet_count = 0
    start = time.time()
    end = time.time()
    # Reads the content of the stream using the read() method of the StreamStreamCall object.
    while True:
        # Read an incoming packet.
        message = await stream.read()
        # If this is the first sent packet, mark the start of audio processing to compute latency and FPS.
        if packet_count == 0:
            load_test_time = time.time()
            if print_fps:
                print("Performance: First packet received", load_test_time)
            start = time.time()
            packet_count += 1
        if message == grpc.aio.EOF:
            # Update the end time for one audio processing.
            end = time.time()
            if print_fps:
                # Print FPS count by dividing the number of packets to the total duration of the requests.
                diff_packets = end-start
                diff_packets = diff_packets if diff_packets != 0 else 1
                print("FPS", packet_count / diff_packets)
            # Print the resulting animation and emotion data to files if specified.
            if should_print_to_files:
                dir_name = compute_output_directory_name() 
                os.makedirs(dir_name, exist_ok=True)
                # End of File signals that the stream has been read completely.
                # Not to be confused with the Status Message that contains the response of the RPC call.
                save_audio_data_to_file(dir_name, audio_header, audio_buffer)

                # Normalize the dictionary data to output in JSON.
                df_animation = pd.json_normalize(animation_key_frames)
                df_smoothed_output = pd.json_normalize(emotion_key_frames["a2f_smoothed_output"])
                df_input = pd.json_normalize(emotion_key_frames["input"])

                # Save data to csv.
                print(f"Saving data into {dir_name} folder...")
                df_animation.to_csv(f"{dir_name}/animation_frames.csv")
                df_smoothed_output.to_csv(f"{dir_name}/a2f_3d_smoothed_emotion_output.csv")
                df_input.to_csv(f"{dir_name}/a2f_3d_input_emotions.csv")
            return

        if message.HasField("animation_data_stream_header"):
            # Message is a header
            print("Receiving data from server...")
            animation_data_stream_header: AnimationDataStreamHeader = message.animation_data_stream_header
            # Save blendshapes names for later use
            bs_names = animation_data_stream_header.skel_animation_header.blend_shapes
            # Save audio header for later use
            audio_header = animation_data_stream_header.audio_header
        elif message.HasField("animation_data"):
            print(".", end="", flush=True)
            # Message is animation data.
            animation_data: AnimationData = message.animation_data
            parse_emotion_data(animation_data, emotion_key_frames)
            blendshape_list = animation_data.skel_animation.blend_shape_weights
            for blendshapes in blendshape_list:
                # We assign each blendshape name to its corresponding weight.
                bs_values_dict = dict(zip(bs_names, blendshapes.values))
                time_code = blendshapes.time_code
                # Append an object to the list of animation key frames
                animation_key_frames.append({
                    "timeCode": time_code,
                    "blendShapes": bs_values_dict
                })
            # Append audio data to the final audio buffer.
            audio_buffer += animation_data.audio.audio_buffer
            packet_count += 1
        elif message.HasField("status"):
            # Message is status
            print()
            status = message.status
            if status.code == 0:
                print("Status code: SUCCESS")
            if status.code == 1:
                print("Status code: INFO")
            if status.code == 2:
                print("Status code: WARNING")
            if status.code == 3:
                print("Status code: ERROR")
                print("If you have an error related to audio input limits, you can adjust the microservice advanced configuration file to fix this issue.")
            print(f"{status.message}")


async def write_to_stream(stream, config_path, audio_file_path, print_fps):
    # Read the content of the audio file, extracting sample rate and data.
    samplerate, data = scipy.io.wavfile.read(audio_file_path)

    bit_depth = data.dtype.itemsize * BITS_PER_BYTE
    channels = data.shape[1] if len(data.shape) > 1 else 1

    print("Input audio header info:")
    print(f"Sample rate: {samplerate} Hz")
    print(f"Bit depth: {bit_depth} bits")
    print(f"Channels: {channels}")

    if bit_depth != REQUIRED_BITS_PER_SAMPLE:
        raise ValueError("Unsupported audio type. Only 16-bit audio is supported.")

    if channels != REQUIRED_CHANNEL_COUNT:
        raise ValueError("Unsupported audio type. Only mono audio is supported.")

    config = None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Each message in the Stream should be an AudioStream message.
    # An AudioStream message can be composed of the following messages:
    # - AudioStreamHeader: must be the first message to be send,
    #       contains metadata about the audio file.
    # - AudioWithEmotion: audio bytes as well as emotions to apply.
    # - EndOfAudio: final message to signal audio sending termination.
    audio_stream_header = AudioStream(
        audio_stream_header=AudioStreamHeader(
            audio_header=AudioHeader(
                samples_per_second=samplerate,
                bits_per_sample=bit_depth,
                channel_count=channels,
                audio_format=REQUIRED_AUDIO_FORMAT,
            ),
            emotion_post_processing_params=EmotionPostProcessingParameters(
                **config["post_processing_parameters"]
            ),
            face_params=FaceParameters(float_params=config["face_parameters"]),
            blendshape_params=BlendShapeParameters(
                bs_weight_multipliers=config["blendshape_parameters"]["multipliers"],
                bs_weight_offsets=config["blendshape_parameters"]["offsets"],
                enable_clamping_bs_weight=config["blendshape_parameters"]["enable_clamping_bs_weight"]
            ),
            emotion_params=EmotionParameters(
                live_transition_time=config["live_transition_time"],
                beginning_emotion=config["beginning_emotion"],
            )
        )
    )

    # Compute and print when first packet was sent.
    load_test_time = time.time()
    if print_fps:
        print("Performance: First packet sent", load_test_time)

    # Sending the AudioStreamHeader message encapsulated into an AudioStream object.
    await stream.write(audio_stream_header)

    for i in range(len(data) // samplerate + 1):
        # Cutting the audio into arbitrary chunks, here we use sample rate to send exactly one
        # second of audio per packet but the size does not matter.
        chunk = data[i * samplerate : i * samplerate + samplerate]
        # Send audio buffer to A2F-3D.
        # Packet 0 contains the emotion with timecode list
        # Here we send all the emotion with timecode alongside the first audio buffer
        # as they are available. In a streaming scenario if you don't have access
        # to some emotions right away you can send them in the next audio buffers.
        if i == 0:
            list_emotion_tc = [
                EmotionWithTimeCode(emotion={**v["emotions"]}, time_code=v["time_code"])
                for v in config["emotion_with_timecode_list"].values()
            ]
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk.astype(np.int16).tobytes(),
                        emotions=list_emotion_tc
                    )
                )
            )
        else:
            # Send only the audio buffer
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk.astype(np.int16).tobytes()
                    )
                )
            )
    # Sending the EndOfAudio message to signal end of sending.
    # This is necessary to obtain the status code at the end of the generation of
    # blendshapes. This status code tells you about the end of animation data stream.
    await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))


async def check_health(channel: grpc.aio.Channel):
    """
    Check the health of the Audio2Face-3D service
    Args:
        channel (grpc.aio.Channel): A async gRPC channel to communicate with the Audio2Face-3D service

    Returns:
        bool: True if the service is healthy, False otherwise

    """
    stub = HealthStub(channel)

    try:
        response = await stub.Check(HealthCheckRequest())
        return response.status == HealthCheckResponse.SERVING
    except grpc.RpcError as err:
        print(f"{err.details()}")
        return False



def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sample python3 application to send audio and receive animation data and emotion "
            "data through the A2F-3D pipeline."
        ),
        epilog="NVIDIA CORPORATION.  All rights reserved.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Health check subparser
    health_check_parser = subparsers.add_parser("health_check", help="Check GRPC service health")
    health_check_parser.add_argument("--url", type=str, required=True, help="GRPC service URL")
    health_check_parser.add_argument("--secure-mode", type=str, choices=["disabled", "tls", "mtls"], default="disabled", help="Security mode")
    health_check_parser.add_argument("--root-cert-path", type=str, help="Path to the root certificate")
    health_check_parser.add_argument("--client-cert-path", type=str, help="Path to the client certificate (required for mTLS)")
    health_check_parser.add_argument("--client-key-path", type=str, help="Path to the client key (required for mTLS)")

    # Run inference subparser
    inference_parser = subparsers.add_parser("run_inference", help="Send GRPC request and run inference for an audio file")
    inference_parser.add_argument("file", help="PCM-16 bits mono Audio file to send to the pipeline")
    inference_parser.add_argument("config", help="Configuration file")
    inference_parser.add_argument("-u", "--url", help="URL of the A2F-3D NIM", required=True)
    inference_parser.add_argument("--print-fps",action="store_true", required=False, default=False, help=argparse.SUPPRESS)
    inference_parser.add_argument("--secure-mode", type=str, choices=["disabled", "tls", "mtls"], default="disabled", help="Security mode")
    inference_parser.add_argument("--root-cert-path", type=str, help="Path to the root certificate")
    inference_parser.add_argument("--client-cert-path", type=str, help="Path to the client certificate (required for mTLS)")
    inference_parser.add_argument("--client-key-path", type=str, help="Path to the client key (required for mTLS)")
    inference_parser.add_argument(
        "--skip-print-to-files",
        help=(
            "Optional. If set, the script will NOT print the resulting animation and emotion data to csv files."
        ),
        action="store_true", required=False, default=False)
    return parser

def create_grpc_channel(server_address, secure_mode=None, root_cert_path=None, client_cert_path=None, client_key_path=None):
    if secure_mode == "mtls":
        with open(root_cert_path, 'rb') as f:
            root_cert = f.read()
        with open(client_cert_path, 'rb') as f:
            client_cert = f.read()
        with open(client_key_path, 'rb') as f:
            client_key = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert,
                                                   private_key=client_key,
                                                   certificate_chain=client_cert)
        return grpc.aio.secure_channel(server_address, credentials)
    elif secure_mode == "tls":
        with open(root_cert_path, 'rb') as f:
            root_cert = f.read()
        credentials = grpc.ssl_channel_credentials(root_certificates=root_cert)
        return grpc.aio.secure_channel(server_address, credentials)
    else:
        return grpc.aio.insecure_channel(server_address)

async def main():
    args = create_parser().parse_args()
    # Suppress WavFileWarning warnings
    warnings.filterwarnings("ignore", category=scipy.io.wavfile.WavFileWarning)

    secure_mode = args.secure_mode.lower()
    if secure_mode in ["tls", "mtls"]:
        if not args.root_cert_path:
            print("Error: root_cert_path is required for secure_mode 'tls' or 'mtls'")
            return
        if secure_mode == "mtls" and (not args.client_cert_path or not args.client_key_path):
            print("Error: client_cert_path and client_key_path are required for secure_mode 'mtls'")
            return

    channel = create_grpc_channel(args.url, secure_mode, args.root_cert_path, args.client_cert_path, args.client_key_path)

    if args.command == "health_check":
        # Checks the health of the service at the specified URL.
        # Prints "ONLINE" if the service is available, "OFFLINE" otherwise.
        is_online = await check_health(channel)
        print(f'Service {args.url} is {"ONLINE" if is_online else "OFFLINE"}')

    elif args.command == "run_inference":
        # Creating a channel to connect to the A2F-3D NIM.
        # If behind HTTPS proxy or using HTTPS, please refer to
        # https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.secure_channel
        async with channel as c:
            # Creating a stub for the service. This allows us to use the remote channel to communicate
            # via RPC to the controller.
            stub = A2FControllerServiceStub(c)

            # ProcessAudioStream is a bidirectionnal stream, or StreamStreamCall object
            # It exposes a read and write interface as shown here:
            # https://grpc.github.io/grpc/python/grpc_asyncio.html#grpc.aio.StreamStreamCall
            stream = stub.ProcessAudioStream()
            # We create an asyncio task for reading the content of the string, into a async function
            # called read_from_stream.

            try:
                read = asyncio.create_task(read_from_stream(stream, not args.skip_print_to_files, args.print_fps))
                # We create another asyncio task for writing into the stream. This allows us to run them
                # both in parrallel instead of sequentially.
                write = asyncio.create_task(write_to_stream(stream, args.config, args.file, args.print_fps))
                # Await task termination.
                await write
                # Await task termination.
                await read
            except ValueError as e:
                print(f"Error: {e}")
                exit(1)
            except grpc.aio.AioRpcError as e:
                print(f"{e.details()}")
                exit(1)
            except Exception as e:
                print(f"Unexpected error: {e}")
                exit(1)



if __name__ == "__main__":
    asyncio.run(main())
