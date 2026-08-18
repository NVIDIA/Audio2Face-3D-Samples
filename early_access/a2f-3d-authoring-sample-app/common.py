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
"""Common methods for communicating with Audio2Face-3D Authoring service."""

import argparse
import os
import threading
import time
import random
import sys
from dataclasses import dataclass

import grpc
import matplotlib.pyplot as plt
import numpy as np
import scipy

import auth

from nvidia_ace.health_pb2_grpc import HealthStub
from nvidia_ace.health_pb2 import HealthCheckRequest, HealthCheckResponse
from nvidia_ace.a2f.v1_pb2 import (
    BlendShapeParameters,
    EmotionPostProcessingParameters,
    FaceParameters,
)
from nvidia_ace.a2f_authoring.v1_pb2 import (
    AudioClip,
    AudioClipHandle,
    BlendShapeData,
    FacePoseRequest,
)
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.services.a2f_authoring.v1_pb2_grpc import A2FAuthoringServiceStub

PERF_DATA_FILE = "output_perf.txt"
OUT_IMG = "output_performance.png"
OUTPUT_FILE = "output_blendshape.csv"

DEFAULT_FACE_PARAMS = {
    "lowerFaceSmoothing": 0.0,
    "upperFaceSmoothing": 0.0,
    "lowerFaceStrength": 1.30,
    "upperFaceStrength": 1.0,
    "faceMaskLevel": 0.6,
    "faceMaskSoftness": 0.008,
    "skinStrength": 1.0,
    "blinkStrength": 1.0,
    "eyelidOpenOffset": 0.6,
    "lipOpenOffset": -0.03,
    "blinkOffset": 0.53,
    "tongueStrength": 0.53,
    "tongueHeightOffset": 0.53,
    "tongueDepthOffset": 0.53,
}

EMOTION_KEYS = [
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "outofbreath",
    "pain",
    "sadness",
]

TIME_1_FRAME = 1 / 30


@dataclass
class GrpcChannelParams:
    """
    Represents parameters to connect with secure or insecure gRPC channel.

    is_local (bool): Whether or not the script should connect to local deployment.
        If True, use the url to connect to the service. If False, use the metadata_args.
    url (str, optional): URL of the local Authoring service deployment.
    metadata_args(list, optional): Metadata for connecting to the NVCF Authoring deployment.
    """

    is_local: bool = False
    url: str = ""
    metadata_args: list = None


class LatencyResult:
    """
    Represents the latency result of a test, containing various metrics related to
    upload time, total request time, and latencies for requests.

    Attributes:
        request_number (int): Total number of requests made during the test.
        number_concurrent_users (int): Number of users executing requests concurrently.
        time_upload (float): Time taken to upload data, in milliseconds.
        time_request_all (float): Total time taken to complete all requests, in milliseconds.
        list_latencies_ms (list of float): List of latency values for individual requests,
            in milliseconds.
        blendshape_names (list of strings): List of blendshape names computed by the microservice.

    """

    def __init__(
        self,
        request_number: int,
        number_parallel_request: int,
        time_upload: float,
        time_request_all: float,
        list_latencies_ms: list[float],
        blendshape_names: list[str],
    ):
        """
        Initializes the LatencyResult instance with the given parameters.

        Args:
            request_number (int): Total number of requests made.
            number_parallel_request (int): Number of concurrent users making requests.
            time_upload (float): Time taken to upload data, in milliseconds.
            time_request_all (float): Time taken to complete all requests, in milliseconds.
            list_latencies_ms (list of float): List of latency values in milliseconds.
            blendshape_names (list of strings): List of blendshape names computed by the microservice.

        """
        self.request_number = request_number
        self.number_concurrent_users = number_parallel_request
        self.time_upload = time_upload
        self.time_request_all = time_request_all
        self.list_latencies_ms = list_latencies_ms
        self.blendshape_names = blendshape_names

    def __str__(self):
        """
        Provides a formatted string representation of the latency performance result, including:
            - Upload time.
            - Total request processing time.
            - Minimum, median, mean, and various percentile latencies.

        If the list of latencies is empty, returns "NO latencies!!".

        The output includes the following metrics:
            - Upload Time: Time taken to upload data, in milliseconds.
            - Request Processing Time:
                Total time taken to process all requests, in milliseconds.
            - Minimum Latency: Shortest latency recorded, in milliseconds.
            - 1st Percentile Latency (P1):
                Latency value below which 1% of data points fall, in milliseconds.
            - 5th Percentile Latency (P5):
                Latency value below which 5% of data points fall, in milliseconds.
            - 10th Percentile Latency (P10):
                Latency value below which 10% of data points fall, in milliseconds.
            - Median Latency (P50): Middle latency value (50th percentile), in milliseconds.
            - Mean Latency: Average latency value, in milliseconds.
            - 90th Percentile Latency (P90):
                Latency value below which 90% of data points fall, in milliseconds.
            - 95th Percentile Latency (P95):
                Latency value below which 95% of data points fall, in milliseconds.
            - 99th Percentile Latency (P99):
                Latency value below which 99% of data points fall, in milliseconds.

        Returns:
            str: A string representation of the latency result.

        """
        if self.list_latencies_ms is None or len(self.list_latencies_ms) == 0:
            return "NO latencies!!"

        data = np.array(self.list_latencies_ms)

        min_latency = np.min(data)
        p1_latency = np.percentile(data, 1)
        p5_latency = np.percentile(data, 5)
        p10_latency = np.percentile(data, 10)
        median_latency = np.median(data)
        mean_latency = np.mean(data)
        p90_latency = np.percentile(data, 90)
        p95_latency = np.percentile(data, 95)
        p99_latency = np.percentile(data, 99)

        res = f"Latency Metrics for {self.request_number} Requests "
        res += f"+ {self.number_concurrent_users} concurrent users:\n"
        res += f"Upload Time: {self.time_upload:.2f} ms\n"
        res += f"Request Processing Time: {self.time_request_all:.2f} ms\n"
        res += f"Minimum Latency: {min_latency:.2f} ms\n"
        res += f"1st Percentile Latency (P1): {p1_latency:.2f} ms\n"
        res += f"5th Percentile Latency (P5): {p5_latency:.2f} ms\n"
        res += f"10th Percentile Latency (P10): {p10_latency:.2f} ms\n"
        res += f"Median Latency (P50): {median_latency:.2f} ms\n"
        res += f"Mean Latency: {mean_latency:.2f} ms\n"
        res += f"90th Percentile Latency (P90): {p90_latency:.2f} ms\n"
        res += f"95th Percentile Latency (P95): {p95_latency:.2f} ms\n"
        res += f"99th Percentile Latency (P99): {p99_latency:.2f} ms\n"

        return res

    def plot_percentiles(self, output_file: str) -> None:
        """
        Generates and saves a histogram plot of latencies with percentile markers.

        The histogram represents the distribution of latencies, and vertical lines are drawn
        at specific percentiles to indicate their values.

        Args:
            output_file (str): The file path to save the generated plot image.

        Returns:
            None

        """
        # Percentiles to display on the plot
        percentiles_keys = [1, 5, 50, 95, 99]

        # Initialize plot figure size
        plt.figure(figsize=(20, 10))

        # Calculate percentiles
        percentiles = np.percentile(self.list_latencies_ms, percentiles_keys)

        # Plot histogram of latencies
        plt.hist(self.list_latencies_ms, bins=30, alpha=0.7, edgecolor="black")

        # Plot vertical lines at each percentile with different colors
        # Generate a range of colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles_keys)))

        for percentile, value, color in zip(percentiles_keys, percentiles, colors):
            plt.axvline(
                x=value,
                linestyle="--",
                color=color,
                label=f"{percentile}th Percentile: {value:.2f}ms",
            )

        # Set plot title, labels, and legend
        plt.title(
            f"Histogram of latencies with Percentiles for {self.request_number} requests"
            f" and {self.number_concurrent_users} concurrent connections"
        )
        plt.xlabel("Time in ms")
        plt.ylabel("Frequency")
        plt.legend()

        # Save the plot to a file, replacing any existing file
        if os.path.exists(output_file):
            os.remove(output_file)
        plt.savefig(output_file)
        plt.close()


def upload_audio_clip_and_get_hash(stub: A2FAuthoringServiceStub, filepath: str) -> tuple[str, list[str]] | None:
    """
    Upload an audio clip from a file to a gRPC service and return its hash.

    Args:
        stub (A2FAuthoringSerivceStub): The stub used to call gRPC.
        filepath (str): The path to the audio clip file.

    Returns:
        str: The hash of the uploaded audio clip.
        list[str]: The blenshape names that the authoring service will use when processing the clip.

    Raises:
        RuntimeError: If there is an issue uploading the audio clip.

    """
    # Read the audio clip file using SciPy.
    samplerate, data = scipy.io.wavfile.read(filepath)

    if data.ndim != 1:
        raise ValueError(f"Audio file must be mono; got {data.ndim} channels")
    if data.dtype != np.int16:
        raise ValueError(f"Audio file must be 16-bit PCM; got dtype {data.dtype}")

    # Create a request to upload the audio clip.
    request = AudioClip(
        audio_header=AudioHeader(
            samples_per_second=samplerate,
            bits_per_sample=16,
            channel_count=1,
            audio_format=AudioHeader.AUDIO_FORMAT_PCM,
        ),
        content=data.tobytes(),
    )
    try:
        # Upload the audio clip, retrieve its hash, and get the list of blendshape names that the authoring service will
        # use when processing the clip.
        handle: AudioClipHandle = stub.UploadAudioClip(request)
        audio_clip_hash = handle.audio_clip_id
        blendshape_names = handle.blendshape_names
        print(f"audio_clip_hash={audio_clip_hash}")
        return (audio_clip_hash, blendshape_names)
    except grpc.RpcError as rpc_error:
        print(f"Receivedd RPC error: {rpc_error}")
        raise RuntimeError("Issue with uploading audio clip") from rpc_error


def make_face_pose_request(audio_hash: str, time_code: float, bs_names: list[str]) -> FacePoseRequest:
    """
    Creates a FacePoseRequest proto from a given hash and index frame.

    Args:
      audio_hash (str): The hash of the audio clip after upload.
      time_code (float): The time of the frame for which to create the FacePoseRequest.
      bs_names (list): A list of blendshape names used when passing parameters for each blendshape.

    Returns:
      FacePoseRequest: A proto used to call the A2F-3D Authoring microservice.

    """
    epp = EmotionPostProcessingParameters(
        emotion_contrast=0.5,
        live_blend_coef=0.5,
        enable_preferred_emotion=True,
        preferred_emotion_strength=0.5,
        emotion_strength=1.0,
        max_emotions=6,
    )
    fpr = FacePoseRequest(
        audio_hash=audio_hash,
        preferred_emotions={elm: 0 for elm in EMOTION_KEYS},
        time_stamp=time_code,
        face_params=FaceParameters(float_params=DEFAULT_FACE_PARAMS),
        emotion_pp_params=epp,
        blendshape_params=BlendShapeParameters(
            bs_weight_multipliers={elm: 1.0 for elm in bs_names},
            bs_weight_offsets={elm: 0.0 for elm in bs_names},
            enable_clamping_bs_weight=False,
        ),
    )
    return fpr


def get_avatar_face_pose(stub: A2FAuthoringServiceStub, request: FacePoseRequest) -> BlendShapeData | None:
    """
    Retrieves face pose parameters from the Authoring server using gRPC.

    Args:
        stub (A2FAuthoringSerivceStub): The stub used to call gRPC.
        request (FacePoseRequest): The input FacePoseRequest proto message.

    Returns:
        BlendShapeData: The proto response from the service.

    Raises:
        RuntimeError: If there is an error with calling get avatar face pose over gRPC.

    """
    print(f"\r{request.time_stamp:0.03} seconds processed  ", end="")
    response = None
    try:
        response = stub.GetAvatarFacePose(request)
    except grpc.RpcError as rpc_error:
        print(f"Received RPC error: {rpc_error}")
        raise RuntimeError("Could not get avatar face pose") from rpc_error
    return response


def check_health(channel: grpc.Channel):
    """
    Check the health of the Authoring service
    Args:
        channel (grpc.Channel): A gRPC channel to communicate with the Authoring service.

    Returns:
        bool: True if the service is healthy, False otherwise

    """
    stub = HealthStub(channel)

    try:
        response = stub.Check(HealthCheckRequest())
        return response.status == HealthCheckResponse.SERVING
    except grpc.RpcError as err:
        print(f"Error checking health: {err}")
        return False


def convert_seconds_to_milliseconds(seconds):
    """
    Converts a time duration from seconds to milliseconds.

    Args:
        seconds (float): The time duration in seconds.

    Returns:
        float: The time duration in milliseconds.

    """
    return seconds * 1000


def prepare_requests(audio_clip, num_requests):
    """
    Prepares a list of requests for data exchange by reading an audio clip,
    uploading it to get a hash, and generating timecodes for the requests.

    Args:
        audio_clip (str): The file path of the audio clip to be processed.
        num_requests (int): The total number of requests to be made.

    Returns:
        tuple: A tuple containing:
            - requests (list of tuple): A list of tuples where each tuple contains
             an index and a request data object.
            - time_upload_duration (float): The time taken to upload the audio
             and get the hash, in seconds.

    """
    # Read audio file and calculate the number of frames needed
    samplerate, data = scipy.io.wavfile.read(audio_clip)
    num_frames = max(int((len(data) / samplerate) * 30), 1)

    # Generate a list of timecodes for frames
    time_codes = [i * TIME_1_FRAME for i in range(num_frames)]

    # Determine how many times to duplicate the list to reach the desired request number
    duplication_factor = (num_requests // num_frames) + 1
    extended_time_codes = time_codes * duplication_factor

    # Shuffle the time codes deterministically
    random.seed(42)
    random.shuffle(extended_time_codes)
    shuffled_time_codes = extended_time_codes[:num_requests]


    return shuffled_time_codes


def get_params_with_stub(stub: A2FAuthoringServiceStub, request: FacePoseRequest) -> float:
    """
    Retrieves face pose parameters from the Authoring server using a pre-existing gRPC stub.

    Args:
        stub (A2FAuthoringStub): The pre-existing gRPC stub.
        request (FacePoseRequest): The input FacePoseRequest message.

    Returns:
        float: The time taken to retrieve the face pose parameters,
            or raises an exception if failed.

    """
    print(".", end="")
    sys.stdout.flush()
    try:
        time1 = time.perf_counter()
        stub.GetAvatarFacePose(request)
        time2 = time.perf_counter()
    except grpc.RpcError as rpc_error:
        print(f"Received RPC error: {rpc_error}")
        raise RuntimeError("Could not get avatar face pose") from rpc_error

    return time2 - time1


def split_list(lst, n):
    # Calculate the size of each chunk
    avg = len(lst) / float(n)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last) : int(last + avg)])
        last += avg

    return out


def perform_parallel_requests(
    shuffled_requests: list,
    num_parallel_requests: int,
    num_requests: int,
    channel_params: GrpcChannelParams,
    audio_clip: str,
) -> LatencyResult:
    """
    Executes a list of prepared requests in parallel and records latency metrics.

    Args:
        shuffled_requests (list of tuple): A list of tuples where each tuple contains an index
         and a request data object.
        num_parallel_requests (int): The number of parallel requests to be executed concurrently.
        num_requests (int): The total number of requests to be made.
        upload_time (float): The time taken to upload the audio and get the hash, in milliseconds.
            This one is only used to create the output object
        channel_params (GrpcChannelParams): Params to connect to local or NVCF deployment

    Returns:
        LatencyResult: An object containing the latency metrics of the operation.

    """
    url = channel_params.url if channel_params.is_local else auth.GRPC_URI
    time_code_lists = split_list(shuffled_requests, num_parallel_requests)

    list_test_clients = [
        LatencyTesterClient(channel_params.is_local, url, channel_params.metadata_args, audio_clip, time_code_lists[i])
        for i in range(num_parallel_requests)
    ]
    run_latency_client = lambda i: list_test_clients[i].run()

    # Measure the time taken for executing all requests in parallel
    time_start_requests = time.perf_counter()
    threads = []
    for i in range(num_parallel_requests):
        thread = threading.Thread(target=run_latency_client, args=(i,))
        threads.append(thread)
        thread.start()

    # Optionally, wait for all threads to complete
    for thread in threads:
        thread.join()
    time_end_requests = time.perf_counter()

    failed = [client for client in list_test_clients if client.exception is not None]
    if failed:
        raise RuntimeError("One or more latency tester threads failed") from failed[0].exception

    # Collect the results and handle exceptions
    list_latencies = [elm.get_latencies() for elm in list_test_clients]
    list_latencies = np.concatenate([np.array(latency) for latency in list_latencies])

    # Convert request duration and latencies to milliseconds
    request_duration = convert_seconds_to_milliseconds(time_end_requests - time_start_requests)
    list_latencies_ms = convert_seconds_to_milliseconds(np.array(list_latencies))
    upload_time_ms = convert_seconds_to_milliseconds(
        np.array([elm.time_upload for elm in list_test_clients]).mean()
    )

    # Get the blendshape names from one of the tester clients. This is ok since all tester clients connect to the same
    # authoring microservice.
    blendshape_names = list_test_clients[0].blenshape_names

    # Return latency metrics
    return LatencyResult(
        num_requests,
        num_parallel_requests,
        upload_time_ms,
        request_duration,
        list_latencies_ms,
        blendshape_names,
    )


class LatencyTesterClient:
    def __init__(self, is_local, url, metadata_args, audio_clip, time_code_list):
        self.is_local = is_local
        self.url = url
        self.metadata_args = metadata_args
        self.audio_clip = audio_clip
        self.time_code_list = time_code_list
        self.latencies_list = []
        self.time_upload = None
        self.channel = None
        self.requests = None
        self.stub = None
        self.blenshape_names = None
        self.exception = None

    def initialize_channel(self):
        if self.is_local:
            # Open an insecure gRPC channel to connect to local deployment
            self.channel = grpc.insecure_channel(self.url)
        else:
            # Open a secure gRPC channel to connect to NVCF deployment.
            self.channel = auth.create_channel(uri=auth.GRPC_URI, use_ssl=True, metadata=self.metadata_args)
        self.stub = A2FAuthoringServiceStub(self.channel)

    def upload_audio(self):
        # Upload audio and measure the time taken to get the hash
        time_start_hash = time.perf_counter()

        (hash_value, bs_names) = upload_audio_clip_and_get_hash(self.stub, self.audio_clip)
        time_end_hash = time.perf_counter()
        self.time_upload = time_end_hash - time_start_hash
        if self.blenshape_names is None:
            self.blenshape_names = bs_names

        # Create a list of requests with the hash and corresponding timecodes
        self.requests = [
            (i, make_face_pose_request(hash_value, timecode, bs_names))
            for i, timecode in enumerate(self.time_code_list)
        ]

    def make_requests(self):
        for i, request in self.requests:
            latency = get_params_with_stub(self.stub, request)
            self.latencies_list.append(latency)

    def run(self):
        try:
            self.initialize_channel()
            self.upload_audio()
            self.make_requests()
        except Exception as exc:
            self.exception = exc

    def get_latencies(self):
        return self.latencies_list


def perform_parallel_data_exchange(
    audio_clip: str, num_requests: int, num_parallel_requests: int, channel_params: GrpcChannelParams
) -> LatencyResult:
    """
    Combines the preparation of requests and their parallel execution to perform data exchange.

    Args:
        audio_clip (str): The file path of the audio clip to be processed.
        num_requests (int): The total number of requests to be made.
        num_parallel_requests (int): The number of parallel requests to be executed concurrently.
        channel_params (GrpcChannelParams): Params to connect to local or NVCF deployment

    Returns:
        LatencyResult: An object containing the latency metrics of the operation.

    """
    # Prepare requests

    shuffled_requests = prepare_requests(audio_clip, num_requests)

    # Execute requests in parallel and return latency result
    return perform_parallel_requests(shuffled_requests, num_parallel_requests, num_requests, channel_params, audio_clip)


def positive_int(value) -> int | None:
    """
    Checks if an argparse value is an integer strictly greater than 0.

    Args:
        value: An untyped argparse value to be checked.

    Returns:
        An integer representing the input value if it matches the condition.

    Raises:
        ArgumentTypeError: If the value is not valid.

    """
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a positive nonzero integer.")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a positive nonzero integer.") from ValueError
