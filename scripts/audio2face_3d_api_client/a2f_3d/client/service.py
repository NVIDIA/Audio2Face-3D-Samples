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

import argparse, asyncio, os, grpc, scipy, numpy, yaml, pandas, warnings
from sys import stderr
from datetime import datetime
from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
from nvidia_ace.a2f.v1_pb2 import AudioWithEmotion, EmotionPostProcessingParameters, FaceParameters, BlendShapeParameters
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode
from nvidia_ace.emotion_aggregate.v1_pb2 import EmotionAggregate

# Bit depth of the audio file, only 16 bit PCM audio is currently supported.
BITS_PER_SAMPLE = 16
# Channel count, only mono audio is currently supported.
CHANNEL_COUNT = 1
# Audio format, only PCM is supported.
AUDIO_FORMAT = AudioHeader.AUDIO_FORMAT_PCM

def get_audio_bit_format(audio_header: AudioHeader):
    """
    Reads the audio_header parameters and returns the write type to interpret
    the audio data sent back by the server.
    """
    if audio_header.audio_format == AudioHeader.AUDIO_FORMAT_PCM:
        # We only support 16 bits PCM.
        if audio_header.bits_per_sample == 16:
            return numpy.int16
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

    audio_data_to_save = numpy.frombuffer(audio_buffer, dtype=dtype) 
    # Write the audio data output as a wav file.
    scipy.io.wavfile.write(f"{outdir}/out.wav", audio_header.samples_per_second, audio_data_to_save)


def parse_emotion_data(animation_data, emotion_key_frames):
    """
    Fills the emotion key frames dictionnary using the data found in the emotion_aggregate metadata.

    Each emotion aggregate contains the following values:
    - input_emotions: Emotions that are manually inputed by the user.
    - a2e_output: The output of the emotion inference on the audio out of Audio2Emotion.
    - a2f_smoothed_output: The smoothed and post-processed emotions output, used for the actual blendshape generation.

    They are grouped into `emotion key frames` which are a timestamp as well as emotion parameters.
    """
    emotion_aggregate: EmotionAggregate = EmotionAggregate() 
    # Metadata is an Any type, try to unpack it into an EmotionAggregate object
    if animation_data.metadata["emotion_aggregate"] and animation_data.metadata["emotion_aggregate"].Unpack(emotion_aggregate):
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

async def read_from_stream(stream):
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
    # Reads the content of the stream using the read() method of the StreamStreamCall object.
    while True:
        # Read an incoming packet.
        message = await stream.read()
        if message == grpc.aio.EOF:
            # Create directory with current date and time
            timestamp = datetime.now()
            dir_name = timestamp.strftime("%Y%m%d_%H%M%S_%f")
            os.makedirs(dir_name, exist_ok=False)
            # End of File signals that the stream has been read completely.
            # Not the be confused with the Status Message that contains the response of the RPC call.
            save_audio_data_to_file(dir_name, audio_header, audio_buffer)

            # Normalize the dictionnary data to output in JSON.
            df_animation = pandas.json_normalize(animation_key_frames)
            df_smoothed_output = pandas.json_normalize(emotion_key_frames["a2f_smoothed_output"])
            df_input = pandas.json_normalize(emotion_key_frames["input"])

            # Save data to csv.
            df_animation.to_csv(f"{dir_name}/animation_frames.csv")
            df_smoothed_output.to_csv(f"{dir_name}/a2f_smoothed_emotion_output.csv")
            df_input.to_csv(f"{dir_name}/a2f_input_emotions.csv")
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
                # We assign each blendshape name to it's corresponding weight.
                bs_values_dict = dict(zip(bs_names, blendshapes.values))
                time_code = blendshapes.time_code
                # Append an object to the list of animation key frames
                animation_key_frames.append({
                    "timeCode": time_code,
                    "blendShapes": bs_values_dict
                })
            # Append audio data to the final audio buffer.
            audio_buffer += animation_data.audio.audio_buffer
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

            print(f"Received status message with value: '{status.message}'")
           

async def write_to_stream(stream, config_path, audio_file_path):
    # Read the content of the audio file, extracting sample rate and data.
    samplerate, data = scipy.io.wavfile.read(audio_file_path)
    config = None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Each message in the Stream should be an AudioStream message.
    # An AudioStream message can be composed of the following messages:
    # - AudioStreamHeader: must be the first message to be send, contains metadata about the audio file.
    # - AudioWithEmotion: audio bytes as well as emotions to apply.
    # - EndOfAudio: final message to signal audio sending termination.
    audio_stream_header = AudioStream(
        audio_stream_header=AudioStreamHeader(
            audio_header=AudioHeader(
                samples_per_second=samplerate,
                bits_per_sample=BITS_PER_SAMPLE,
                channel_count=CHANNEL_COUNT,
                audio_format=AUDIO_FORMAT
            ),
            emotion_post_processing_params=EmotionPostProcessingParameters(
                **config["post_processing_parameters"]
            ),
            face_params=FaceParameters(float_params=config["face_parameters"]),
            blendshape_params=BlendShapeParameters(
                bs_weight_multipliers=config["blendshape_parameters"]["multipliers"],
                bs_weight_offsets=config["blendshape_parameters"]["offsets"]
            )
        )
    )

    # Sending the AudioStreamHeader message encapsulated into an AudioStream object.
    await stream.write(audio_stream_header)

    for i in range(len(data) // samplerate + 1):
        # Cutting the audio into arbitrary chunks, here we use sample rate to send exactly one second
        # of audio per packet but the size does not matter.
        chunk = data[i * samplerate: i * samplerate + samplerate]
        # Send audio buffer to A2F-3D.
        # Packet 0 contains the emotion with timecode list
        # Here we send all the emotion with timecode alongside the first audio buffer
        # as they are available. In a streaming scenario if you don't have access
        # to some emotions right away you can send them in the next audio buffers.
        if i == 0:
            list_emotion_tc = [
                EmotionWithTimeCode(
                    emotion={
                        **v["emotions"]
                    },
                    time_code=v["time_code"]
                ) for v in config["emotion_with_timecode_list"].values()
            ]
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk.astype(numpy.int16).tobytes(),
                        emotions=list_emotion_tc
                    )
                )
            )
        else:
            # Send only the audio buffer
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk.astype(numpy.int16).tobytes()
                    )
                )
            )
    # Sending the EndOfAudio message to signal end of sending.
    # This is necessary to obtain the status code at the end of the generation of
    # blendshapes. This status code tells you about the end of animation data stream.
    await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))
    