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
import json
import os

from a2f_data_classes.a2f_config import A2FConfig
from a2f_data_classes.a2f_controller_config import A2FControllerConfig
from a2f_data_classes.ucs_app_config import UCSConfig
from ruamel.yaml import YAML
# Initialize YAML object
yaml = YAML()
from dataclasses import fields, is_dataclass

CHOICE_CONVERT = ["ucs", "docker_config"]
OUTPUT_BASE_FOLDER = "output"
FOLDER_DEFAULT = "default_configs_v1_2"
STYLE_JAMES_CFG = "james_stylization_config.yaml"
STYLE_CLAIRE_CFG = "claire_stylization_config.yaml"
STYLE_MARK_CFG = "mark_stylization_config.yaml"
DEPLOY_CFG = "deployment_config.yaml"
ADVANCED_CFG = "advanced_config.yaml"

LIST_FACE_PARAM_ORDER = [
   "EyeBlinkLeft",
   "EyeLookDownLeft",
   "EyeLookInLeft",
   "EyeLookOutLeft",
   "EyeLookUpLeft",
   "EyeSquintLeft",
   "EyeWideLeft",
   "EyeBlinkRight",
   "EyeLookDownRight",
   "EyeLookInRight",
   "EyeLookOutRight",
   "EyeLookUpRight",
   "EyeSquintRight",
   "EyeWideRight",
   "JawForward",
   "JawLeft",
   "JawRight",
   "JawOpen",
   "MouthClose",
   "MouthFunnel",
   "MouthPucker",
   "MouthLeft",
   "MouthRight",
   "MouthSmileLeft",
   "MouthSmileRight",
   "MouthFrownLeft",
   "MouthFrownRight",
   "MouthDimpleLeft",
   "MouthDimpleRight",
   "MouthStretchLeft",
   "MouthStretchRight",
   "MouthRollLower",
   "MouthRollUpper",
   "MouthShrugLower",
   "MouthShrugUpper",
   "MouthPressLeft",
   "MouthPressRight",
   "MouthLowerDownLeft",
   "MouthLowerDownRight",
   "MouthUpperUpLeft",
   "MouthUpperUpRight",
   "BrowDownLeft",
   "BrowDownRight",
   "BrowInnerUp",
   "BrowOuterUpLeft",
   "BrowOuterUpRight",
   "CheekPuff",
   "CheekSquintLeft",
   "CheekSquintRight",
   "NoseSneerLeft",
   "NoseSneerRight",
   "TongueOut",
]


def compute_output_directory_name():
    index = 1
    dir_name = f"{OUTPUT_BASE_FOLDER}_{index:06}"
    while os.path.exists(dir_name):
        index += 1
        dir_name = f"{OUTPUT_BASE_FOLDER}_{index:06}"
    return dir_name


def load_yaml_to_dataclass(yaml_data: dict, dataclass_type):
    # Prepare data for the dataclass
    field_values = {}
    for field in fields(dataclass_type):
        yaml_key = field.metadata.get('yaml_key', field.name)
        field_type = field.type
        field_value = yaml_data.get(yaml_key)

        # If the field is itself a dataclass, load it recursively
        if is_dataclass(field_type) and isinstance(field_value, dict):
            field_values[field.name] = load_yaml_to_dataclass(field_value, field_type)
        else:
            field_values[field.name] = field_value
    return dataclass_type(**field_values)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert config from either 'ucs' or 'docker_config'.")
    parser.add_argument(
        "option",
        type=str,
        choices=CHOICE_CONVERT,
        help="Specify 'ucs' or 'docker_config' as the option."
    )
    return parser.parse_args()


def convert_dict_keys_to_underscores(data):
    if isinstance(data, dict):
        return {
            k.replace("-", "_"): convert_dict_keys_to_underscores(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_dict_keys_to_underscores(v) for v in data]
    return data


def load_class(file_path, classtype):
    with open(file_path, "r") as file:
        yaml_data = yaml.load(file)

    yaml_data = convert_dict_keys_to_underscores(yaml_data)
    config = load_yaml_to_dataclass(yaml_data, classtype)
    return config


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        data = yaml.load(file)
    return data


def save_yaml_file(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def get_config_style_from_name_model(name):
    selected_config_style = None
    if "claire" in name:
        selected_config_style = STYLE_CLAIRE_CFG
    elif "james" in name:
        selected_config_style = STYLE_JAMES_CFG
    elif "mark" in name:
        selected_config_style = STYLE_MARK_CFG
    else:
        raise Exception("Expect claire james or mark to be in UCS config as model name!")
    return selected_config_style


def convert_ucs_config(output_folder):
    my_ucs: UCSConfig = load_class("ucs_app_configs/a2f_config.yaml", UCSConfig)

    selected_config_style = get_config_style_from_name_model(my_ucs.a2fModelName)

    map_style = load_yaml_file(os.path.join(FOLDER_DEFAULT, selected_config_style))
    map_deploy = load_yaml_file(os.path.join(FOLDER_DEFAULT, DEPLOY_CFG))
    map_advanced = load_yaml_file(os.path.join(FOLDER_DEFAULT, ADVANCED_CFG))

    face_params_old = json.loads(my_ucs.faceParams) if my_ucs.faceParams != "" else {}

    #
    map_deploy["common"]["stream_number"] = int(my_ucs.streamNumber)
    map_deploy["common"]["add_silence_padding_after_audio"] = my_ucs.addSilencePaddingAfterAudio.lower() == "true"

    map_style["a2e"]["enabled"] = my_ucs.a2eEnabled.lower() == "true"

    list_component_face = [
        "eyelid_offset",
        "face_mask_level",
        "face_mask_softness",
        "input_strength",
        "lip_close_offset",
        "lower_face_smoothing",
        "lower_face_strength",
        "skin_strength",
        "upper_face_smoothing",
        "upper_face_strength"
    ]
    for elm in list_component_face:
        try:
            map_style["a2f"]["face_params"][elm] = face_params_old["face_params"][elm]
        except:
            # missing key; this is allowed; nothing to do
            pass

    list_emotion = [
        "amazement",
        "anger",
        "cheekiness",
        "disgust",
        "fear",
        "grief",
        "joy",
        "outofbreath",
        "pain",
        "sadness"
    ]

    for i, elm in enumerate(list_emotion):
        try:
            map_style["default_beginning_emotions"][elm] = face_params_old["face_params"]["emotion"][i]
        except:
            # missing key; this is allowed; nothing to do
            pass

    map_style["a2e"]["post_processing_params"]["emotion_contrast"] = float(my_ucs.a2eEmotionContrast)
    map_style["a2e"]["post_processing_params"]["live_blend_coef"] = float(my_ucs.a2eLiveBlendCoef)
    map_style["a2e"]["post_processing_params"]["enable_preferred_emotion"] = my_ucs.a2eEnablePreferredEmotion.lower() == "true"
    map_style["a2e"]["post_processing_params"]["preferred_emotion_strength"] = float(my_ucs.a2ePreferredEmotionStrength)
    map_style["a2e"]["post_processing_params"]["emotion_strength"] = float(my_ucs.a2eEmotionStrength)
    map_style["a2e"]["post_processing_params"]["max_emotions"] = int(my_ucs.a2eMaxEmotions)

    map_advanced["a2e"]["inference_interval"] = int(my_ucs.a2eInferenceInterval)
    map_advanced["a2e"]["device_id"] = int(my_ucs.a2fDeviceId)
    map_advanced["pipeline_parameters"]["queue_size_after_streammux"] = int(my_ucs.queueAfterStreammux)
    map_advanced["pipeline_parameters"]["queue_size_after_a2f"] = int(my_ucs.queueAfterA2F)
    map_advanced["pipeline_parameters"]["queue_size_after_a2e"] = int(my_ucs.queueAfterA2E)
    map_advanced["input_sanitization"]["max_len_uuid"] = int(my_ucs.maxLenUUID)
    map_advanced["input_sanitization"]["max_sample_rate"] = int(my_ucs.maxSampleRate)
    map_advanced["input_sanitization"]["min_sample_rate"] = int(my_ucs.minSampleRate)
    map_advanced["input_sanitization"]["low_fps"] = int(my_ucs.lowFps)
    map_advanced["input_sanitization"]["low_fps_max_duration_second"] = int(my_ucs.lowFpsMaxDurationSecond)
    map_advanced["trt_model_generation"]["a2f"]["precision"] = "fp16" if str(my_ucs.useFP16A2F).lower() == "true" else "fp32"
    map_advanced["trt_model_generation"]["a2e"]["precision"] = "fp16" if str(my_ucs.useFP16A2E).lower() == "true" else "fp32"

    save_yaml_file(map_style, os.path.join(output_folder, selected_config_style))
    save_yaml_file(map_deploy, os.path.join(output_folder, DEPLOY_CFG))
    save_yaml_file(map_advanced, os.path.join(output_folder, ADVANCED_CFG))


def convert_docker_config(output_folder):
    my_a2f: A2FConfig = load_class("docker_container_configs/a2f_config.yaml", A2FConfig)
    my_controller: A2FControllerConfig = load_class("docker_container_configs/ac_a2f_config.yaml", A2FControllerConfig)

    selected_config_style = get_config_style_from_name_model(my_a2f.A2F.model_path.split("/")[-1])

    map_style = load_yaml_file(os.path.join(FOLDER_DEFAULT, selected_config_style))
    map_deploy = load_yaml_file(os.path.join(FOLDER_DEFAULT, DEPLOY_CFG))
    map_advanced = load_yaml_file(os.path.join(FOLDER_DEFAULT, ADVANCED_CFG))

    map_deploy["common"]["stream_number"] = my_a2f.common.stream_number
    map_deploy["common"]["add_silence_padding_after_audio"] = my_a2f.common.add_silence_padding_after_audio
    map_advanced["pipeline_parameters"]["queue_size_after_streammux"] = my_a2f.common.queue_size_after_streammux
    map_advanced["pipeline_parameters"]["queue_size_after_a2e"] = my_a2f.common.queue_size_after_a2e
    map_advanced["pipeline_parameters"]["queue_size_after_a2f"] = my_a2f.common.queue_size_after_a2f
    map_advanced["input_sanitization"]["max_len_uuid"] = my_a2f.common.max_len_uuid
    map_advanced["input_sanitization"]["min_sample_rate"] = my_a2f.common.min_sample_rate
    map_advanced["input_sanitization"]["max_sample_rate"] = min(my_a2f.common.max_sample_rate,
                                                                my_controller.common.max_sample_rate)

    a2f_server_unidir_url = f"0.0.0.0:{my_a2f.grpc_input.port}"

    map_deploy["endpoints"]["unidirectional"]["server"]["url"] = a2f_server_unidir_url
    map_advanced["input_sanitization"]["low_fps"] = my_a2f.grpc_input.low_fps
    map_advanced["input_sanitization"]["low_fps_max_duration_second"] = my_a2f.grpc_input.low_fps_max_duration_second

    a2f_client_unidir_url = f"{my_a2f.grpc_output.ip}:{my_a2f.grpc_output.port}"
    map_deploy["endpoints"]["unidirectional"]["client"]["url"] = a2f_client_unidir_url

    map_style["a2e"]["enabled"] = my_a2f.A2E.enabled
    map_advanced["a2e"]["inference_interval"] = my_a2f.A2E.inference_interval
    map_style["a2e"]["post_processing_params"]["emotion_contrast"] = my_a2f.A2E.emotions.emotion_contrast
    map_style["a2e"]["post_processing_params"]["live_blend_coef"] = my_a2f.A2E.emotions.live_blend_coef
    map_style["a2e"]["post_processing_params"][
        "preferred_emotion_strength"] = my_a2f.A2E.emotions.preferred_emotion_strength
    map_style["a2e"]["post_processing_params"][
        "enable_preferred_emotion"] = my_a2f.A2E.emotions.enable_preferred_emotion
    map_style["a2e"]["post_processing_params"]["emotion_strength"] = my_a2f.A2E.emotions.emotion_strength
    map_style["a2e"]["post_processing_params"]["max_emotions"] = my_a2f.A2E.emotions.max_emotions
    map_style["a2f"]["blendshape_params"]["weight_multipliers"] = dict(zip(
        LIST_FACE_PARAM_ORDER, my_a2f.A2F.api.bs_weight_multipliers))

    controller_internal_client_url = f"{my_controller.audio2face.send_audio.ip}:{my_controller.audio2face.send_audio.port}"
    controller_internal_server_url = f"0.0.0.0:{my_controller.audio2face.receive_anim_data.port}"

    if a2f_server_unidir_url != controller_internal_client_url:
        print(
            f"a2f unidirectional server url is not same as a2f-controller client url: {a2f_server_unidir_url} != {controller_internal_client_url}")
        print(f"Using {a2f_server_unidir_url} for a2f unidirectional server")

    if a2f_client_unidir_url != controller_internal_server_url:
        print(
            f"a2f unidirectional client url is not same as a2f-controller server url: {a2f_client_unidir_url} != {controller_internal_server_url}")
        print(f"Using {a2f_client_unidir_url} for a2f unidirectional client")

    map_advanced["input_sanitization"][
        "max_wait_time_idle_ms"] = my_controller.audio2face.receive_anim_data.max_wait_time_idle_ms
    bidirectional_server_url = f"0.0.0.0:{my_controller.public_interface.port}"
    map_deploy["endpoints"]["bidirectional"]["server"]["url"] = bidirectional_server_url
    if my_a2f.common.stream_number != my_controller.public_interface.max_user_number:
        print(
            f"conflicting stream number ({my_a2f.common.stream_number} != {my_controller.public_interface.max_user_number});"
            f" using  {my_a2f.common.stream_number}")

    map_advanced["input_sanitization"][
        "max_processing_duration_second"] = my_controller.common.max_processing_duration_second
    map_advanced["input_sanitization"][
        "max_audio_buffer_size_second"] = my_controller.common.max_audio_buffer_size_second
    map_advanced["input_sanitization"]["max_audio_clip_size_second"] = my_controller.common.max_audio_clip_size_second

    map_deploy["logging"]["fps_logging_interval_second"] = my_controller.common.fps_logging_interval_second
    map_advanced["garbage_collector"]["enabled"] = my_controller.common.garbage_collector.enabled
    map_advanced["garbage_collector"][
        "interval_run_second"] = my_controller.common.garbage_collector.interval_run_second
    map_advanced["garbage_collector"][
        "max_size_stored_data_second"] = my_controller.common.garbage_collector.max_size_stored_data_second

    save_yaml_file(map_style, os.path.join(output_folder, selected_config_style))
    save_yaml_file(map_deploy, os.path.join(output_folder, DEPLOY_CFG))
    save_yaml_file(map_advanced, os.path.join(output_folder, ADVANCED_CFG))


if __name__ == "__main__":
    args = parse_arguments()
    output_folder = compute_output_directory_name()
    os.makedirs(output_folder)
    print("Processing...")
    if args.option == "ucs":
        convert_ucs_config(output_folder)
    elif args.option == "docker_config":
        convert_docker_config(output_folder)
    else:
        raise Exception(f"invalid selection of choice must be part of: {CHOICE_CONVERT}")
    print(f"Saved new configs to: {output_folder}")
