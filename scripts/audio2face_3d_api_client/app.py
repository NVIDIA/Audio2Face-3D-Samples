# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr
import asyncio
import os
import tempfile
import shutil
import numpy as np
import scipy.io.wavfile
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import cv2
import json

# Audio2Face imports
import a2f_3d.client.auth
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
from nvidia_ace.a2f.v1_pb2 import AudioWithEmotion, EmotionPostProcessingParameters, FaceParameters, BlendShapeParameters
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode
from nvidia_ace.emotion_aggregate.v1_pb2 import EmotionAggregate
import grpc

# Constants
BITS_PER_SAMPLE = 16
CHANNEL_COUNT = 1
AUDIO_FORMAT = AudioHeader.AUDIO_FORMAT_PCM

# API Configuration
API_KEY = "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"

# Function IDs for different models (with tongue animation)
FUNCTION_IDS = {
    "Mark": "8efc55f5-6f00-424e-afe9-26212cd2c630",
    "Claire": "0961a6da-fb9e-4f2e-8491-247e5fd7bf8d",
    "James": "9327c39f-a361-4e02-bd72-e11b4c9b7b5e",
}

# Function IDs without tongue animation (legacy)
FUNCTION_IDS_LEGACY = {
    "Mark (Legacy)": "cf145b84-423b-4222-bfdd-15bb0142b0fd",
    "Claire (Legacy)": "617f80a7-85e4-4bf0-9dd6-dcb61e886142",
    "James (Legacy)": "8082bdcb-9968-4dc5-8705-423ea98b8fc2",
}

# Emotion list
EMOTIONS = ["amazement", "anger", "cheekiness", "disgust", "fear", "grief", "joy", "outofbreath", "pain", "sadness"]

# Sample audio files
SAMPLE_AUDIO_DIR = Path(__file__).parent.parent.parent / "example_audio"
SAMPLE_AUDIOS = {
    "-- Select Sample Audio --": None,
    "Claire - Neutral": "Claire_neutral.wav",
    "Claire - Joy (Mandarin)": "Claire_joy_mandarin.wav",
    "Claire - Anger": "Claire_anger.wav",
    "Claire - Sadness": "Claire_sadness.wav",
    "Claire - Out of Breath (Mandarin)": "Claire_outofbreath_mandarin.wav",
    "Claire - Sadness (5 sec, 16kHz)": "Claire_sadness_16khz_5_sec.wav",
    "Claire - Sadness (10 sec, 16kHz)": "Claire_sadness_16khz_10_sec.wav",
    "Claire - Sadness (20 sec, 16kHz)": "Claire_sadness_16khz_20_sec.wav",
    "Mark - Neutral": "Mark_neutral.wav",
    "Mark - Joy": "Mark_joy.wav",
    "Mark - Anger": "Mark_anger.wav",
    "Mark - Sadness": "Mark_sadness.wav",
    "Mark - Out of Breath": "Mark_outofbreath.wav",
}


def get_default_config():
    """Returns default configuration for Audio2Face"""
    return {
        "face_parameters": {
            "upperFaceStrength": 1.0,
            "upperFaceSmoothing": 0.001,
            "lowerFaceStrength": 1.25,
            "lowerFaceSmoothing": 0.006,
            "faceMaskLevel": 0.6,
            "faceMaskSoftness": 0.0085,
            "skinStrength": 1.0,
            "eyelidOpenOffset": 0.0,
            "lipOpenOffset": 0.0,
        },
        "blendshape_parameters": {
            "enable_clamping_bs_weight": False,
            "multipliers": {
                "EyeBlinkLeft": 1.0, "EyeLookDownLeft": 0.0, "EyeLookInLeft": 0.0,
                "EyeLookOutLeft": 0.0, "EyeLookUpLeft": 0.0, "EyeSquintLeft": 1.0,
                "EyeWideLeft": 1.0, "EyeBlinkRight": 1.0, "EyeLookDownRight": 0.0,
                "EyeLookInRight": 0.0, "EyeLookOutRight": 0.0, "EyeLookUpRight": 0.0,
                "EyeSquintRight": 1.0, "EyeWideRight": 1.0, "JawForward": 0.7,
                "JawLeft": 0.2, "JawRight": 0.2, "JawOpen": 1.0, "MouthClose": 1.0,
                "MouthFunnel": 1.2, "MouthPucker": 1.2, "MouthLeft": 0.2,
                "MouthRight": 0.2, "MouthSmileLeft": 0.8, "MouthSmileRight": 0.8,
                "MouthFrownLeft": 0.4, "MouthFrownRight": 0.4, "MouthDimpleLeft": 0.7,
                "MouthDimpleRight": 0.7, "MouthStretchLeft": 0.1, "MouthStretchRight": 0.1,
                "MouthRollLower": 0.9, "MouthRollUpper": 0.5, "MouthShrugLower": 0.9,
                "MouthShrugUpper": 0.4, "MouthPressLeft": 0.8, "MouthPressRight": 0.8,
                "MouthLowerDownLeft": 0.8, "MouthLowerDownRight": 0.8,
                "MouthUpperUpLeft": 0.8, "MouthUpperUpRight": 0.8, "BrowDownLeft": 1.0,
                "BrowDownRight": 1.0, "BrowInnerUp": 1.0, "BrowOuterUpLeft": 1.0,
                "BrowOuterUpRight": 1.0, "CheekPuff": 0.2, "CheekSquintLeft": 1.0,
                "CheekSquintRight": 1.0, "NoseSneerLeft": 0.8, "NoseSneerRight": 0.8,
                "TongueOut": 0.0,
            },
            "offsets": {k: 0.0 for k in [
                "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
                "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight",
                "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight",
                "EyeSquintRight", "EyeWideRight", "JawForward", "JawLeft", "JawRight",
                "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft",
                "MouthRight", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
                "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
                "MouthStretchRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower",
                "MouthShrugUpper", "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft",
                "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight",
                "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
                "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
                "NoseSneerLeft", "NoseSneerRight", "TongueOut",
            ]},
        },
        "post_processing_parameters": {
            "emotion_contrast": 1.0,
            "live_blend_coef": 0.7,
            "enable_preferred_emotion": False,
            "preferred_emotion_strength": 0.5,
            "emotion_strength": 0.6,
            "max_emotions": 3,
        },
        "emotion_with_timecode_list": {
            "emotion_with_timecode1": {
                "time_code": 0.0,
                "emotions": {e: 0.0 for e in EMOTIONS}
            }
        }
    }


def create_config_with_emotion(primary_emotion: str, emotion_strength: float = 1.0):
    """Create config with specified primary emotion"""
    config = get_default_config()
    emotions = {e: 0.0 for e in EMOTIONS}
    if primary_emotion.lower() in emotions:
        emotions[primary_emotion.lower()] = emotion_strength
    config["emotion_with_timecode_list"]["emotion_with_timecode1"]["emotions"] = emotions
    return config


class A2FProcessor:
    """Audio2Face-3D Processor"""
    
    def __init__(self):
        self.bs_names = []
        self.animation_key_frames = []
        self.audio_buffer = b''
        self.audio_header = None
        self.emotion_key_frames = {"input": [], "a2f_smoothed_output": []}
    
    def reset(self):
        self.bs_names = []
        self.animation_key_frames = []
        self.audio_buffer = b''
        self.audio_header = None
        self.emotion_key_frames = {"input": [], "a2f_smoothed_output": []}
    
    async def read_from_stream(self, stream, progress_callback=None):
        """Read animation data from gRPC stream"""
        frame_count = 0
        while True:
            message = await stream.read()
            if message == grpc.aio.EOF:
                break
            
            if message.HasField("animation_data_stream_header"):
                header = message.animation_data_stream_header
                self.bs_names = list(header.skel_animation_header.blend_shapes)
                self.audio_header = header.audio_header
            
            elif message.HasField("animation_data"):
                animation_data = message.animation_data
                
                # Parse emotion data
                emotion_aggregate = EmotionAggregate()
                if animation_data.metadata.get("emotion_aggregate") and \
                   animation_data.metadata["emotion_aggregate"].Unpack(emotion_aggregate):
                    for ewt in emotion_aggregate.input_emotions:
                        self.emotion_key_frames["input"].append({
                            "time_code": ewt.time_code,
                            "emotion_values": dict(ewt.emotion),
                        })
                    for ewt in emotion_aggregate.a2f_smoothed_output:
                        self.emotion_key_frames["a2f_smoothed_output"].append({
                            "time_code": ewt.time_code,
                            "emotion_values": dict(ewt.emotion),
                        })
                
                # Parse blendshape data
                for blendshapes in animation_data.skel_animation.blend_shape_weights:
                    bs_values_dict = dict(zip(self.bs_names, blendshapes.values))
                    self.animation_key_frames.append({
                        "timeCode": blendshapes.time_code,
                        "blendShapes": bs_values_dict
                    })
                    frame_count += 1
                
                self.audio_buffer += animation_data.audio.audio_buffer
                
                if progress_callback:
                    progress_callback(frame_count)
            
            elif message.HasField("status"):
                status = message.status
                return status.code == 0, status.message
        
        return True, "Stream completed"
    
    async def write_to_stream(self, stream, config: dict, audio_data: np.ndarray, sample_rate: int):
        """Write audio data to gRPC stream"""
        # Send header
        audio_stream_header = AudioStream(
            audio_stream_header=AudioStreamHeader(
                audio_header=AudioHeader(
                    samples_per_second=sample_rate,
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
        await stream.write(audio_stream_header)
        
        # Send audio in chunks
        chunk_size = sample_rate  # 1 second chunks
        for i in range(len(audio_data) // chunk_size + 1):
            chunk = audio_data[i * chunk_size: (i + 1) * chunk_size]
            if len(chunk) == 0:
                continue
            
            if i == 0:
                # First chunk includes emotions
                list_emotion_tc = [
                    EmotionWithTimeCode(
                        emotion={**v["emotions"]},
                        time_code=v["time_code"]
                    ) for v in config["emotion_with_timecode_list"].values()
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
                await stream.write(
                    AudioStream(
                        audio_with_emotion=AudioWithEmotion(
                            audio_buffer=chunk.astype(np.int16).tobytes()
                        )
                    )
                )
        
        # Signal end of audio
        await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))


def create_visualization_video(animation_frames: list, audio_path: str, output_path: str, fps: int = 30):
    """
    Create a visualization video showing blendshape values as animated bars
    with the audio track.
    """
    if not animation_frames:
        return None
    
    # Video dimensions
    width, height = 1280, 720
    
    # Key blendshapes to visualize
    key_blendshapes = [
        "JawOpen", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
        "MouthPucker", "MouthFunnel", "BrowInnerUp", "BrowDownLeft", "BrowDownRight",
        "EyeBlinkLeft", "EyeBlinkRight", "EyeWideLeft", "EyeWideRight",
        "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft", "NoseSneerRight",
    ]
    
    # Calculate duration based on animation frames
    if animation_frames:
        duration = animation_frames[-1]["timeCode"]
    else:
        duration = 1.0
    
    total_frames = int(duration * fps) + 1
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # Color scheme
    bg_color = (30, 30, 40)  # Dark background
    bar_color = (0, 200, 100)  # Green bars
    text_color = (255, 255, 255)  # White text
    accent_color = (100, 150, 255)  # Blue accent
    
    # Animation frame index
    anim_idx = 0
    
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        
        # Find the closest animation frame
        while anim_idx < len(animation_frames) - 1 and \
              animation_frames[anim_idx + 1]["timeCode"] <= current_time:
            anim_idx += 1
        
        # Create frame
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Title
        cv2.putText(frame, "NVIDIA Audio2Face-3D Blendshape Visualization",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, accent_color, 2)
        
        # Time display
        cv2.putText(frame, f"Time: {current_time:.2f}s / {duration:.2f}s",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
        
        # Progress bar
        progress = current_time / duration if duration > 0 else 0
        cv2.rectangle(frame, (40, 105), (width - 40, 115), (60, 60, 70), -1)
        cv2.rectangle(frame, (40, 105), (int(40 + (width - 80) * progress), 115), accent_color, -1)
        
        # Draw blendshape bars
        if animation_frames and anim_idx < len(animation_frames):
            blendshapes = animation_frames[anim_idx].get("blendShapes", {})
            
            bar_height = 25
            bar_max_width = 400
            start_y = 150
            col1_x = 50
            col2_x = 650
            
            for i, bs_name in enumerate(key_blendshapes):
                value = blendshapes.get(bs_name, 0.0)
                value = max(0, min(1, value))  # Clamp to 0-1
                
                # Determine column
                if i < len(key_blendshapes) // 2:
                    x = col1_x
                    y = start_y + i * (bar_height + 15)
                else:
                    x = col2_x
                    y = start_y + (i - len(key_blendshapes) // 2) * (bar_height + 15)
                
                # Draw label
                cv2.putText(frame, bs_name, (x, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Draw bar background
                bar_x = x + 180
                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_width, y + bar_height),
                              (60, 60, 70), -1)
                
                # Draw bar value
                bar_width = int(bar_max_width * value)
                if bar_width > 0:
                    # Color gradient based on value
                    color = (int(bar_color[0] * (1 - value * 0.5)),
                             int(bar_color[1]),
                             int(bar_color[2] * (1 - value * 0.3)))
                    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                                  color, -1)
                
                # Draw value text
                cv2.putText(frame, f"{value:.2f}", (bar_x + bar_max_width + 10, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Footer
        cv2.putText(frame, "Generated with NVIDIA Audio2Face-3D API",
                    (40, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        out.write(frame)
    
    out.release()
    
    # Combine video with audio using ffmpeg
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.remove(temp_video)
    except Exception as e:
        # If ffmpeg fails, just use the video without audio
        shutil.move(temp_video, output_path)
        print(f"Warning: Could not add audio to video: {e}")
    
    return output_path


async def process_audio_async(audio_path: str, model: str, emotion: str, emotion_strength: float, 
                               progress=gr.Progress()):
    """Process audio through Audio2Face-3D API"""
    
    # Get function ID
    all_models = {**FUNCTION_IDS, **FUNCTION_IDS_LEGACY}
    function_id = all_models.get(model)
    if not function_id:
        return None, None, f"Unknown model: {model}"
    
    # Read audio file
    try:
        sample_rate, audio_data = scipy.io.wavfile.read(audio_path)
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
    except Exception as e:
        return None, None, f"Error reading audio: {str(e)}"
    
    # Create config with emotion
    config = create_config_with_emotion(emotion, emotion_strength)
    
    # Setup gRPC connection
    metadata_args = [
        ("function-id", function_id),
        ("authorization", f"Bearer {API_KEY}")
    ]
    
    try:
        channel = a2f_3d.client.auth.create_channel(
            uri="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
            metadata=metadata_args
        )
        stub = A2FControllerServiceStub(channel)
        stream = stub.ProcessAudioStream()
        
        # Process
        processor = A2FProcessor()
        
        def update_progress(frame_count):
            progress(frame_count / 100, desc=f"Processing frame {frame_count}...")
        
        # Run write and read concurrently
        write_task = asyncio.create_task(
            processor.write_to_stream(stream, config, audio_data, sample_rate)
        )
        read_task = asyncio.create_task(
            processor.read_from_stream(stream, update_progress)
        )
        
        await write_task
        success, message = await read_task
        
        if not success:
            return None, None, f"API Error: {message}"
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path(f"outputs/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        output_audio = output_dir / "output.wav"
        if processor.audio_buffer and processor.audio_header:
            audio_out = np.frombuffer(processor.audio_buffer, dtype=np.int16)
            scipy.io.wavfile.write(str(output_audio), processor.audio_header.samples_per_second, audio_out)
        else:
            shutil.copy(audio_path, output_audio)
        
        # Save animation data as CSV
        df = pd.json_normalize(processor.animation_key_frames)
        csv_path = output_dir / "animation_frames.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for download
        json_path = output_dir / "animation_data.json"
        with open(json_path, 'w') as f:
            json.dump({
                "blendshape_names": processor.bs_names,
                "frames": processor.animation_key_frames,
                "emotions": processor.emotion_key_frames
            }, f, indent=2)
        
        # Create visualization video
        progress(0.8, desc="Creating visualization video...")
        video_path = output_dir / "visualization.mp4"
        create_visualization_video(
            processor.animation_key_frames,
            str(output_audio),
            str(video_path)
        )
        
        progress(1.0, desc="Complete!")
        
        return str(video_path), str(json_path), f"✅ Success! Generated {len(processor.animation_key_frames)} animation frames."
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def process_audio(audio_path, sample_audio, model, emotion, emotion_strength, progress=gr.Progress()):
    """Wrapper to run async function"""
    # Use sample audio if selected, otherwise use uploaded audio
    if sample_audio and sample_audio != "-- Select Sample Audio --":
        audio_file = SAMPLE_AUDIOS.get(sample_audio)
        if audio_file:
            audio_path = str(SAMPLE_AUDIO_DIR / audio_file)
    
    if audio_path is None:
        return None, None, "Please upload an audio file or select a sample audio."
    
    return asyncio.run(process_audio_async(audio_path, model, emotion, emotion_strength, progress))


# Create Gradio Interface
def create_ui():
    with gr.Blocks(title="NVIDIA Audio2Face-3D") as demo:
        gr.Markdown("""
        # 🎭 NVIDIA Audio2Face-3D
        ### Convert Audio to Facial Animation
        
        Upload your audio file and generate facial animation blendshapes with emotion control.
        The output video shows a visualization of the generated blendshapes synced with your audio.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                with gr.Tab("📁 Upload Audio"):
                    audio_input = gr.Audio(
                        label="Upload Audio (WAV format, 16-bit PCM recommended)",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                
                with gr.Tab("🎵 Sample Audio"):
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLE_AUDIOS.keys()),
                        value="-- Select Sample Audio --",
                        label="Select Sample Audio",
                        info="Choose from pre-loaded sample audio files"
                    )
                    gr.Markdown("*Tip: Use Claire samples with Claire model, Mark samples with Mark model*")
                
                model_dropdown = gr.Dropdown(
                    choices=list(FUNCTION_IDS.keys()) + list(FUNCTION_IDS_LEGACY.keys()),
                    value="Claire",
                    label="🎭 Character Model",
                    info="Select the face model to use"
                )
                
                with gr.Accordion("🎨 Emotion Settings", open=True):
                    emotion_dropdown = gr.Dropdown(
                        choices=EMOTIONS,
                        value="joy",
                        label="Primary Emotion"
                    )
                    emotion_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Emotion Strength"
                    )
                
                process_btn = gr.Button("🚀 Generate Animation", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Output")
                
                status_text = gr.Textbox(label="Status", interactive=False)
                
                video_output = gr.Video(label="Animation Visualization")
                
                json_output = gr.File(label="📄 Download Animation Data (JSON)")
        
        gr.Markdown("""
        ---
        ### 📋 Instructions
        1. **Select Audio**: Upload a WAV file, record from microphone, OR select a sample audio
        2. **Select Model**: Choose from Mark, Claire, or James characters
        3. **Set Emotion**: Pick the primary emotion and adjust strength
        4. **Generate**: Click the button and wait for processing
        
        ### 📊 Output Files
        - **Video**: Visualization of blendshape values over time with audio
        - **JSON**: Complete animation data including all 52 ARKit blendshapes
        
        ### 🔗 Resources
        - [NVIDIA Audio2Face Documentation](https://docs.nvidia.com/ace/latest/modules/a2f-docs/index.html)
        - [ARKit Blendshape Reference](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation)
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, sample_dropdown, model_dropdown, emotion_dropdown, emotion_strength],
            outputs=[video_output, json_output, status_text]
        )
    
    return demo


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Launch the app
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
