# Audio2Face-3D NIM API Client

A sample Python application to showcase the Audio2Face-3D NIM hosted on NVIDIA Cloud Functions (NVCF). This client demonstrates how to send audio files and receive facial animation blendshapes data using NVIDIA's Audio2Face-3D API.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Gradio Web Interface](#gradio-web-interface)
- [Configuration](#configuration)
- [Available Models](#available-models)
- [Sample Audio Files](#sample-audio-files)
- [Output](#output)
- [Project Structure](#project-structure)
- [License](#license)

## ✨ Features

- **CLI Client**: Command-line interface for batch processing audio files
- **Web Interface**: Interactive Gradio-based web UI for real-time testing
- **Multiple Character Models**: Support for Mark, Claire, and James stylization models
- **Emotion Control**: Configurable emotion parameters for animation generation
- **Blendshape Output**: ARKit-compatible blendshape weights export
- **Audio Streaming**: Efficient gRPC-based audio streaming

## 📦 Prerequisites

- Python 3.8+
- NVIDIA API Key (from [NVIDIA API Catalog](https://build.nvidia.com/))
- Function ID for the Audio2Face-3D API

## 🚀 Installation

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip3 install -r requirements
pip3 install ../../proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
```

> **Note**: The `nvidia_ace-1.2.0` wheel is compatible with Audio2Face-3D NIM 1.3

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical operations |
| scipy | 1.13.0 | Audio file I/O |
| grpcio | 1.72.0rc1 | gRPC communication |
| protobuf | 4.24.1 | Protocol buffers |
| PyYAML | 6.0.1 | Configuration parsing |
| pandas | 2.2.2 | Data manipulation |
| gradio | 6.0.1 | Web interface |
| opencv-python-headless | 4.12.0.88 | Image processing |

## 💻 Usage

### Command Line Interface

Run the CLI client with the following command:

```bash
python3 ./nim_a2f_3d_client.py <audio_file.wav> <config.yml> --apikey <API_KEY> --function-id <FUNCTION_ID>
```

#### Example

```bash
python3 ./nim_a2f_3d_client.py \
    ../../example_audio/Claire_neutral.wav \
    config/config_claire.yml \
    --apikey nvapi-xxxxxxxxxxxx \
    --function-id 0961a6da-fb9e-4f2e-8491-247e5fd7bf8d
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `file` | ✅ | PCM 16-bit single channel audio file in WAV format |
| `config` | ✅ | YAML configuration file for inference parameters |
| `--apikey` | ✅ | NGC API Key from NVIDIA API Catalog |
| `--function-id` | ✅ | Function ID for the specific character model |

### Gradio Web Interface

Launch the interactive web interface:

```bash
python3 ./app.py
```

The web interface provides:
- Drag-and-drop audio upload
- Sample audio selection
- Real-time emotion parameter adjustment
- Visual blendshape output preview
- CSV export functionality

## ⚙️ Configuration

Configuration files are located in the `config/` directory:

- `config_claire.yml` - Claire character settings
- `config_james.yml` - James character settings  
- `config_mark.yml` - Mark character settings

### Face Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `upperFaceStrength` | Range of motion for upper face | 1.0 |
| `upperFaceSmoothing` | Temporal smoothing for upper face | 0.001 |
| `lowerFaceStrength` | Range of motion for lower face | 1.25 |
| `lowerFaceSmoothing` | Temporal smoothing for lower face | 0.006 |
| `faceMaskLevel` | Boundary between upper/lower regions | 0.6 |
| `faceMaskSoftness` | Blend smoothness between regions | 0.0085 |
| `skinStrength` | Range of motion for skin | 1.0 |
| `eyelidOpenOffset` | Default eyelid pose adjustment | 0.0 |
| `lipOpenOffset` | Default lip pose adjustment | 0.0 |

### Blendshape Parameters

The configuration supports ARKit-compatible blendshape multipliers and offsets. See [Apple ARKit documentation](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation) for more details.

## 🎭 Available Models

### With Tongue Animation

| Character | Function ID |
|-----------|-------------|
| Mark | `8efc55f5-6f00-424e-afe9-26212cd2c630` |
| Claire | `0961a6da-fb9e-4f2e-8491-247e5fd7bf8d` |
| James | `9327c39f-a361-4e02-bd72-e11b4c9b7b5e` |

### Legacy (Without Tongue Animation)

| Character | Function ID |
|-----------|-------------|
| Mark | `cf145b84-423b-4222-bfdd-15bb0142b0fd` |
| Claire | `617f80a7-85e4-4bf0-9dd6-dcb61e886142` |
| James | `8082bdcb-9968-4dc5-8705-423ea98b8fc2` |

## 🎵 Sample Audio Files

Sample audio files are available in `../../example_audio/`:

| File | Description |
|------|-------------|
| `Claire_neutral.wav` | Claire - Neutral emotion |
| `Claire_anger.wav` | Claire - Anger emotion |
| `Claire_joy_mandarin.wav` | Claire - Joy (Mandarin) |
| `Claire_sadness.wav` | Claire - Sadness emotion |
| `Claire_outofbreath_mandarin.wav` | Claire - Out of breath (Mandarin) |
| `Mark_neutral.wav` | Mark - Neutral emotion |
| `Mark_joy.wav` | Mark - Joy emotion |
| `Mark_anger.wav` | Mark - Anger emotion |
| `Mark_sadness.wav` | Mark - Sadness emotion |
| `Mark_outofbreath.wav` | Mark - Out of breath |

## 📤 Output

The application generates the following outputs:

1. **Blendshapes CSV**: Animation keyframes with blendshape names, values, and timecodes
2. **Emotions CSV**: Emotion data with timecodes
3. **Audio WAV**: Processed audio output (`out.wav`)

### Supported Emotions

- Amazement
- Anger
- Cheekiness
- Disgust
- Fear
- Grief
- Joy
- Out of Breath
- Pain
- Sadness

## 📁 Project Structure

```
audio2face_3d_api_client/
├── README.md                 # This file
├── nim_a2f_3d_client.py      # CLI client script
├── app.py                    # Gradio web interface
├── requirements              # Python dependencies
├── config/
│   ├── config_claire.yml     # Claire model configuration
│   ├── config_james.yml      # James model configuration
│   └── config_mark.yml       # Mark model configuration
└── a2f_3d/
    └── client/
        ├── auth.py           # Authentication utilities
        └── service.py        # gRPC service handlers
```

## 🔄 How It Works

1. **Read Audio**: Loads audio data from a 16-bit PCM WAV file
2. **Load Config**: Parses emotion and face parameters from YAML configuration
3. **Stream Audio**: Sends audio data via gRPC to the Audio2Face-3D API
4. **Receive Animation**: Gets back blendshape weights, audio, and emotion data
5. **Export Results**: Saves animation keyframes and emotions to CSV files

## 📄 License

```
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Licensed under the Apache License, Version 2.0. See [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) for details.
