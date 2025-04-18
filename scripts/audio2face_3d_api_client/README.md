# Sample Application connecting to Audio2Face-3D NIM hosted on NVCF

A sample Python application to showcase the Audio2Face-3D NIM hosted on NVIDIA Cloud Functions (NVCF).

## Getting started

Start by creating a python venv using:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required dependencies:

```bash
pip3 install -r requirements
pip3 install ../../proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
```

Note: This wheel is compatible with Audio2Face-3D NIM 1.3


```bash
python3 ./nim_a2f_3d_client.py <audio_file.wav> <config.yml> --apikey <API_KEY> --function-id <Function_ID>
```

By Default:

```bash
python3 ./nim_a2f_3d_client.py ../../example_audio/Claire_neutral.wav config/config_claire.yml --apikey <API_KEY> --function-id <Function_ID>
```

The scripts takes four mandatory parameters, an audio file at format PCM 16 bits,
 a yaml configuration file for the emotions parameters, the API Key generated by API Catalogue, and the Function ID
 used to access the API function.

--apikey for the API Key generated through the API Catalogue
--function-id for the Function ID provided to access the API function.

## What does this example do?

1. Reads the audio data from a wav 16bits PCM file
2. Reads emotions and parameters from the yaml configuration file
3. Sends emotions, parameters and audio to the A2F-3D
4. Receives back blendshapes, audio and emotions
5. Saves blendshapes as animation key frames in a csv file with their name, value
and time codes
6. Same process for the emotion data.
7. Saves the received audio as out.wav (Should be the same as input audio)
