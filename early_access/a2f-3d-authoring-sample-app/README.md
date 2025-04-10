# Sample python scripts to test A2F-3D Authoring MS

The `client_local_deploy.py` script provides sample code for interacting with the Audio2Face-3D Authoring Microservice.
First make sure the Audio2Face-3D Authoring Microservice is started.

The `client_nvcf_deploy.py` script provides sample code for interacting with the Audio2Face-3D Authoring Microservice
deployed on NVCF. To use this script, make sure you have the right API keys.

## Requirements

This sample scripts are intended to be used on python3.10

Start by creating a python virtual environment using:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installing python packages

```bash
pip3 install -r requirements.txt
```

### Installing gRPC python proto

Install the the gRPC proto for python by:

* Quick installation: Install the provided `nvidia_ace` python wheel package from the
[sample_wheel/](../../proto/sample_wheel) folder.

Note: This wheel is compatible with Audio2Face-3D NIM 1.3

  ```bash
  pip3 install ../../proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
  ```

* Manual installation: Follow the [README](../../proto/README.md) in the
[proto/](../../proto/) folder.

## client_local_deploy

This script has 3 modes:

* health_check: checks if the service is reachable.
* data_capture: capture data from a full audio clip.
* latency: record latency data of the Authoring service.

### Health Checking

```bash
usage: client_local_deploy.py health_check [-h] --url URL

options:
  -h, --help  show this help message and exit
  --url URL   GRPC service URL
```

#### Online

This is using a correct URL.

```bash
$ python3 client_local_deploy.py health_check --url 0.0.0.0:50051
Service 0.0.0.0:50051 is ONLINE
```

#### Offline

This is using an invalid URL.

```bash
$ python3 client_local_deploy.py health_check --url 0.0.0.0:50052
Error checking health: <_InactiveRpcError of RPC that terminated with:
 status = StatusCode.UNAVAILABLE
 details = "failed to connect to all addresses; last error: UNKNOWN: ipv4:0.0.0.0:50052: Failed to connect to remote host: Connection refused"
 debug_error_string = "UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:0.0.0.0:50052: Failed to connect to remote host: Connection refused", grpc_status:14, created_time:"2024-09-06T17:58:55.301073978+02:00"}"
>
Service 0.0.0.0:50052 is OFFLINE
```

### Process Audio clip

```bash
usage: client_local_deploy.py data_capture [-h] --url URL --audio-clip AUDIO_CLIP [--print-bs-names]

options:
  -h, --help            show this help message and exit
  --url URL             GRPC service URL
  --audio-clip AUDIO_CLIP
                        Path to audio clip file
  --print-bs-names      Optional. If enabled, the command will print out the names of the returned blendshapes.
```

This call will save blendshapes to a csv file similarly as A2F-3D sample app

```bash
$ python3 client_local_deploy.py data_capture --url 0.0.0.0:50051 --audio-clip ../../example_audio/Claire_neutral.wav  --print-bs-names
audio_clip_hash=5c8de20d275e53fcfceba0f39505f8b2b1fd960294518fbadf210d51dcc3a2e4
Perform sequential requests for the full audio clip...
3.73 seconds processed    
Blendshape names are:['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft',
'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight',
'JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft',
'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft',
'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut',
'HeadRoll_deprecated', 'HeadPitch_deprecated', 'HeadYaw_deprecated']
Saved results to output_blendshape.csv
Saved results to output_emotions.csv
```

### Latency test

```bash
usage: client_local_deploy.py latency [-h] --url URL --audio-clip AUDIO_CLIP --number-requests NUMBER_REQUESTS --concurrent CONCURRENT [--print-bs-names]

options:
  -h, --help            show this help message and exit
  --url URL             GRPC service URL
  --audio-clip AUDIO_CLIP
                        Path to audio clip file
  --number-requests NUMBER_REQUESTS
                        Number of requests to simulate
  --concurrent CONCURRENT
                        Number of concurrent requests
  --print-bs-names      Optional. If enabled, the command will print out the names of the returned blendshapes.
```

The following call will perform performance recording for latency. 2 parameters can be tweaked: the number of requests
and the number of parallel connections. Too high concurrency will increase latencies. Too low concurrency will increase
processing time.

```bash
$ python3 client_local_deploy.py latency --url 0.0.0.0:50051 --audio-clip ../../example_audio/Claire_neutral.wav --number-requests 1000 --concurrent 10 --print-bs-names
Computing data for 1000 requests with concurrency of 10
audio_clip_hash=5c8de20d275e53fcfceba0f39505f8b2b1fd960294518fbadf210d51dcc3a2e4
........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
Blendshape names are:['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft',
'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight',
'JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft',
'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft',
'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut',
'HeadRoll_deprecated', 'HeadPitch_deprecated', 'HeadYaw_deprecated']
Saved lantecy data in output_perf.txt
Plotted latency data in output_latency.png
```

## client_nvcf_deploy

This script has the same modes as the one above, except that it communicates with the NVCF deployment and not a local one.

```bash
GRPC client tool connecting to NVCF deployment

positional arguments:
  {health_check,data_capture,latency}
    health_check        Check GRPC service health
    data_capture        Check data integrity of GRPC service
    latency             Check latency of GRPC service

options:
  -h, --help            show this help message and exit
```

### Health check

```bash
python3 client_nvcf_deploy.py health_check --apikey {API_KEY} --function-id {FUNCTION_ID} --version-id {VERSION_ID}
```

The script checks if the NVCF deployment is Online or Offline.

### Data capture

Example usage

```bash
python3 client_nvcf_deploy.py data_capture --function-id {FUNCTION_ID} --version-id {VERSION_ID} --apikey {API_KEY} --audio-clip ../../example_audio/Claire_neutral.wav [--print-bs-names]
```

The script does the following:

1. Reads the audio data from a wav 16bits PCM file.
2. Uploads the audio to the server.
3. For each frame, calls the Authoring service to get the blendshape animation data.
4. Saves blendshapes in a csv file with their name, value and time codes.
5. Saves the emotions in a csv file with their name, value and time codes.
6. If `--print-bs-names` is enabled, the script prints out the blendshape names computed by the microservice.

### Performance - latency

```bash
python3 client_nvcf_deploy.py latency --apikey {API_KEY} --function-id {FUNCTION_ID} --version-id {VERSION_ID} --audio-clip ../../example_audio/Claire_neutral.wav --number-requests 100 --concurrent 10 [--print-bs-names]
```

The above command will perform performance recording for latency. 2 parameters can be tweaked: the number of requests
and the number of parallel connections. Too high concurrency will increase latencies. Too low concurrency will increase
processing time.

If `--print-bs-names` is enabled, the script prints out the blendshape names computed by the microservice.
