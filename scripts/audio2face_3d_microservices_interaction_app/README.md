# Sample application interacting with A2F-3D

In this folder there are 2 sample applications interacting with Audio2Face-3D:

* `a2f_3d.py` - This application is a sample Python3 application to send audio and receive animation data and emotion data through the A2F-3D NIM.
* `nim_performance_test.py` - This application is a Python3 application to measure latency and frames per second (FPS) for an A2F-3D instance.

## Prerequisites

Both applications require the following dependencies:

* python3
* python3-venv

You will need to provide an audio file to test out.

You will need to have a running instance of Audio2Face-3D NIM.

### Setting up the environment

Start by creating a python venv using

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the the gRPC proto for python by:

* Quick installation: Install the provided `nvidia_ace` python wheel package from the
[sample_wheel/](../../proto/sample_wheel) folder.

  ```bash
  pip3 install ../../proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
  ```

* Manual installation: Follow the [README](../../proto/README.md) in the
[proto/](../../proto/) folder.

Then install the required dependencies:

```bash
pip3 install -r requirements.txt
```

### Check that the service is up and running

```bash
./a2f_3d.py health_check --url <ip>:<port>
```

### Running inference requests with the a2f_3d.py script

```bash
./a2f_3d.py run_inference <audio_file.wav> <config.yml> -u <ip>:<port> [--skip-print-to-files]
```

The scripts takes three mandatory parameters:

* an audio file at format PCM 16 bits
* a yaml configuration file for the emotions parameters; you can find sample configuration files under `config/` folder.
* a parameter `-u` which is the URL of a running A2F-3D NIM

The scripts has 2 optional parameters:

* `--skip-print-to-files`. When present, the script does not output the animation and emotion data to files
* `--print-fps`: When present, the script will also print data used for performance evaluation

This script also measures and prints to console the latency and FPS for processing the given audio file, but the result
is not statistically meaningful for your Audio2Face-3D deployment. To perform a performance test, follow
[Running the nim_performance_test.py script](#running-the-nim_performance_testpy-script).

E.g.:
For a local deployment with default configuration or when using the
`docker-compose` quick start, you can use `127.0.0.1:52000`.

## What does a2f_3d.py script do?

1. Reads the audio data from a wav 16bits PCM file.
2. Reads emotions and parameters from the yaml configuration file.
3. Sends emotions, parameters and audio to the A2F-3D.
4. Receives back blendshapes, audio and emotions.
5. Saves blendshapes as animation key frames in a csv file with their name,
value and time codes.
6. Saves emotions data in multiple csv files with their values and time codes.
7. Saves the received audio as out.wav.

## Notes

The API to retrieve the emotions via metadata object is still alpha and
susceptible to changes.

### Running the nim_performance_test.py script

Usage (`./nim_performance_test.py --help`)

```bash
./nim_performance_test.py --request-nb <REQUEST_NB> --max-stream-nb <MAX_STREAM_NB> --url <ip>:<port>
```

The scripts takes three mandatory parameters:

* `--request-nb` - Number of requests to simulate for each audio file.
* `--max-stream-nb` - Maximum number of concurrent streams connecting to Audio2Face-3D. This needs to correspond to the `common.stream_number` config option used when starting the A2F-3D NIM.
* `--url` - URL of the A2F-3D NIM.

The script will use the `a2f_3d.py` to send requests to a running Audio2Face-3D NIM.

E.g.:
For a local deployment with default configuration or when using the
`docker-compose` quick start, you can use `127.0.0.1:52000`.

## What does nim_performance_test script do?

The script benchmarks performance on 6 audio files, of 5, 10, 20 seconds durations
and 16khz and 44.1khz sample rates. These audio files are under `example_audio/`
subfolder of this repo:

```bash
├── Claire_sadness_16khz_10_sec.wav
├── Claire_sadness_16khz_20_sec.wav
├── Claire_sadness_16khz_5_sec.wav
├── Claire_sadness_44.1khz_10_sec.wav
├── Claire_sadness_44.1khz_20_sec.wav
├── Claire_sadness_44.1khz_5_sec.wav
```

For each audio file, the script:

1. Starts `MAX_STREAM_NB` clients connecting to Audio2Face-3D.
2. Each client makes `REQUEST_NB` requests for the same audio file. It will run
the following command `REQUEST_NB` number of times.

```bash
python3 a2f_3d.py run_inference <audio_file> config/config_mark.yml -u <url> --skip-print-to-files
```

3. Read the latency and FPS calculations printed out to stdout by all `a2f_3d.py` runs.
4. Compute simple statistics of latency and FPS.
5. Print the results to `performance_output_{VERSION}/` folder.

Inside that folder you will find 4 files:

* fps_stream_*.txt: this file indicates the expected minimum FPS number for 99% of the requests made to the A2F Service
* latency_stream*.txt: This is similar but for latencies; 99% of requests start outputting data below the "Worst case scenario" number
* stream*.csv: raw CSV latencies and FPS data that can be used for plotting
* stream*.png: generated image from the raw CSV data