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

""" Tool to simulate multiple clients connecting to A2X NIM with varied audio 
file lengths and sample rates.
"""

import argparse
import logging
import os
import random
import re
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
OUTPUT_FOLDER = 'performance_output'

def add_values_on_bars(ax, bars):
    """ Add values on top of the bars in bar plots.
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)

def plot_metrics(data, output_file):
    """ Plots the metrics for latency and FPS.
    """
    # Extract relevant columns for plotting
    latency_columns = [
        '1st packet min latency (ms)',
        '1st packet percentile1 latency (ms)',
        '1st packet percentile5 latency (ms)',
        '1st packet percentile10 latency (ms)',
        '1st packet median latency (ms)',
        '1st packet mean latency (ms)',
        '1st packet percentile90 latency (ms)',
        '1st packet percentile95 latency (ms)',
        '1st packet percentile99 latency (ms)',
        '1st packet max latency (ms)'
    ]
    fps_columns = [
        'min fps',
        'percentile1 fps',
        'percentile5 fps',
        'percentile10 fps',
        'median fps',
        'mean fps',
        'percentile90 fps',
        'percentile95 fps',
        'percentile99 fps',
        'max fps'
    ]

    # Combine audio length and sample rate for x-axis labels
    x_labels = [f"{row['audio_length (s)']}s, {row['sample_rate']}" for idx, row in data.iterrows()]
    request_number = data['request_number'].iloc[0]
    max_stream_number = data['max_stream_number'].iloc[0]

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    indices = np.arange(len(data['audio_length (s)']))
    bar_width = 0.1

    # Plot latency metrics
    for i, col in enumerate(latency_columns):
        bars = ax1.bar(indices + i * bar_width, data[col], bar_width, label=col)
        add_values_on_bars(ax1, bars)

    ax1.set_xlabel('Audio Length and Sample Rate')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'Latency Metrics (Max Stream Number: {max_stream_number}, Number of Requests: {request_number})')
    ax1.set_xticks(indices + bar_width * (len(latency_columns) - 1) / 2)
    ax1.set_xticklabels(x_labels, rotation=0, ha='center')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot FPS metrics
    for i, col in enumerate(fps_columns):
        bars = ax2.bar(indices + i * bar_width, data[col], bar_width, label=col)
        add_values_on_bars(ax2, bars)

    ax2.set_xlabel('Audio Length and Sample Rate')
    ax2.set_ylabel('FPS')
    ax2.set_title(f'FPS Metrics (Max Stream Number: {max_stream_number}, Number of Requests: {request_number})')
    ax2.set_xticks(indices + bar_width * (len(fps_columns) - 1) / 2)
    ax2.set_xticklabels(x_labels, rotation=0, ha='center')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(output_file)
    plt.close()
    logger.info(f'Results plotted in {output_file}')


def create_parser():
    parser = argparse.ArgumentParser(description='Call Audio2Face-3D NIM with different audiofiles. '
                                                 'Outputs NIM performance in a csv file in output/ folder. '
                                                 'Uses a2f_3d.py to create multiple clients to connect to A2F-3D NIM.')
    parser.add_argument('--request-nb', type=int, help='Number of requests to simulate for each audio file', required=True)
    parser.add_argument('--max-stream-nb', type=int, help='Maximum number of A2F-3D streams', required=True)
    parser.add_argument('--url', help="IP of the Audio2Face-3D NIM", required=True)
    return parser


def run_single_client(client_info):
    """ Simulate a single client with specified client info
    """
    audio_file = client_info['audio_file']
    url = client_info['url']
    cmd = ['python3', 'a2f_3d.py', 'run_inference', audio_file, 'config/config_mark.yml', '-u', url, '--skip-print-to-files', "--print-fps"]
    time.sleep(client_info['sleep_time'])

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        latency_ms, fps = get_single_client_perf(proc.stdout)
        print(".", end="")
        sys.stdout.flush()
        return latency_ms, fps
    except Exception as e:
        return -1, -1

def get_single_client_perf(client_stdout):
    """ Extract latency and FPS data from single client simulation for a single request.
    """
    sent_str, recv_str, fps_string = "", "", ""
    latency_ms, fps = -1, -1
    pattern = r'(\d+\.\d+)'
    lines = client_stdout.strip().split('\n')

    # Find the lines in output that container raw data
    for line in lines:
        if "First packet sent" in line:
            sent_str += line
        if "First packet received" in line:
            recv_str += line
        if "FPS" in line:
            fps_string += line

    # Find the timestamp using regular expression
    fps_match = re.search(pattern, fps_string)
    time_sent_match = re.search(pattern, sent_str)
    time_recv_match = re.search(pattern, recv_str)

    if fps_match:
        fps = float(fps_match.group(1))

    if time_sent_match and time_recv_match:
        time_sent = float(time_sent_match.group(1))
        time_recv = float(time_recv_match.group(1))
        latency_ms = (time_recv - time_sent) * 1000

    return latency_ms, fps


class StatsHolder:
    """
    Holds statistical metrics for performance data.

    This class calculates and stores various statistics (e.g., minimum, percentiles, median, mean, maximum) 
    for a given dataset, for example for a set of latency or FPS measurements.

    Attributes:
        min (float): The minimum value in the dataset.
        percentile_1 (float): The 1st percentile of the dataset.
        percentile_5 (float): The 5th percentile of the dataset.
        percentile_10 (float): The 10th percentile of the dataset.
        median (float): The median of the dataset.
        mean (float): The mean (average) of the dataset.
        percentile_90 (float): The 90th percentile of the dataset.
        percentile_95 (float): The 95th percentile of the dataset.
        percentile_99 (float): The 99th percentile of the dataset.
        max (float): The maximum value in the dataset.
        name_with_place_holder (str): A descriptive name for the dataset.
        data (list): The dataset to analyze provided as a list.
    """

    def __init__(self, name_with_place_holder, data):
        if type(data) is list:
            data = np.array(data)

        if data.size == 0:
            data = np.array([np.nan])

        self.min = np.min(data)
        self.percentile_1 = np.percentile(data, 1)
        self.percentile_5 = np.percentile(data, 5)
        self.percentile_10 = np.percentile(data, 10)
        self.median = np.median(data)
        self.mean = np.mean(data)
        self.percentile_90 = np.percentile(data, 90)
        self.percentile_95 = np.percentile(data, 95)
        self.percentile_99 = np.percentile(data, 99)
        self.max = np.max(data)
        self.name_with_place_holder = name_with_place_holder

    def get_filed_name(self, info_type):
        return self.name_with_place_holder.format(info_type=info_type)

    def get_as_dict(self):
        res = {
            self.get_filed_name("min"): [self.min],
            self.get_filed_name("percentile1"): [self.percentile_1],
            self.get_filed_name("percentile5"): [self.percentile_5],
            self.get_filed_name("percentile10"): [self.percentile_10],
            self.get_filed_name("median"): [self.median],
            self.get_filed_name("mean"): [self.mean],
            self.get_filed_name("percentile90"): [self.percentile_90],
            self.get_filed_name("percentile95"): [self.percentile_95],
            self.get_filed_name("percentile99"): [self.percentile_99],
            self.get_filed_name("max"): [self.max],
        }
        return res


def compute_results(audio_files_map, url, request_nb, max_stream_nb):
    """Make multiple requests for each audio file and measure the latency and fps in a pandas DataFrame.
    """
    list_result_dfs = []
    audio_data = [
        (sample_rate, audiolen, filepath)
        for sample_rate, file_len_dict in audio_files_map.items()
        for audiolen, filepath in file_len_dict.items()
    ]
    random.seed(45)
    # Generate a list of random times with a fixed seed.
    sleep_time_list = [random.random() for i in range(request_nb)]

    index = 0
    print("Starting clients...")
    for sample_rate, audio_length, filepath in audio_data:
        client_infos = [{'audio_file': filepath, 'url': url, 'sleep_time': sleep_time_list[i]} for i in
                        range(request_nb)]
        # Pack basic client info and simulate multiple clients.
        latency_list, fps_list, failed_requests = simulate_multiple_clients(max_stream_nb, client_infos)

        # Gather metrics for all streams.
        stats_holder_latencies = StatsHolder('1st packet {info_type} latency (ms)', latency_list)
        stats_holder_fps = StatsHolder('{info_type} fps', fps_list)

        successful_requests = request_nb - failed_requests

        # Create a DataFrame of results for current audio file
        result_df = pd.DataFrame({
            'max_stream_number': [max_stream_nb],
            'request_number': [request_nb],
            'audio_length (s)': [audio_length],
            'sample_rate': [sample_rate],
            **stats_holder_latencies.get_as_dict(),
            **stats_holder_fps.get_as_dict(),
            'successful_requests': [successful_requests]
        })
        # Append the DataFrame to the list.
        list_result_dfs.append(result_df)

        # Give time for the service to clean up states.
        time.sleep(3)
        index+=1
        print(f"\n{int(index/len(audio_data) * 100)}%")

    # Concatenate all DataFrames in the list.
    combined_df = pd.concat(list_result_dfs, ignore_index=True)
    return combined_df


def record_to_file(df, request_nb, summary_file):
    """ Save all results from the performance test to a csv file.
    """
    # Save all data to file
    logger.info(f'Printing the contents to {summary_file}')
    df.to_csv(summary_file, index=False, mode='a')
    logger.info(f'Results recorded for {request_nb} simulated clients in {summary_file}')


def record_fps_latency_summary_files(df, fps_file_name, latency_file_name, request_nb):
    """ Aggregate and write results from a pandas DataFrame to file.
    """
    percentile_1_fps = df['percentile1 fps'].min()
    fps_file = open(fps_file_name, 'w')
    fps_file.write(f'percentile1 FPS: {percentile_1_fps}\n')
    fps_file.close()
    logger.info(f'FPS results recorded for {request_nb} simulated clients in {fps_file_name}')

    max_latency = df['1st packet percentile99 latency (ms)'].max()
    avg_mean_latency = df['1st packet mean latency (ms)'].mean()
    latency_file = open(latency_file_name, 'w')
    latency_file.write(f'Worst case scenario: {max_latency} ms\n')
    latency_file.write(f'99% of the requests have latency below {max_latency} ms\n')
    latency_file.write(f'Average scenario: {avg_mean_latency} ms\n')
    latency_file.write(f'On average requests have latency below {avg_mean_latency} ms\n')
    logger.info(f'Latency results recorded for {request_nb} simulated clients in {latency_file_name}')


def simulate_multiple_clients(stream_nb, client_info_list):
    """ Start multiple processes to simulate multiple clients
    and collect performance metrics.
    """

    # Gather latency and fps metrics in one tuple per client.
    # Each tuple is in the form of (latency_ms, fps).
    with Pool(stream_nb) as pool:
        results = pool.map(run_single_client, client_info_list)

    # Prepare lists in which to unwrap the results.
    latency_list = []
    fps_list = []
    failed_requests = 0

    # Unwrap the tuples in results and count the number of failed requests.
    for latency_ms, fps in results:
        if latency_ms == -1 or fps == -1:
            failed_requests += 1
        else:
            latency_list.append(latency_ms)
            fps_list.append(fps)

    return latency_list, fps_list, failed_requests


def main():
    args = create_parser().parse_args()

    # Define audio files to use by their length in seconds.
    audio_files_16khz = {5: '../../example_audio/Claire_sadness_16khz_5_sec.wav',
                         10: '../../example_audio/Claire_sadness_16khz_10_sec.wav',
                         20: '../../example_audio/Claire_sadness_16khz_20_sec.wav', }
    audio_files_44_1_khz = {5: '../../example_audio/Claire_sadness_44.1khz_5_sec.wav',
                            10: '../../example_audio/Claire_sadness_44.1khz_10_sec.wav',
                            20: '../../example_audio/Claire_sadness_44.1khz_20_sec.wav', }
    audio_files = {
        "16khz": audio_files_16khz,
        "44.1khz": audio_files_44_1_khz,
    }

    # Prep the output folder with versioning. 
    output_folder_name = OUTPUT_FOLDER
    index = 1
    output_folder_name = f"{OUTPUT_FOLDER}_{index:06}"
    while os.path.exists(output_folder_name):
        index+=1
        output_folder_name = f"{OUTPUT_FOLDER}_{index:06}"
    os.makedirs(output_folder_name)

    summary_file = os.path.join(output_folder_name, f'stream_{args.max_stream_nb}_request_{args.request_nb}.csv')
    plot_file = os.path.join(output_folder_name, f'stream_{args.max_stream_nb}_request_{args.request_nb}.png')
    fps_summary_file = os.path.join(output_folder_name, f'fps_stream_{args.max_stream_nb}_request_{args.request_nb}.txt')
    latency_summary_file = os.path.join(output_folder_name, f'latency_stream_{args.max_stream_nb}_request_{args.request_nb}.txt')

    result = compute_results(audio_files, args.url, args.request_nb, args.max_stream_nb)
    print(f"Saving results to {output_folder_name} folder...")
    record_to_file(result, args.request_nb, summary_file)
    plot_metrics(result, plot_file)
    record_fps_latency_summary_files(result, fps_summary_file, latency_summary_file, args.request_nb)

if __name__ == '__main__':
    main()
