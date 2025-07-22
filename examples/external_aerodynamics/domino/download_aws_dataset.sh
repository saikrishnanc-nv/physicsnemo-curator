#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This Bash script downloads the AWS DrivAer files from the Amazon S3 bucket to a local directory.
# Only the volume files (.vtu), STL files (.stl), and VTP files (.vtp) are downloaded.
# It uses a function, download_run_files, to check for the existence of three specific files (".vtu", ".stl", ".vtp") in a run directory.
# If a file doesn't exist, it's downloaded from the S3 bucket. If it does exist, the download is skipped.
# The script runs multiple downloads in parallel, both within a single run and across multiple runs.
# It also includes checks to prevent overloading the system by limiting the number of parallel downloads.

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --local-dir DIR     Local directory to download files (default: ./drivaer_data_full)"
    echo "  -s, --run-start NUM     Starting run number (default: 1)"
    echo "  -e, --run-end NUM       Ending run number (default: 5, max: 500)"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d ./my_data -s 1 -e 10"
    exit 1
}

# Default values
LOCAL_DIR="./drivaer_data_full"
RUN_START=1
RUN_END=5

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        -s|--run-start)
            RUN_START="$2"
            shift 2
            ;;
        -e|--run-end)
            RUN_END="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate arguments
if ! [[ "$RUN_START" =~ ^[0-9]+$ ]] || ! [[ "$RUN_END" =~ ^[0-9]+$ ]]; then
    echo "Error: run_start and run_end must be positive integers"
    exit 1
fi

if [ "$RUN_START" -gt "$RUN_END" ]; then
    echo "Error: run_start cannot be greater than run_end"
    exit 1
fi

if [ "$RUN_END" -gt 500 ]; then
    echo "Error: run_end cannot be greater than 500 (maximum available runs)"
    exit 1
fi

# Set the S3 bucket and prefix
S3_BUCKET="caemldatasets"
S3_PREFIX="drivaer/dataset"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Function to download files for a specific run
download_run_files() {
    local i=$1
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"

    # Create the run directory if it doesn't exist
    mkdir -p "$RUN_LOCAL_DIR"

    # Check if the .vtu file exists before downloading
    if [ ! -f "$RUN_LOCAL_DIR/volume_$i.vtu" ]; then
        aws s3 cp --no-sign-request "s3://$S3_BUCKET/$S3_PREFIX/$RUN_DIR/volume_$i.vtu" "$RUN_LOCAL_DIR/" &
    else
        echo "File volume_$i.vtu already exists, skipping download."
    fi

    # Check if the .stl file exists before downloading
    if [ ! -f "$RUN_LOCAL_DIR/drivaer_$i.stl" ]; then
        aws s3 cp --no-sign-request "s3://$S3_BUCKET/$S3_PREFIX/$RUN_DIR/drivaer_$i.stl" "$RUN_LOCAL_DIR/" &
    else
        echo "File drivaer_$i.stl already exists, skipping download."
    fi

    # Check if the .vtp file exists before downloading
    if [ ! -f "$RUN_LOCAL_DIR/boundary_$i.vtp" ]; then
        aws s3 cp --no-sign-request "s3://$S3_BUCKET/$S3_PREFIX/$RUN_DIR/boundary_$i.vtp" "$RUN_LOCAL_DIR/" &
    else
        echo "File boundary_$i.vtp already exists, skipping download."
    fi

    wait # Ensure that both files for this run are downloaded before moving to the next run
}

echo "Starting download from run $RUN_START to run $RUN_END to directory: $LOCAL_DIR"

# Loop through the run folders and download the files
for i in $(seq "$RUN_START" "$RUN_END"); do
    download_run_files "$i" &

    # Limit the number of parallel jobs to avoid overloading the system
    if (( $(jobs -r | wc -l) >= 8 )); then
        wait -n # Wait for the next background job to finish before starting a new one
    fi
done

# Wait for all remaining background jobs to finish
wait

echo "Download completed for runs $RUN_START to $RUN_END"
