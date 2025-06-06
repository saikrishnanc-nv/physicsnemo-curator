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

from unittest.mock import Mock, patch

import pytest

from physicsnemo_curator.etl.data_processors import ParallelProcessor
from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class MockDataSource(DataSource):
    """Mock implementation of DataSource for testing ParallelProcessor."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.files = ["file1.txt", "file2.txt"]
        self.data = {"key": "value"}
        self.written_data = {}
        self.skipped_files = set()

    def get_file_list(self):
        return self.files

    def read_file(self, filename: str):
        if filename not in self.files:
            raise FileNotFoundError(f"File not found: {filename}")
        return self.data.copy()  # Return a copy to prevent mutations

    def write(self, data, filename: str):
        self.written_data[filename] = data

    def should_skip(self, filename: str) -> bool:
        """Checks whether the file should be skipped."""
        return filename in self.skipped_files


class MockTransformation(DataTransformation):
    """Mock implementation of DataTransformation for testing ParallelProcessor."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.transform_called = False

    def transform(self, data):
        self.transform_called = True
        data["transformed"] = True
        return data


@pytest.fixture
def processor_setup():
    """Setup a basic processor with mocked components."""
    config = ProcessingConfig(num_processes=2)
    source = MockDataSource(config)
    transforms = {"mock": MockTransformation(config)}
    sink = MockDataSource(config)

    return {"config": config, "source": source, "transform": transforms, "sink": sink}


def test_processor_initialization(processor_setup):
    """Test that processor initializes correctly."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    assert processor.source == processor_setup["source"]
    assert processor.transformations == processor_setup["transform"]
    assert processor.sink == processor_setup["sink"]
    assert processor.config == processor_setup["config"]
    assert processor.logger.name == "ParallelProcessor"
    assert processor.progress_counter.value == 0
    assert processor.total_files == 0


def test_process_files(processor_setup):
    """Test processing of a single file subset."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Process a subset of files
    file_subset = ["file1.txt"]
    processor.process_files(file_subset, worker_id=0)

    # Verify the pipeline was executed correctly
    assert processor_setup["sink"].written_data  # Check that data was written
    assert (
        "transformed" in processor_setup["sink"].written_data["file1.txt"]
    )  # Check transformation
    assert processor.progress_counter.value == 1  # Check progress counter


def test_process_files_with_error(processor_setup):
    """Test handling of errors during file processing."""
    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Make source.read_file raise an exception
    processor_setup["source"].read_file = Mock(side_effect=Exception("Test error"))

    # Process should log the error but not raise it
    with patch.object(processor.logger, "error") as mock_logger:
        processor.process_files(["file1.txt"], worker_id=0)
        mock_logger.assert_called_once()  # Verify error was logged

    # Verify no data was written
    assert not processor_setup["sink"].written_data


@patch("multiprocessing.get_context")
@patch("physicsnemo_curator.etl.data_processors.tqdm")
def test_run_parallel_processing(mock_tqdm, mock_get_context, processor_setup):
    """Test the parallel processing execution."""
    # Mock the multiprocessing context
    mock_process = Mock()
    mock_process.is_alive.side_effect = [
        True,
        True,
        False,
        False,
    ]  # Simulate process completion
    mock_context = Mock()
    mock_context.Process.return_value = mock_process
    mock_get_context.return_value = mock_context

    # Mock tqdm progress bar
    mock_pbar = Mock()
    mock_tqdm.return_value = mock_pbar

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Run the processor
    processor.run()

    # Verify processes were started and joined
    assert mock_context.Process.called
    assert mock_process.start.called
    assert mock_process.join.called

    # Verify progress bar was created and updated
    mock_tqdm.assert_called_once_with(
        total=len(processor_setup["source"].files), desc="Processing files", unit="file"
    )
    assert mock_pbar.refresh.called
    assert mock_pbar.close.called


def test_run_with_no_files(processor_setup):
    """Test handling of empty file list."""
    processor_setup["source"].files = []  # Set empty file list

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    # Should complete without error
    processor.run()
    assert processor.progress_counter.value == 0  # No progress for empty file list


@patch("multiprocessing.get_context")
@patch("physicsnemo_curator.etl.data_processors.tqdm")
def test_file_distribution(mock_tqdm, mock_get_context, processor_setup):
    """Test that files are distributed correctly among workers."""
    # Create a processor with 2 workers and 3 files
    processor_setup["config"] = ProcessingConfig(num_processes=2)
    processor_setup["source"].files = ["file1.txt", "file2.txt", "file3.txt"]

    # Mock the multiprocessing context
    mock_process = Mock()
    mock_process.is_alive.side_effect = [
        True,
        True,
        False,
        False,
    ]  # Simulate process completion
    mock_context = Mock()
    mock_context.Process.return_value = mock_process
    mock_get_context.return_value = mock_context

    # Mock tqdm progress bar
    mock_pbar = Mock()
    mock_tqdm.return_value = mock_pbar

    processor = ParallelProcessor(
        processor_setup["source"],
        processor_setup["transform"],
        processor_setup["sink"],
        processor_setup["config"],
    )

    processor.run()

    # Verify process creation calls
    calls = mock_context.Process.call_args_list
    assert len(calls) == 2  # Should create 2 processes

    # First process should get 2 files, second process should get 1 file
    first_call_files = calls[0][1]["args"][0]
    second_call_files = calls[1][1]["args"][0]
    assert len(first_call_files) == 2
    assert len(second_call_files) == 1
