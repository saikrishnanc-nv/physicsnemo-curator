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

from typing import Any, Dict, List

import pytest

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig


class MockDataSource(DataSource):
    """Mock implementation of DataSource for testing."""

    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.files = ["file1.txt", "file2.txt"]
        self.data = {"key": "value"}
        self.written_data = {}
        self.skipped_files = set()

    def get_file_list(self) -> List[str]:
        return self.files

    def read_file(self, filename: str) -> Dict[str, Any]:
        if filename not in self.files:
            raise FileNotFoundError(f"File not found: {filename}")
        return self.data

    def write(self, data: Dict[str, Any], filename: str) -> None:
        self.written_data[filename] = data

    def should_skip(self, filename: str) -> bool:
        """Checks whether the file should be skipped."""
        return filename in self.skipped_files


def test_datasource_direct_instantiation_raises_error():
    """Test that DataSource cannot be instantiated directly."""
    config = ProcessingConfig(num_processes=1)
    with pytest.raises(TypeError):
        DataSource(config)


def test_mock_datasource_instantiation():
    """Test that concrete implementation can be instantiated."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    assert isinstance(source, DataSource)
    assert source.config == config
    assert source.logger.name == "MockDataSource"


def test_get_file_list():
    """Test get_file_list returns expected files."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    files = source.get_file_list()
    assert files == ["file1.txt", "file2.txt"]


def test_read_file():
    """Test read_file returns expected data."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    data = source.read_file("file1.txt")
    assert data == {"key": "value"}

    with pytest.raises(FileNotFoundError):
        source.read_file("nonexistent.txt")


def test_write():
    """Test write stores data correctly."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    test_data = {"test": "data"}
    source.write(test_data, "output.txt")
    assert source.written_data["output.txt"] == test_data


def test_logger_creation():
    """Test that logger is created with correct name."""
    config = ProcessingConfig(num_processes=1)
    source = MockDataSource(config)
    assert source.logger.name == "MockDataSource"
