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


import pytest

from physicsnemo_curator.etl.processing_config import ProcessingConfig


def test_valid_minimal_config():
    """Test creation of ProcessingConfig with minimal valid parameters."""
    config = ProcessingConfig(num_processes=4)
    assert config.num_processes == 4
    assert config.serialization_method == "numpy"
    assert config.compression is None


def test_valid_zarr_config():
    """Test creation of ProcessingConfig with zarr and compression."""
    config = ProcessingConfig(
        num_processes=2,
        serialization_method="zarr",
        compression={"method": "zstd", "level": 3},
    )
    assert config.num_processes == 2
    assert config.serialization_method == "zarr"
    assert config.compression == {"method": "zstd", "level": 3}


def test_invalid_num_processes_raises_value_error():
    """Test that invalid num_processes raises ValueError."""
    with pytest.raises(ValueError, match="num_processes must be positive"):
        ProcessingConfig(num_processes=0)

    with pytest.raises(ValueError, match="num_processes must be positive"):
        ProcessingConfig(num_processes=-1)


def test_invalid_compression_with_numpy_raises_value_error():
    """Test that compression with numpy serialization raises ValueError."""
    with pytest.raises(ValueError, match="Compression is only supported with zarr"):
        ProcessingConfig(
            num_processes=1,
            serialization_method="numpy",
            compression={"method": "zstd", "level": 3},
        )


def test_invalid_compression_config_raises_value_error():
    """Test that invalid compression configuration raises ValueError."""
    # Missing required keys
    with pytest.raises(ValueError, match="missing required keys"):
        ProcessingConfig(
            num_processes=1,
            serialization_method="zarr",
            compression={"method": "zstd"},  # missing level
        )

    # Invalid compression level type
    with pytest.raises(ValueError, match="must be an integer between 1 and 9"):
        ProcessingConfig(
            num_processes=1,
            serialization_method="zarr",
            compression={"method": "zstd", "level": "3"},  # string instead of int
        )

    # Compression level out of range
    with pytest.raises(ValueError, match="must be an integer between 1 and 9"):
        ProcessingConfig(
            num_processes=1,
            serialization_method="zarr",
            compression={"method": "zstd", "level": 10},
        )


def test_invalid_serialization_method_raises_value_error():
    """Test that invalid serialization method raises ValueError."""
    with pytest.raises(
        ValueError, match="Serialization method must be either 'numpy' or 'zarr'"
    ):
        ProcessingConfig(
            num_processes=1,
            serialization_method="invalid",
        )
