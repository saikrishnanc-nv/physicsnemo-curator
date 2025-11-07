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

import tempfile
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import zarr

from examples.structural_mechanics.crash.data_sources import (
    CrashD3PlotDataSource,
    CrashVTPDataSource,
    CrashZarrDataSource,
)
from examples.structural_mechanics.crash.schemas import (
    CrashExtractedDataInMemory,
    CrashMetadata,
)


@pytest.fixture
def mock_crash_data():
    """Create mock crash simulation data for testing."""
    # Positions over time (timesteps, nodes, 3)
    pos_raw = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # t0
            [[0, 0, 0], [1.1, 0, 0], [0, 1, 0], [1.1, 1, 0]],  # t1
            [[0, 0, 0], [1.2, 0, 0], [0, 1, 0], [1.2, 1, 0]],  # t2
        ],
        dtype=np.float32,
    )

    # Mesh connectivity (one quad)
    mesh_connectivity = [[0, 1, 3, 2]]

    # Node thickness
    node_thickness = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Edges
    edges = {(0, 1), (1, 3), (2, 3), (0, 2)}

    return CrashExtractedDataInMemory(
        metadata=CrashMetadata(filename="test_run"),
        pos_raw=pos_raw,
        mesh_connectivity=mesh_connectivity,
        node_thickness=node_thickness,
        filtered_pos_raw=pos_raw,
        filtered_mesh_connectivity=mesh_connectivity,
        filtered_node_thickness=node_thickness,
        edges=edges,
    )


@pytest.fixture
def temp_k_file():
    """Create a temporary .k file for testing."""
    k_content = """$# LS-DYNA Keyword file
*PART
Part 1
1   1   1
*SECTION_SHELL
1
2.5
*PART
Part 2
2   2   1
*SECTION_SHELL
2
1.5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".k", delete=False) as f:
        f.write(k_content)
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


# Tests for CrashD3PlotDataSource static methods


def test_parse_k_file(temp_k_file):
    """Test parsing of .k file for thickness information."""
    part_thickness_map = CrashD3PlotDataSource.parse_k_file(temp_k_file)

    # Should have 2 parts
    assert (
        len(part_thickness_map) == 2
    ), f"Expected 2 parts, got {len(part_thickness_map)}"

    # Check thickness values
    assert (
        part_thickness_map[1] == 2.5
    ), f"Expected part 1 thickness 2.5, got {part_thickness_map[1]}"
    assert (
        part_thickness_map[2] == 1.5
    ), f"Expected part 2 thickness 1.5, got {part_thickness_map[2]}"


def test_compute_node_thickness_basic():
    """Test node thickness computation from element thickness."""
    # Simple mesh: 2 elements, 6 nodes (sharing 2 nodes)
    mesh_connectivity = [
        [0, 1, 2, 3],  # Element 0 (quad)
        [1, 4, 5, 2],  # Element 1 (quad, shares nodes 1 and 2)
    ]

    part_ids = np.array([1, 2])  # Element 0 is part 1, element 1 is part 2
    part_thickness_map = {1: 2.0, 2: 1.0}
    actual_part_ids = np.array([0, 1, 2])  # Index 0 unused, 1->part1, 2->part2

    node_thickness = CrashD3PlotDataSource.compute_node_thickness(
        mesh_connectivity, part_ids, part_thickness_map, actual_part_ids
    )

    # Node 0 and 3 only in element 0: thickness = 2.0
    assert (
        node_thickness[0] == 2.0
    ), f"Expected thickness 2.0 for node 0, got {node_thickness[0]}"
    assert (
        node_thickness[3] == 2.0
    ), f"Expected thickness 2.0 for node 3, got {node_thickness[3]}"

    # Nodes 1 and 2 in both elements: thickness = (2.0 + 1.0) / 2 = 1.5
    assert (
        node_thickness[1] == 1.5
    ), f"Expected thickness 1.5 for node 1, got {node_thickness[1]}"
    assert (
        node_thickness[2] == 1.5
    ), f"Expected thickness 1.5 for node 2, got {node_thickness[2]}"

    # Nodes 4 and 5 only in element 1: thickness = 1.0
    assert (
        node_thickness[4] == 1.0
    ), f"Expected thickness 1.0 for node 4, got {node_thickness[4]}"
    assert (
        node_thickness[5] == 1.0
    ), f"Expected thickness 1.0 for node 5, got {node_thickness[5]}"


def test_compute_node_thickness_no_actual_ids():
    """Test node thickness computation without actual_part_ids."""
    mesh_connectivity = [[0, 1, 2]]
    part_ids = np.array([1])
    part_thickness_map = {10: 3.0}  # Part ID 10 (will be mapped to index 1)

    node_thickness = CrashD3PlotDataSource.compute_node_thickness(
        mesh_connectivity, part_ids, part_thickness_map, actual_part_ids=None
    )

    # All nodes should have thickness 3.0
    assert node_thickness[0] == 3.0
    assert node_thickness[1] == 3.0
    assert node_thickness[2] == 3.0


# Tests for CrashVTPDataSource


def test_vtp_sink_output_path():
    """Test VTP output path generation."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashVTPDataSource(
            cfg, temp_dir, overwrite_existing=True, time_step=0.005
        )

        output_path = sink._get_output_path("Run100")
        assert output_path == Path(temp_dir) / "Run100.vtp"


def test_vtp_sink_write(mock_crash_data):
    """Test VTP file writing."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashVTPDataSource(
            cfg, temp_dir, overwrite_existing=True, time_step=0.005
        )

        # Write the data
        sink.write(mock_crash_data, "test_run")

        # Check file was created
        output_file = Path(temp_dir) / "test_run.vtp"
        assert output_file.exists(), "VTP file was not created"

        # Load and verify the VTP file
        mesh = pv.read(str(output_file))

        # Check points (should be reference coordinates)
        assert mesh.n_points == 4, f"Expected 4 points, got {mesh.n_points}"

        # Check point data contains thickness
        assert "thickness" in mesh.point_data.keys(), "Thickness not found in VTP"
        np.testing.assert_array_almost_equal(
            mesh.point_data["thickness"], mock_crash_data.node_thickness
        )

        # Check displacement fields
        assert (
            "displacement_t0.000" in mesh.point_data.keys()
        ), "displacement_t0.000 not found"
        assert (
            "displacement_t0.005" in mesh.point_data.keys()
        ), "displacement_t0.005 not found"
        assert (
            "displacement_t0.010" in mesh.point_data.keys()
        ), "displacement_t0.010 not found"

        # First timestep should have zero displacement
        disp_t0 = mesh.point_data["displacement_t0.000"]
        np.testing.assert_array_almost_equal(disp_t0, np.zeros((4, 3)))


def test_vtp_sink_should_skip():
    """Test VTP sink skip logic."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy file
        dummy_file = Path(temp_dir) / "existing_run.vtp"
        dummy_file.touch()

        # Test with overwrite_existing=True (should not skip)
        sink = CrashVTPDataSource(cfg, temp_dir, overwrite_existing=True)
        assert not sink.should_skip(
            "existing_run"
        ), "Should not skip when overwrite_existing=True"

        # Test with overwrite_existing=False (should skip)
        sink = CrashVTPDataSource(cfg, temp_dir, overwrite_existing=False)
        assert sink.should_skip(
            "existing_run"
        ), "Should skip when file exists and overwrite_existing=False"

        # Test with non-existent file
        assert not sink.should_skip(
            "non_existent_run"
        ), "Should not skip non-existent file"


# Tests for CrashZarrDataSource


def test_zarr_sink_output_path():
    """Test Zarr output path generation."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=True)

        output_path = sink._get_output_path("Run100")
        assert output_path == Path(temp_dir) / "Run100.zarr"


def test_zarr_sink_write(mock_crash_data):
    """Test Zarr store writing."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(
            cfg, temp_dir, overwrite_existing=True, compression_level=3
        )

        # Write the data
        sink.write(mock_crash_data, "test_run")

        # Check store was created
        output_store = Path(temp_dir) / "test_run.zarr"
        assert output_store.exists(), "Zarr store was not created"
        assert output_store.is_dir(), "Zarr store should be a directory"

        # Open and verify the Zarr store
        store = zarr.open(str(output_store), mode="r")

        # Check arrays exist
        assert "mesh_pos" in store, "mesh_pos not in Zarr store"
        assert "node_thickness" in store, "node_thickness not in Zarr store"
        assert "edges" in store, "edges not in Zarr store"

        # Check shapes
        assert store["mesh_pos"].shape == (
            3,
            4,
            3,
        ), f"Expected mesh_pos shape (3, 4, 3), got {store['mesh_pos'].shape}"
        assert store["node_thickness"].shape == (
            4,
        ), f"Expected node_thickness shape (4,), got {store['node_thickness'].shape}"

        # Check metadata
        assert "filename" in store.attrs, "filename not in metadata"
        assert store.attrs["filename"] == "test_run"
        assert store.attrs["num_timesteps"] == 3
        assert store.attrs["num_nodes"] == 4


def test_zarr_sink_should_skip():
    """Test Zarr sink skip logic."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy zarr directory
        dummy_store = Path(temp_dir) / "existing_run.zarr"
        dummy_store.mkdir()

        # Test with overwrite_existing=True (should not skip)
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=True)
        assert not sink.should_skip(
            "existing_run"
        ), "Should not skip when overwrite_existing=True"

        # Test with overwrite_existing=False (should skip)
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=False)
        assert sink.should_skip(
            "existing_run"
        ), "Should skip when store exists and overwrite_existing=False"

        # Test with non-existent store
        assert not sink.should_skip(
            "non_existent_run"
        ), "Should not skip non-existent store"


def test_zarr_sink_metadata_statistics(mock_crash_data):
    """Test that Zarr store includes thickness statistics in metadata."""
    from physicsnemo_curator.etl.processing_config import ProcessingConfig

    cfg = ProcessingConfig(num_processes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        sink = CrashZarrDataSource(cfg, temp_dir, overwrite_existing=True)
        sink.write(mock_crash_data, "test_run")

        store = zarr.open(str(Path(temp_dir) / "test_run.zarr"), mode="r")

        # Check statistics
        assert "thickness_min" in store.attrs
        assert "thickness_max" in store.attrs
        assert "thickness_mean" in store.attrs

        assert store.attrs["thickness_min"] == 1.0
        assert store.attrs["thickness_max"] == 1.0
        assert store.attrs["thickness_mean"] == 1.0
