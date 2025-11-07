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

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyvista as pv
import zarr
from lasso.dyna import ArrayType, D3plot
from numcodecs import Blosc

from physicsnemo_curator.etl.data_sources import DataSource
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .schemas import CrashExtractedDataInMemory, CrashMetadata


class CrashD3PlotDataSource(DataSource):
    """Data source for reading LS-DYNA d3plot simulation files."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        input_dir: str,
    ):
        super().__init__(cfg)
        self.input_dir = Path(input_dir)
        logging.basicConfig(
            format="%(asctime)s - Process %(process)d - %(levelname)s - %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

    def get_file_list(self) -> List[str]:
        """Find all run folders containing d3plot files."""
        run_folders = []
        for item in self.input_dir.iterdir():
            if item.is_dir():
                d3plot_path = item / "d3plot"
                if d3plot_path.exists():
                    run_folders.append(item.name)

        self.logger.info(f"Found {len(run_folders)} d3plot runs to process")
        return sorted(run_folders)

    def read_file(self, run_id: str) -> CrashExtractedDataInMemory:
        """Read d3plot and .k file for one simulation run.

        Returns CrashExtractedDataInMemory object.
        """
        run_dir = self.input_dir / run_id
        d3plot_path = run_dir / "d3plot"

        self.logger.info(f"Reading {d3plot_path}")

        coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids = (
            self.load_d3plot_data(str(d3plot_path))
        )

        # Parse .k file for thickness
        k_file_path = self.find_k_file(run_dir=run_dir)
        node_thickness = np.zeros(len(coords))
        if k_file_path:
            self.logger.info(f"Parsing thickness from {k_file_path}")
            part_thickness_map = self.parse_k_file(k_file_path=k_file_path)
            node_thickness = self.compute_node_thickness(
                mesh_connectivity=mesh_connectivity,
                part_ids=part_ids,
                part_thickness_map=part_thickness_map,
                actual_part_ids=actual_part_ids,
            )
        else:
            self.logger.warning(
                f"No .k file found in {run_dir}, defaulting thickness=0"
            )

        return CrashExtractedDataInMemory(
            metadata=CrashMetadata(
                filename=run_id,
            ),
            pos_raw=pos_raw,
            mesh_connectivity=mesh_connectivity,
            node_thickness=node_thickness,
        )

    @staticmethod
    def load_d3plot_data(data_path: str):
        """Load node coordinates and displacements from a d3plot file."""
        dp = D3plot(data_path)
        coords = dp.arrays[ArrayType.node_coordinates]  # (num_nodes, 3)
        pos_raw = dp.arrays[ArrayType.node_displacement]  # (timesteps, num_nodes, 3)
        mesh_connectivity = dp.arrays[ArrayType.element_shell_node_indexes]
        part_ids = dp.arrays[ArrayType.element_shell_part_indexes]

        # Get actual part IDs if available
        actual_part_ids = None
        if ArrayType.part_ids in dp.arrays:
            actual_part_ids = dp.arrays[ArrayType.part_ids]

        return coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids

    @staticmethod
    def find_k_file(run_dir: Path):
        """Find .k file in run directory."""
        k_files = list(run_dir.glob("*.k"))
        return k_files[0] if k_files else None

    @staticmethod
    def parse_k_file(k_file_path: Path):
        """Parse LS-DYNA .k file to extract part thickness information."""
        part_to_section = {}
        section_thickness = {}

        with open(k_file_path, "r") as f:
            lines = [
                line.strip() for line in f if line.strip() and not line.startswith("$")
            ]

        i = 0
        while i < len(lines):
            line = lines[i]
            if "*PART" in line.upper():
                # After *PART:
                # i+1 = part name (skip)
                # i+2 = part id, section id, material id
                if i + 2 < len(lines):
                    tokens = lines[i + 2].split()
                    if len(tokens) >= 2:
                        part_id = int(tokens[0])
                        section_id = int(tokens[1])
                        part_to_section[part_id] = section_id
                i += 3
            elif "*SECTION_SHELL" in line.upper():
                # Multiple sections can be defined under one *SECTION_SHELL keyword
                # Each section has two lines: header line and thickness line
                i += 1  # Skip the *SECTION_SHELL line
                while i < len(lines) and not lines[i].startswith("*"):
                    # Check if this line looks like a section header (starts with a number)
                    if i < len(lines) and lines[i].strip() and lines[i][0].isdigit():
                        header_line = lines[i]
                        thickness_line = lines[i + 1] if i + 1 < len(lines) else ""

                        # Extract section ID from header line (first number)
                        header_tokens = header_line.split()
                        if len(header_tokens) >= 1:
                            try:  # noqa: PERF203
                                section_id = int(header_tokens[0])
                            except ValueError:
                                section_id = None
                        else:
                            section_id = None

                        # Extract thickness values from thickness line
                        thickness_values = []
                        thickness_tokens = thickness_line.split()
                        for t in thickness_tokens:
                            try:
                                thickness_values.append(float(t))
                            except ValueError:  # noqa: PERF203
                                thickness_values.append(0.0)
                        # Calculate average thickness (ignore zeros)
                        non_zero_thicknesses = [t for t in thickness_values if t > 0.0]
                        if non_zero_thicknesses:
                            thickness = sum(non_zero_thicknesses) / len(
                                non_zero_thicknesses
                            )
                        elif thickness_values:
                            thickness = sum(thickness_values) / len(thickness_values)
                        else:
                            thickness = 0.0
                        if section_id is not None:
                            section_thickness[section_id] = thickness

                        i += 2  # Skip both header and thickness lines
                    else:
                        i += 1
            else:
                i += 1

        part_thickness = {
            pid: section_thickness.get(sid, 0.0) for pid, sid in part_to_section.items()
        }
        return part_thickness

    @staticmethod
    def compute_node_thickness(
        mesh_connectivity: np.ndarray,
        part_ids: np.ndarray,
        part_thickness_map: dict,
        actual_part_ids: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute thickness for each node based on elements connected to it.

        Args:
            mesh_connectivity: Element connectivity array (num_elements, num_nodes_per_element)
            part_ids: Part IDs for each element (num_elements)
            part_thickness_map: Mapping from part ID to thickness
            actual_part_ids: Actual part IDs if available (num_parts)

        Returns:
            node_thickness: Array of thickness values for each node (num_nodes)
        """
        # Create mapping from part index to actual part ID
        if actual_part_ids is not None:
            part_index_to_id = {
                i: actual_part_id
                for i, actual_part_id in enumerate(actual_part_ids)
                if i > 0  # Skip index 0
            }
        else:
            sorted_part_ids = sorted(part_thickness_map.keys())
            part_index_to_id = {
                i: part_id for i, part_id in enumerate(sorted_part_ids, 1)
            }

        # Get element thickness
        element_thickness = np.zeros(len(part_ids))
        for i, part_index in enumerate(part_ids):
            actual_part_id = part_index_to_id.get(part_index)
            if actual_part_id is not None:
                thickness = part_thickness_map.get(actual_part_id, 0.0)
                element_thickness[i] = thickness

        # Find maximum node index to initialize node thickness array
        max_node_idx = 0
        for element in mesh_connectivity:
            max_node_idx = max(max_node_idx, max(element))

        node_thickness = np.zeros(max_node_idx + 1)
        node_thickness_count = np.zeros(max_node_idx + 1)

        # Accumulate thickness from all elements connected to each node
        for i, element in enumerate(mesh_connectivity):
            thickness = element_thickness[i]
            for node_idx in element:
                node_thickness[node_idx] += thickness
                node_thickness_count[node_idx] += 1

        # Average thickness for nodes connected to multiple elements
        for i in range(len(node_thickness)):
            if node_thickness_count[i] > 0:
                node_thickness[i] /= node_thickness_count[i]

        return node_thickness

    def _get_output_path(self, filename: str) -> Path:
        """Not implemented - this source only reads."""
        raise NotImplementedError("CrashD3PlotDataSource only supports reading")

    def _write_impl_temp_file(self, data: Dict[str, Any], output_path: Path) -> None:
        """Not implemented - this source only reads."""
        raise NotImplementedError("CrashD3PlotDataSource only supports reading")

    def should_skip(self, filename: str) -> bool:
        """Never skip for reading."""
        return False

    def write(self, data: Any, filename: str) -> None:
        """Not implemented - this source only reads."""
        raise NotImplementedError("CrashD3PlotDataSource only supports reading")


class CrashVTPDataSource(DataSource):
    """Data source for writing crash simulation VTP files.

    Outputs one VTP file per run containing all timesteps as displacement fields.
    """

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        time_step: float = 0.005,
    ):
        """Initialize the VTP data source.

        Args:
            cfg: Processing configuration
            output_dir: Directory to write VTP files
            overwrite_existing: Whether to overwrite existing files
            time_step: Time step between frames (default: 0.005s)
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.time_step = time_step

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashVTPDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashVTPDataSource only supports writing")

    def _get_output_path(self, filename: str) -> Path:
        """Get the output file path for a given run.

        Args:
            filename: Run ID (e.g., "Run100")

        Returns:
            Path to the VTP file (e.g., Run100.vtp)
        """
        return self.output_dir / f"{filename}.vtp"

    def _get_temporary_output_path(self, final_path: Path) -> Path:
        """Get temporary path for a final output path.

        Args:
            final_path: The final destination path

        Returns:
            Temporary path with _temp suffix before the extension
        """
        vtp_path = final_path.with_suffix(".vtp")
        return vtp_path.with_name(f"{vtp_path.name}_temp.vtp")

    def _write_impl_temp_file(
        self, data: CrashExtractedDataInMemory, output_path: Path
    ) -> None:
        """Write a single VTP file containing all timesteps as displacement fields.

        This method receives a TEMPORARY path from the base class (e.g., Run100.vtp_temp).
        After writing completes, the base class atomically renames it to the final path.

        The VTP format expected by the reader:
        - poly.points = reference coordinates (t=0)
        - poly.point_data['displacement_t{time}'] = displacement at each timestep
        - poly.point_data['thickness'] = node thickness values

        Args:
            data: Transformed crash data containing filtered_pos_raw,
                  filtered_mesh_connectivity, filtered_node_thickness, and edges
            output_path: TEMPORARY file path where VTP should be written.
                        Base class will rename this to final path after writing completes.
        """
        self.logger.info(f"Writing VTP file to temporary location: {output_path.name}")

        n_timesteps = data.filtered_pos_raw.shape[0]

        # Use the first timestep as reference coordinates
        reference_coords = data.filtered_pos_raw[0, :, :]

        # Build mesh connectivity for PyVista
        faces = []
        for cell in data.filtered_mesh_connectivity:
            if len(cell) == 3:  # Triangle
                faces.extend([3, *cell])
            elif len(cell) == 4:  # Quad
                faces.extend([4, *cell])
            elif len(cell) > 4:  # Higher order elements - skip
                continue

        faces = np.array(faces)
        mesh = pv.PolyData(reference_coords, faces)

        # Add thickness as point data (must match number of points in mesh)
        n_points = len(reference_coords)
        if len(data.filtered_node_thickness) != n_points:
            self.logger.error(
                f"Thickness array size ({len(data.filtered_node_thickness)}) "
                f"does not match number of points ({n_points})"
            )
            return

        mesh.point_data["thickness"] = data.filtered_node_thickness

        # Add displacement fields for each timestep
        for t in range(n_timesteps):
            # Compute displacement from reference position
            displacement = data.filtered_pos_raw[t, :, :] - reference_coords

            # Format time value with 3 decimal places (e.g., t0.000, t0.005, t0.010)
            time_value = t * self.time_step
            time_str = f"t{time_value:.3f}"
            field_name = f"displacement_{time_str}"

            mesh.point_data[field_name] = displacement

        # Save the single VTP file
        mesh.save(output_path)

        self.logger.info(
            f"Wrote VTP file with {n_timesteps} timesteps for {output_path.stem}"
        )

    def should_skip(self, run_id: str) -> bool:
        """Check if output already exists.

        Args:
            run_id: Run ID to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.overwrite_existing:
            return False

        output_path = self._get_output_path(run_id)
        if output_path.exists():
            self.logger.info(f"Skipping {run_id} - VTP file already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary VTP files from interrupted runs."""
        if not self.output_dir or not self.output_dir.exists():
            return

        # Find all temp VTP files (*._temp.vtp)
        for temp_file in self.output_dir.glob("*_temp.vtp"):
            self.logger.warning(f"Removing orphaned temp VTP file: {temp_file}")
            temp_file.unlink()


class CrashZarrDataSource(DataSource):
    """Data source for writing crash simulation data to Zarr format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        output_dir: str,
        overwrite_existing: bool = True,
        compression_level: int = 3,
    ):
        """Initialize the Zarr data source.

        Args:
            cfg: Processing configuration
            output_dir: Directory to write Zarr stores
            overwrite_existing: Whether to overwrite existing files
            compression_level: Compression level (1-9, higher = more compression)
        """
        super().__init__(cfg)
        self.output_dir = Path(output_dir)
        self.overwrite_existing = overwrite_existing
        self.compression_level = compression_level

        # Set up compressor
        self.compressor = Blosc(
            cname="zstd", clevel=compression_level, shuffle=Blosc.SHUFFLE
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list(self) -> List[str]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashZarrDataSource only supports writing")

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Not implemented - this sink only writes."""
        raise NotImplementedError("CrashZarrDataSource only supports writing")

    def _get_output_path(self, filename: str) -> Path:
        """Get the output path for the Zarr store.

        Args:
            filename: Run ID (e.g., "run_001")

        Returns:
            Path to the Zarr store (e.g., run_001.zarr)
        """
        return self.output_dir / f"{filename}.zarr"

    def _write_impl_temp_file(
        self, data: CrashExtractedDataInMemory, output_path: Path
    ) -> None:
        """Write crash data to a Zarr store.

        This method receives a TEMPORARY path from the base class (e.g., run_001.zarr_temp).
        After writing completes, the base class atomically renames it to the final path.

        Args:
            data: Transformed crash data containing:
                - filtered_pos_raw: (timesteps, nodes, 3) temporal positions
                - filtered_mesh_connectivity: list of cell connectivities
                - filtered_node_thickness: (nodes,) thickness values
                - edges: (num_edges, 2) edge connectivity
            output_path: TEMPORARY directory path for the Zarr store.
                        Base class will rename this to final path after writing completes.
        """
        self.logger.info(
            f"Creating Zarr store at temporary location: {output_path.name}"
        )

        # Create Zarr store
        zarr_store = zarr.DirectoryStore(output_path)
        root = zarr.group(store=zarr_store)

        # Write metadata as root attributes
        root.attrs["filename"] = data.metadata.filename
        root.attrs["num_timesteps"] = data.filtered_pos_raw.shape[0]
        root.attrs["num_nodes"] = data.filtered_pos_raw.shape[1]
        root.attrs["num_edges"] = len(data.edges)
        root.attrs["compression"] = "zstd"
        root.attrs["compression_level"] = self.compression_level

        # Calculate optimal chunks for temporal data
        num_timesteps, num_nodes, _ = data.filtered_pos_raw.shape
        chunk_timesteps = min(10, num_timesteps)  # Chunk along time dimension
        chunk_nodes = min(1000, num_nodes)  # Chunk along node dimension

        # Write temporal position data
        root.create_dataset(
            "mesh_pos",
            data=data.filtered_pos_raw.astype(np.float32),
            chunks=(chunk_timesteps, chunk_nodes, 3),
            compressor=self.compressor,
            dtype=np.float32,
        )

        # Write node thickness (static per node)
        root.create_dataset(
            "node_thickness",
            data=data.filtered_node_thickness.astype(np.float32),
            chunks=(chunk_nodes,),
            compressor=self.compressor,
            dtype=np.float32,
        )

        # Write edges connectivity
        edges_array = np.array(list(data.edges), dtype=np.int64)
        root.create_dataset(
            "edges",
            data=edges_array,
            chunks=(min(10000, len(edges_array)), 2),
            compressor=self.compressor,
            dtype=np.int64,
        )

        # Write mesh connectivity (convert list of lists to ragged array representation)
        # Store as: flat array of node indices + offsets array
        flat_connectivity = []
        offsets = [0]
        for cell in data.filtered_mesh_connectivity:
            flat_connectivity.extend(cell)
            offsets.append(len(flat_connectivity))

        root.create_dataset(
            "mesh_connectivity_flat",
            data=np.array(flat_connectivity, dtype=np.int64),
            chunks=(min(10000, len(flat_connectivity)),),
            compressor=self.compressor,
            dtype=np.int64,
        )

        root.create_dataset(
            "mesh_connectivity_offsets",
            data=np.array(offsets, dtype=np.int64),
            chunks=(min(1000, len(offsets)),),
            compressor=self.compressor,
            dtype=np.int64,
        )

        # Add some statistics as metadata
        root.attrs["thickness_min"] = float(np.min(data.filtered_node_thickness))
        root.attrs["thickness_max"] = float(np.max(data.filtered_node_thickness))
        root.attrs["thickness_mean"] = float(np.mean(data.filtered_node_thickness))

        self.logger.info(
            f"Successfully created Zarr store with {num_timesteps} timesteps, "
            f"{num_nodes} nodes, {len(edges_array)} edges"
        )

    def should_skip(self, run_id: str) -> bool:
        """Check if output already exists.

        Args:
            run_id: Run ID to check

        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.overwrite_existing:
            return False

        output_path = self._get_output_path(run_id)
        if output_path.exists():
            self.logger.info(f"Skipping {run_id} - Zarr store already exists")
            return True
        return False

    def cleanup_temp_files(self) -> None:
        """Clean up orphaned temporary Zarr stores from interrupted runs."""
        if not self.output_dir or not self.output_dir.exists():
            return

        # Find all temp Zarr stores (*.zarr_temp)
        for temp_store in self.output_dir.glob("*.zarr_temp"):
            self.logger.warning(f"Removing orphaned temp Zarr store: {temp_store}")
            import shutil

            shutil.rmtree(temp_store)
