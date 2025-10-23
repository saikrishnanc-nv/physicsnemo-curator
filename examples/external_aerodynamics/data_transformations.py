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
import warnings
from typing import Callable, Optional

import numpy as np
import zarr

from examples.external_aerodynamics.external_aero_geometry_data_processors import (
    default_geometry_processing_for_external_aerodynamics,
)
from examples.external_aerodynamics.external_aero_surface_data_processors import (
    default_surface_processing_for_external_aerodynamics,
)
from examples.external_aerodynamics.external_aero_volume_data_processors import (
    default_volume_processing_for_external_aerodynamics,
)
from physicsnemo_curator.etl.data_transformations import DataTransformation
from physicsnemo_curator.etl.processing_config import ProcessingConfig

from .constants import PhysicsConstants
from .external_aero_utils import (
    to_float32,
)
from .schemas import (
    ExternalAerodynamicsExtractedDataInMemory,
    ExternalAerodynamicsNumpyDataInMemory,
    ExternalAerodynamicsNumpyMetadata,
    ExternalAerodynamicsZarrDataInMemory,
    PreparedZarrArrayInfo,
)


class ExternalAerodynamicsNumpyTransformation(DataTransformation):
    """Transforms External Aerodynamics data for NumPy storage format (legacy support)."""

    def __init__(self, cfg: ProcessingConfig):
        super().__init__(cfg)

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsNumpyDataInMemory:
        """Transform data for NumPy storage format.

        Note: This is a legacy format with minimal metadata support.
        For full metadata support, use Zarr storage format instead.

        Args:
            data: External Aerodynamics extracted data in memory

        Returns:
            Data formatted for NumPy storage with basic metadata
        """
        # Create minimal metadata
        numpy_metadata = ExternalAerodynamicsNumpyMetadata(
            filename=data.metadata.filename,
            stream_velocity=data.metadata.stream_velocity,
            air_density=data.metadata.air_density,
        )

        return ExternalAerodynamicsNumpyDataInMemory(
            stl_coordinates=to_float32(data.stl_coordinates),
            stl_centers=to_float32(data.stl_centers),
            stl_faces=to_float32(data.stl_faces),
            stl_areas=to_float32(data.stl_areas),
            metadata=numpy_metadata,
            surface_mesh_centers=to_float32(data.surface_mesh_centers),
            surface_normals=to_float32(data.surface_normals),
            surface_areas=to_float32(data.surface_areas),
            surface_fields=to_float32(data.surface_fields),
            volume_mesh_centers=to_float32(data.volume_mesh_centers),
            volume_fields=to_float32(data.volume_fields),
        )


class ExternalAerodynamicsSTLTransformation(DataTransformation):
    """Transforms STL data for External Aerodynamics model."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        geometry_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.geometry_processors = geometry_processors
        self.logger = logging.getLogger(__name__)

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:
        """Transform STL data for External Aerodynamics model."""

        # Regardless of whether there are any additional geometry processors,
        # we always apply the default geometry processing.
        # This will ensure that the bare minimum criteria for geometry data is met.
        # That is - The geometry data (vertices, faces, areas and centers) are present.
        data = default_geometry_processing_for_external_aerodynamics(data)

        if self.geometry_processors is not None:
            for processor in self.geometry_processors:
                data = processor(data)

        # Delete raw STL data to save memory
        data.stl_polydata = None

        return data


class ExternalAerodynamicsSurfaceTransformation(DataTransformation):
    """Transforms surface data for External Aerodynamics model."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        surface_variables: Optional[dict[str, str]] = None,
        surface_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.logger = logging.getLogger(__name__)

        self.surface_variables = surface_variables
        self.surface_processors = surface_processors
        self.constants = PhysicsConstants()

        if surface_variables is None:
            self.logger.error("Surface variables are empty!")
            raise ValueError("Surface variables are empty!")

        self.logger.info(
            f"Initializing ExternalAerodynamicsSurfaceTransformation with surface_variables: {surface_variables} and surface_processors: {surface_processors}"
        )
        self.logger.info(
            "This will only be processed if the model_type is surface/combined."
        )

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:
        """Transform surface data for External Aerodynamics model."""

        if data.surface_polydata is not None:

            # Regardless of whether there are any additional surface processors,
            # we always apply the default surface processing.
            # This will ensure that the bare minimum criteria for surface data is met.
            # That is - The surface data (mesh centers, normals, areas and fields) are present.
            data = default_surface_processing_for_external_aerodynamics(
                data, self.surface_variables
            )

            if self.surface_processors is not None:
                for processor in self.surface_processors:
                    data = processor(data)

            # Delete raw surface data to save memory
            data.surface_polydata = None

        return data


class ExternalAerodynamicsVolumeTransformation(DataTransformation):
    """Transforms volume data for External Aerodynamics model."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        volume_variables: Optional[dict[str, str]] = None,
        volume_processors: Optional[tuple[Callable, ...]] = None,
    ):
        super().__init__(cfg)
        self.volume_variables = volume_variables
        self.volume_processors = volume_processors
        self.constants = PhysicsConstants()
        self.logger = logging.getLogger(__name__)

        if volume_variables is None:
            self.logger.error("Volume variables are empty!")
            raise ValueError("Volume variables are empty!")

        self.logger.info(
            f"Initializing ExternalAerodynamicsVolumeTransformation with volume_variables: {volume_variables} and volume_processors: {volume_processors}"
        )
        self.logger.info(
            "This will only be processed if the model_type is volume/combined."
        )

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsExtractedDataInMemory:

        if data.volume_unstructured_grid is not None:

            # Regardless of whether there are any additional volume processors,
            # we always apply the default volume processing.
            # This will ensure that the bare minimum criteria for volume data is met.
            # That is - The volume data (mesh centers and fields) are present.
            data = default_volume_processing_for_external_aerodynamics(
                data, self.volume_variables
            )

            if self.volume_processors is not None:
                for processor in self.volume_processors:
                    data = processor(data)

            # Delete raw volume data to save memory
            data.volume_unstructured_grid = None

        return data


class ExternalAerodynamicsZarrTransformation(DataTransformation):
    """Transforms External Aerodynamics data for Zarr storage format."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        compression_method: str = "zstd",
        compression_level: int = 5,
        chunk_size_mb: float = 1.0,  # Default 1MB chunk size
        chunks_per_shard: int = 1000,  # Number of chunks per shard (~1GB shards with 1MB chunks)
    ):
        super().__init__(cfg)
        # Zarr 3 codec configuration
        self.compressor = zarr.codecs.BloscCodec(
            cname=compression_method,
            clevel=compression_level,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        )
        self.chunk_size_mb = chunk_size_mb
        self.chunks_per_shard = chunks_per_shard

        # Warn if chunk size might be problematic
        if chunk_size_mb < 1.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too small. "
                "This could lead to poor performance due to overhead.",
                UserWarning,
            )
        elif chunk_size_mb > 50.0:
            warnings.warn(
                f"Chunk size of {chunk_size_mb}MB might be too large. "
                "This could lead to memory issues and poor random access performance.",
                UserWarning,
            )

    def _prepare_array(self, array: np.ndarray) -> PreparedZarrArrayInfo:
        """Prepare array for Zarr storage with compression, chunking, and sharding.

        Sharding is always enabled to reduce the number of files created on disk.
        Each shard contains approximately chunks_per_shard chunks (~1GB per shard by default).
        """
        if array is None:
            return None

        # Calculate chunk size based on configured size in MB
        target_chunk_size = int(self.chunk_size_mb * 1024 * 1024)  # Convert MB to bytes
        item_size = array.itemsize
        shape = array.shape

        if len(shape) == 1:
            chunk_size = min(shape[0], target_chunk_size // item_size)
            chunks = (chunk_size,)
            # For 1D arrays: shard contains chunks_per_shard chunks
            shard_size = min(shape[0], chunks[0] * self.chunks_per_shard)
            shards = (shard_size,)
        else:
            # For 2D arrays, try to keep rows together
            chunk_rows = min(
                shape[0], max(1, target_chunk_size // (item_size * shape[1]))
            )
            chunks = (chunk_rows, shape[1])
            # For 2D arrays: extend along first dimension to group multiple chunk-rows
            shard_rows = min(shape[0], chunks[0] * self.chunks_per_shard)
            shards = (shard_rows, shape[1])

        return PreparedZarrArrayInfo(
            data=np.float32(array),
            chunks=chunks,
            compressor=self.compressor,
            shards=shards,
        )

    def transform(
        self, data: ExternalAerodynamicsExtractedDataInMemory
    ) -> ExternalAerodynamicsZarrDataInMemory:
        """Transform data for Zarr storage format.

        Organizes data into hierarchical groups and applies compression settings.

        Args:
            data: Dictionary containing External Aerodynamics data

        Returns:
            Dictionary with data formatted for Zarr storage, including:
                - Data organized into groups (stl, surface, volume)
                - Compression settings
                - Chunking configurations
        """
        return ExternalAerodynamicsZarrDataInMemory(
            stl_coordinates=self._prepare_array(data.stl_coordinates),
            stl_centers=self._prepare_array(data.stl_centers),
            stl_faces=self._prepare_array(data.stl_faces),
            stl_areas=self._prepare_array(data.stl_areas),
            metadata=data.metadata,
            surface_mesh_centers=self._prepare_array(data.surface_mesh_centers),
            surface_normals=self._prepare_array(data.surface_normals),
            surface_areas=self._prepare_array(data.surface_areas),
            surface_fields=self._prepare_array(data.surface_fields),
            volume_mesh_centers=self._prepare_array(data.volume_mesh_centers),
            volume_fields=self._prepare_array(data.volume_fields),
        )
