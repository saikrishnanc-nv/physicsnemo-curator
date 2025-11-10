# How to curate data for Crash Simulation models with PhysicsNeMo-Curator

This document describes the Crash Simulation data processing pipeline in PhysicsNeMo Curator.

## Overview

The Crash Simulation ETL pipeline processes LS-DYNA crash simulation data for machine learning training. It:

**Reads:** LS-DYNA simulation data (d3plot binary format)

- d3plot binary files containing mesh and displacement data
- Optional .k files for part thickness information
- Multi-timestep deformation data

**Transforms:** Raw simulation data into ML-optimized format

- Filters wall nodes (nodes with minimal displacement)
- Computes node thickness from part/section definitions
- Builds edge connectivity from mesh
- Remaps and compacts node indices

**Outputs:** Training-ready datasets in VTP or Zarr format

- **VTP format**: Single file per run with all timesteps as displacement fields
- **Zarr format**: Compressed, chunked format optimized for ML training
- Compatible with PhysicsNeMo training workflows

This pipeline handles the complex data engineering required to convert raw LS-DYNA outputs into
datasets suitable for training AI models for crash simulation applications.

## Running the Curator

Example command line to launch Curator for crash simulation data:

```bash
export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl                                    \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl                                \
    etl.source.input_dir=/data/crash_sims/                 \
    etl.sink.output_dir=/data/crash_processed/             \
    etl.processing.num_processes=4
```

### Configuration Options

The main configuration file is [`crash_etl.yaml`](./config/crash_etl.yaml).

#### Key Parameters

- **`etl.source.input_dir`**: Directory containing run folders with d3plot files
  - Expected structure: `input_dir/Run100/d3plot`, `input_dir/Run100/*.k`

- **`etl.sink.output_dir`**: Directory where processed files will be written

- **`etl.processing.num_processes`**: Number of parallel processes (default: 4)

- **`etl.transformations.crash_transform.wall_threshold`**: Threshold for filtering wall nodes (default: 1.0)
  - Nodes with displacement variation < threshold are considered "wall" nodes and filtered out

#### Output Formats

You can choose between two output formats by configuring the sink:

##### VTP Output (for visualization and training)

```yaml
etl:
  sink:
    _target_: examples.structural_mechanics.crash.data_sources.CrashVTPDataSource
    output_dir: /data/crash_processed_vtp
    overwrite_existing: false
    time_step: 0.005  # Time step between frames (seconds)
```

Output structure:

```bash
crash_processed_vtp/
├── Run100.vtp
├── Run101.vtp
└── ...
```

Each VTP file contains:

- Reference coordinates at t=0
- Displacement fields for each timestep: `displacement_t0.000`, `displacement_t0.005`, etc.
- Node thickness values

##### Zarr Output (for ML training)

```yaml
etl:
  sink:
    _target_: examples.structural_mechanics.crash.data_sources.CrashZarrDataSource
    output_dir: /data/crash_processed_zarr
    overwrite_existing: false
    compression_level: 3  # 1-9, higher = more compression
```

Output structure:

```bash
crash_processed_zarr/
├── Run100.zarr/
│   ├── mesh_pos              # (timesteps, nodes, 3)
│   ├── node_thickness        # (nodes,)
│   ├── edges                 # (num_edges, 2)
│   └── mesh_connectivity_*   # Ragged array format
├── Run101.zarr/
└── ...
```

### Transformation Pipeline

The ETL pipeline applies the following transformations:

1. **Node Filtering** (`CrashDataTransformation`):
   - Identifies wall nodes (low displacement variation)
   - Filters out wall nodes to reduce dataset size
   - Remaps mesh connectivity to filtered indices
   - Compacts to contiguous node indices

2. **Thickness Computation**:
   - Parses .k files for part/section thickness definitions
   - Maps part thickness to node thickness
   - Averages thickness for nodes connected to multiple elements

3. **Edge Building**:
   - Extracts unique edges from mesh connectivity
   - Creates edge list for graph neural network training

### Advanced Configuration

#### Adjusting Wall Filtering

To change the wall node filtering threshold:

```bash
physicsnemo-curator-etl \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    etl.transformations.crash_transform.wall_threshold=2.0
```

Higher threshold = more aggressive filtering (more nodes removed)

#### Incremental Processing

To avoid reprocessing existing files:

```bash
physicsnemo-curator-etl \
    --config-dir=examples/structural_mechanics/crash/config \
    --config-name=crash_etl \
    etl.sink.overwrite_existing=false
```

The pipeline will skip runs that already have output files.

## Input Data Requirements

### Directory Structure

The input directory should contain run folders with d3plot files:

```bash
input_dir/
├── Run100/
│   ├── d3plot          # Required: binary mesh/displacement data
│   └── run100.k        # Optional: part thickness definitions
├── Run101/
│   ├── d3plot
│   └── run101.k
└── ...
```

### File Formats

- **d3plot**: LS-DYNA binary format containing:
  - Node coordinates
  - Node displacements over time
  - Shell element connectivity
  - Part IDs for each element

- **.k files**: LS-DYNA keyword format containing:
  - `*PART` definitions linking parts to sections
  - `*SECTION_SHELL` definitions with thickness values

If no .k file is found, node thickness defaults to 0.

## Output Data Format

### VTP Format

Single VTP file per run compatible with PyVista and ParaView:

```python
import pyvista as pv

mesh = pv.read('Run100.vtp')
print(mesh.point_data.keys())
# ['thickness', 'displacement_t0.000', 'displacement_t0.005', ...]

# Get coordinates and displacement at timestep 5
coords = mesh.points
disp_t5 = mesh.point_data['displacement_t0.025']
pos_t5 = coords + disp_t5
```

### Zarr Format

Compressed, chunked arrays optimized for ML training:

```python
import zarr

store = zarr.open('Run100.zarr', mode='r')
print(store.tree())

# Access data
mesh_pos = store['mesh_pos'][:]  # (timesteps, nodes, 3)
thickness = store['node_thickness'][:]  # (nodes,)
edges = store['edges'][:]  # (num_edges, 2)

# Metadata
print(store.attrs['num_timesteps'])
print(store.attrs['num_nodes'])
```

## Model Training

After processing your data with Curator, you can train crash simulation models using PhysicsNeMo. Please refer to [these instructions](physicsnemo-main-repo/examples/structural_mechanics/crash/README.md).

## Extending the Pipeline

The ETL pipeline is designed to be customizable:

1. **Custom Transformations**: Modify `CrashDataTransformation` to add new filtering or feature extraction logic

2. **Additional Output Formats**: Create new DataSource classes for different output formats

3. **Custom Features**: Add new node or element features by extending the data reading logic

See the [PhysicsNeMo-Curator documentation](https://github.com/NVIDIA/physicsnemo-curator) for more details on extending the pipeline.

## Troubleshooting

### Issue: "No d3plot files found"

**Solution**: Ensure your input directory contains subdirectories with d3plot files

### Issue: "No cells left after filtering"

**Solution**: Wall threshold is too low. Try increasing `wall_threshold` parameter.

### Issue: Out of memory

**Solution**: Reduce `num_processes` or process fewer runs at once
