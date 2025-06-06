# How to curate data for DoMINO model with PhysicsNeMo-Curator

This document describes the DoMINO data processing pipeline in PhysicsNeMo Curator.

## What this pipeline does

The DoMINO ETL pipeline processes automotive aerodynamics simulation data for machine learning training. It:

**Reads:** CFD simulation data in multiple formats (DriveSim, DrivAerML, AhmedML)

- Geometry files (STL format)
- Volume mesh data (VTU format)
- Surface mesh data (VTP format)
- Flow field variables (pressure, velocity, turbulence)

**Transforms:** Raw simulation data into ML-optimized format

- Extracts and normalizes field variables (pressure coefficients, wall shear stress)
- Processes geometry data (coordinates, face connectivity, areas)
- Computes derived quantities and reference values
- Applies mesh decimation for efficiency (optional)

**Outputs:** Training-ready datasets in numpy or Zarr format

- Optimized for DoMINO model training workflows
- Compressed and chunked for efficient data loading
- Preserves metadata and processing parameters

This pipeline handles the complex data engineering required to convert raw CFD outputs into
datasets suitable for training AI models for external aerodynamics applications.

## Running the Curator

Example of the command line that launches Curator configured for DrivAerML dataset:

```bash
physicsnemo-curator-etl                         \
    --config-name=domino_etl                    \
    etl.source.input_dir=/data/drivaerml/       \
    etl.sink.output_dir=/data/drivaerml.processed.surface \
    etl.common.model_type=surface
```

To run on Ahmed Body ML dataset:

```bash
physicsnemo-curator-etl                     \
    --config-name=domino_etl_ahmed_ml      \
    etl.source.input_dir=/data/ahmed_ml/   \
    etl.sink.output_dir=/data/ahmed_ml.processed.surface \
    etl.common.model_type=surface
```

### Some useful configuration options

- `etl.sink.overwrite_existing`: can be either `true` (default) or `false`.
    When set to `false`, the Curator will not overwrite existing files.
    This is useful for incremental runs of the Curator.
- `etl.common.model_type`: can be `surface` (default) or `volume` or `combined`.
    This option is used to specify the type of model that will be trained on the
    dataset.

Please refer to the [config file](../../../config/domino_etl.yaml) for more
options.
