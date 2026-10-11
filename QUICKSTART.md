# Quick Start

The current flow uses DREAMPlace design data and a case-specific optimization
YAML. Start with the [installation instructions](README.md#installation).

## Using the D2D placement flow

After building and installing the parent D2D-Placer repository:

```bash
cd /path/to/D2D-Placer-With-Differentiable-Partitioner/install
python placer/d2d_placer.py test/openroad/asap7/aes.json
```

The D2D JSON selects `configs/openroad/asap7/aes.yaml` through
`differentiable_partitioner_config_root: "openroad/asap7"`. The JSON basename
must match the YAML basename. Reinstall after editing source configurations.
See [Run with D2D-Placer](README.md#run-with-d2d-placer) for the configuration fields.

## Running the partitioner independently

1. Build and install the nested DREAMPlace runtime and install this Python package.
2. Prepare a DREAMPlace JSON with valid design inputs and `global_place_stages`.
3. Choose one of the [ASAP7 YAMLs](configs/openroad/asap7).
4. Use the [standalone Python example](README.md#standalone-python-usage), replacing
   its input path with your design JSON.

The DREAMPlace JSON describes the design; the YAML controls optimization.
`configs/default.yaml` is a separate template and is not merged into case YAMLs.
The standalone repository does not bundle benchmark inputs.

See [Outputs](README.md#outputs) for assignment files and metrics, and
[Configuration fields](README.md#configuration-fields) for schedule behavior.
