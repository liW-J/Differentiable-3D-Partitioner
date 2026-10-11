# Differentiable 3D Partitioner

A PyTorch-based differentiable partitioner for assigning cells to the top or
bottom die in a 3D placement flow. It combines placement-aware wirelength,
cutsize, local area balance, and density objectives with DREAMPlace operators.

For the complete placement flow and benchmark inputs, see
[D2D-Placer with Differentiable Partitioner](https://github.com/liW-J/D2D-Placer-With-Differentiable-Partitioner).

## Current API

The package exports three classes:

- `DreamplaceParser`: reads a DREAMPlace JSON design configuration or attaches
  an existing D2D DREAMPlace database.
- `Differentiable3DPartitionerFlow`: loads a YAML optimization configuration,
  runs the differentiable optimization, and saves binary die assignments.
- `Partitioner`: the underlying PyTorch model for custom integration.

The flow uses connectivity, positions, cell sizes, die bounds, and DREAMPlace
operators. It assigns cells to two dies; the balance grid controls local area
constraints within each die.

## Installation

Run the following commands from the standalone repository:

```bash
git clone --recursive https://github.com/liW-J/Differentiable-3D-Partitioner.git
cd Differentiable-3D-Partitioner
python -m pip install -r requirements.txt
python -m pip install -e .
```

A built DREAMPlace runtime, including its compiled operators, is also required.
Installing this Python package does not compile DREAMPlace. Follow the nested
DREAMPlace repository's build instructions and install its runtime under
`thirdparty/DREAMPlace/install`, where the parser looks for the `dreamplace`
package. Its C++/CUDA extensions must match the Python and PyTorch environment
used to run the partitioner.

When used inside D2D-Placer, use the parent repository's build and run workflow.
The D2D adapter uses its DREAMPlace runtime and prepares the design data for the
partitioner. This repository's CMake installs `partitioner`, `configs`,
`examples`, and `tests` recursively; building a separate nested DREAMPlace
runtime is controlled by `D3D_BUILD_NESTED_DREAMPLACE` (default: `OFF`).

## ASAP7 Configurations

The current case configurations are under [`configs/openroad/asap7`](configs/openroad/asap7).
Each link below opens the complete YAML, including density and gradient schedules.
Loss-weight columns show `start → end`, or one value for a constant schedule.

| Case | Iterations | Random seed | Cutsize weight | Balance weight | Balance grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| [aes](configs/openroad/asap7/aes.yaml) | 500 | 888 | 1e+07 | 1000 | 4 × 4 |
| [ariane133](configs/openroad/asap7/ariane133.yaml) | 1000 | 888 | 1e+12 | 10000 | 1 × 1 |
| [cva6](configs/openroad/asap7/cva6.yaml) | 500 | 888 | 2.2e+07 | 10000 | 20 × 20 |
| [ethmac](configs/openroad/asap7/ethmac.yaml) | 500 | 888 | 1e+07 | 10000 | 20 × 20 |
| [ibex](configs/openroad/asap7/ibex.yaml) | 2000 | 3888 | 2e+11 | 1e+06 | 1 × 1 |
| [jpeg](configs/openroad/asap7/jpeg.yaml) | 1000 | 3888 | 4e+10 | 1000 | 100 × 100 |
| [mock-alu](configs/openroad/asap7/mock-alu.yaml) | 500 | 888 | 2e+09 | 100000 | 1 × 1 |
| [mock-cpu](configs/openroad/asap7/mock-cpu.yaml) | 500 | 888 | 2e+11 | 1e+06 | 1 × 1 |
| [riscv32i](configs/openroad/asap7/riscv32i.yaml) | 500 | 888 | 1e+06 | 5 | 1 × 1 |
| [uart](configs/openroad/asap7/uart.yaml) | 500 | 888 | 2e+09 → 2e+11 | 100000 → 1e+06 | 1 × 1 |

All listed cases use Nesterov with learning rate `0.1` and `use_bb: true`.
The older ICCAD, Nangate45, mixed-technology, Open3DBench, and scaled-case YAMLs
are no longer included in the current configuration set.

[`configs/default.yaml`](configs/default.yaml) is a generic starting point,
not an automatically merged base for the case YAMLs. The flow loads the selected
file directly and supplies code defaults for omitted fields. Its `design.name`
is still `gcd`; set that field when adapting it to another design.

### Configuration fields

| Section | Purpose |
| --- | --- |
| `design.name` | Design name used in output subdirectories. |
| `flow.num_iterations`, `flow.random_seed` | Optimization length and random seed. |
| `flow.optimizer` | Optimizer name, learning rate, and Nesterov BB step option. |
| `flow.lse_smoothing_alpha` | Start/end smoothing parameters for the wirelength model. |
| `flow.cutsize_loss`, `flow.balance_loss`, `flow.density_loss` | Enable flags and loss-weight schedules. |
| `partitioner.t_grad_*` | Gradient scale and warm-up/ramp schedule for assignment logits. |
| `partitioner.gumbel_*` | Sigmoid-to-Gumbel switch and temperature annealing. |
| `partitioner.balance_loss` | Balance grid and per-die area threshold factors. |
| `visualization` | Output directory and save/log intervals. |
| `output.result_dir` | Directory for saved binary assignments. |

Training iterations are zero-based. All current ASAP7 configs set
`gumbel_switch_iteration: 500`, so a 500-iteration run performs iterations
0–499 entirely with sigmoid assignments. The longer `ariane133`, `ibex`, and
`jpeg` runs enter the Gumbel phase. Temperature annealing starts at the switch;
its horizon is 2,000 steps for `ariane133` and 20,000 for the other cases, so
none of these runs completes the full annealing schedule.
Final binary assignments are obtained by thresholding the learned probabilities.

## Run with D2D-Placer

In the parent D2D repository, a placement JSON selects the partitioner and the
case YAML directory:

```json
{
  "txt_input": "benchmarks/openroad/asap7/aes/aes.txt",
  "partitioner": "d3d",
  "differentiable_partitioner_config_root": "openroad/asap7"
}
```

These fields are part of a full D2D placement configuration. After building and
installing D2D, run its supplied configuration from the parent install directory:

```bash
cd /path/to/D2D-Placer-With-Differentiable-Partitioner/install
python placer/d2d_placer.py test/openroad/asap7/aes.json
```

D2D resolves `aes.json` to `configs/openroad/asap7/aes.yaml` in its installed
partitioner tree. Keep the JSON and YAML basenames aligned. The adapter prepares
the DREAMPlace input and writes a runtime copy of the YAML with the case name
and output paths filled in. Changes to source YAMLs must be installed again
before running from the installed tree.

## Standalone Python Usage

Standalone runs need **two configurations**: a DREAMPlace JSON describing the
input design and a partitioner YAML describing optimization. The JSON must point
to existing Bookshelf inputs or LEF/DEF inputs supported by your DREAMPlace build,
and include valid `global_place_stages` so the parser can initialize density
operators. The D2D `txt_input` JSON above is consumed by D2D, not directly by
`DreamplaceParser`.

Run from this repository root after building DREAMPlace and preparing your input
JSON. Replace `/absolute/path/to/aes/dreamplace.json` with your design file:

```python
from partitioner import DreamplaceParser, Differentiable3DPartitionerFlow

parser = DreamplaceParser()
parser.parse_design("/absolute/path/to/aes/dreamplace.json")

flow = Differentiable3DPartitionerFlow(
    num_nodes=parser.num_nodes,
    num_nets=parser.num_nets,
    num_pins=parser.num_pins,
    node_pos=parser.node_pos,
    pin_pos=parser.pin_pos,
    flat_net2pin_map=parser.flat_net2pin_map,
    flat_net2pin_start_map=parser.flat_net2pin_start_map,
    pin2node_map=parser.pin2node_map,
    node_size_x=parser.node_size_x,
    node_size_y=parser.node_size_y,
    dreamplace_basic=parser.dreamplace_basic,
    config_path="configs/openroad/asap7/aes.yaml",
    die_xl=parser.die_xl,
    die_yl=parser.die_yl,
    die_xh=parser.die_xh,
    die_yh=parser.die_yh,
)
metrics = flow.run()
print(metrics["true_d2d_hpwl"], metrics["true_cutsize"])
```

The benchmark inputs are not bundled in this standalone repository.
[`examples/run_partitioner.py`](examples/run_partitioner.py) demonstrates the
file-based flow, but its hard-coded benchmark path must be replaced with your
own input. `python -m partitioner` only provides help/version information;
it does not launch an optimization job.

### Outputs

For the current case YAMLs, standalone runs write:

```text
results/<design.name>/binary_assignment.pt
results/<design.name>/binary_assignment.txt
results/visualizations/<design.name>/
```

Relative output directories are resolved against the partitioner repository
root. The assignment has one entry per physical node: `1` for the top die and
`0` for the bottom die. `flow.run()` returns optimization metrics together with
`true_d2d_hpwl` and `true_cutsize` for the final binary assignment.

D2D overrides these directories under its case scratch directory:

```text
run_tmp/<case>/differentiable-3d-partitioner/
  d2d_partitioner_config.yaml
  results/<case>/binary_assignment.pt
  results/<case>/binary_assignment.txt
  visualizations/<case>/
```

## Project Structure

```text
configs/default.yaml        Generic optimization template
configs/openroad/asap7/     Ten case-specific optimization YAMLs
partitioner/core/flow.py    Optimization flow and result export
partitioner/core/partitioner.py
                           Differentiable PyTorch model
partitioner/parsers/dreamplace_parser.py
                           DREAMPlace design adapter
partitioner/utils/         Visualization and assignment export
examples/                  Experiment scripts with local input paths
tests/                     Development tests
thirdparty/DREAMPlace/     Nested DREAMPlace submodule
CMakeLists.txt             Installation and optional nested runtime build
```

## License

See [LICENSE](LICENSE) for the BSD 3-Clause license.

## Citation

If you use this partitioner in your research, please cite
[our paper](https://arxiv.org/abs/2609.06408):

```bibtex
@misc{jiang2026differentiablepartitioning,
  title         = {Differentiable Partitioning With Dual-Max 3D Wirelength Model and Hybrid Bonding Terminal Awareness for Optimized 3D Placement},
  author        = {Liwen Jiang and Xu Shi and Rufeng Xiao and Changhao Yan and Rujun Jiang and Zhiang Wang and Keren Zhu},
  year          = {2026},
  eprint        = {2609.06408},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.06408}
}
```
