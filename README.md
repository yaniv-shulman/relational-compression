[![Linting and Tests](https://github.com/yaniv-shulman/relational-compression/actions/workflows/linting_and_tests.yml/badge.svg?branch=main)](https://github.com/yaniv-shulman/relational-compression/actions/workflows/linting_and_tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Relational Compression

This repository contains the experiment code and paper sources for:

> **Relational Compression**<br>
> *A Framework for Relational Fidelity in Constrained Representations*

Relational compression treats relational structure itself as the fidelity-bearing source content. The accompanying experiments study finite-codeword collision geometry through controlled synthetic, graph, and image realizations.

## Background

Many representations are constrained not primarily by the accuracy of individual reconstructed elements, but by how well they preserve relationships among elements: graph connectivity, pairwise geometry, or teacher-defined affinities, for example. Relational compression makes these choices explicit by separating the source relation, retained description, reconstructed or evaluated relation, fidelity criterion, and resource constraint.

This separation distinguishes the representation from the decoder that interprets it, and realized codeword occupancy from a complete description-cost convention. It provides a common language for comparing mechanisms that otherwise begin from different settings, including graph summarization, spectral sparsification, similarity-preserving representations, and relational distillation.

## What the Paper Contributes

- A relational-compression framework that identifies the source relation, representation, reconstruction or evaluation rule, fidelity, and resource as separate components.
- A finite-codeword collision realization that connects same-codeword probability with pair-specific alignment and separation, aggregate R'enyi-2 occupancy, and positive-spherical geometry.
- Exact correspondences from inverse-aggregate-mass collision to squared-Euclidean centroid reconstruction and from graph-local inverse-mass affinity to normalized association and cut.
- Five controlled studies that instantiate reconstruction-defined, graph-defined, and teacher-defined relational requirements within one constrained-representation formulation.

## Experimental Studies

1. **Synthetic centroid correspondence:** Numerically checks the exact inverse-mass-weighted centroid--pairwise identity, including encoder gradients, and separates assignment distortion from decoder reproduction mismatch.
2. **Transductive graph fidelities:** Holds a finite graph-local representation and marginal-organization mechanism fixed while exchanging direct-edge, all-mode, transition-collision, and transition-entropy graph fidelities.
3. **Inductive normalized cut:** Trains a shared finite-codeword graph encoder on MalNet-Tiny and evaluates its hard partitions on unseen graphs against normalized-cut and spectral references.
4. **Reconstruction-defined image codes:** Trains a spatial hard-sign image bottleneck on Flowers102 through a joint reconstruction decoder, then examines how marginal organization changes hard-code occupancy and reconstruction quality.
5. **Teacher-defined image relations:** Uses a frozen visual teacher to derive favored and disfavored patch relations on Flowers102, training an independent finite-code student through pair-specific collision without reconstruction or a marginal-organization penalty.

## Repository Layout

```text
src/relational_compression/experiments/  Experiment implementations and runners
src/relational_compression/models/       Shared image models and quantizers
tests/                Unit and integration tests
paper/                LaTeX source, bibliography, and published figures
data/                 Local dataset cache (ignored by Git)
out/                  Generated experiment outputs (ignored by Git)
```

## Requirements

- Python 3.10--3.12
- [Poetry](https://python-poetry.org/)
- A working LaTeX installation with `latexmk` to build the paper
- A CUDA-capable PyTorch installation is recommended for the graph and image studies

## Installation

Install the reproduction environment and development dependencies:

```bash
poetry install --with dev
source configure.sh
```

`configure.sh` defaults datasets to `data/datasets/` and generated artifacts to `out/`. Set `RELCO_DATA_DIR`, `RELCO_OUT_DIR`, or `RELCO_CHECKPOINT_DIR` before sourcing it to use other locations.

## Running the Experiments

Run the controlled synthetic study:

```bash
poetry run python -m relational_compression.experiments.synthetic_collision_centroid.run_scripts.run_synthetic_exp \
  --config default
```

Run the transductive graph study:

```bash
poetry run python -m relational_compression.experiments.transductive_relational_distortion.run_scripts.run_experiment
```

Prepare MalNet-Tiny and run the inductive normalized-cut study:

```bash
poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.prepare_malnet_tiny \
  --dataset-root-dir "$RELCO_DATA_DIR/derived/malnet-tiny/1.0_relational_compression"

poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.run_malnet_tiny_exp \
  --config default \
  --dataset-root-dir "$RELCO_DATA_DIR/derived/malnet-tiny/1.0_relational_compression"
```

Run the reconstruction-trained Flowers102 study:

```bash
poetry run python -m relational_compression.experiments.teacherless_image_compression.run_scripts.run_flowers_exp \
  --config flowers102.flowers102_mean_group
```

Run the teacher-defined Flowers102 study:

```bash
poetry run python -m relational_compression.experiments.teacher_image_compression.run_scripts.run_flowers_exp \
  --config flowers102_dino_vits8
```

Each experiment package contains more focused notes and smoke-test commands. Full runs may download public datasets or pretrained model weights and can require substantial compute.

The maintained reproduction interface covers these five studies; included paper artifacts do not make one-off publication-specific plotting or assembly scripts public interfaces.

## Testing

Run the complete lint, type-check, and test suite with:

```bash
make check
```

The test suite can also be run directly:

```bash
poetry run pytest
```

## Building the Paper

From the repository root:

```bash
latexmk -cd -pdf paper/relational_compression.tex
```

The manuscript master is `paper/relational_compression.tex`. Its section sources are under `paper/sections/`, and all included PNG figures are under `paper/graphics/`.

## License

Original software in this repository is released under the [MIT License](LICENSE). Third-party materials retain their respective licenses; in particular, `paper/elsarticle.cls` is distributed under the LaTeX Project Public License as stated in its file header.
