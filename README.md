# Relational Compression

This repository contains the experiment code and paper sources for:

> **Relational Compression**<br>
> *A Framework for Relational Fidelity in Constrained Representations*

Relational compression treats relational structure itself as the fidelity-bearing source content. The accompanying experiments study finite-codeword collision geometry through controlled synthetic, graph, and image realizations.

## Repository Layout

```text
src/relational_compression/experiments/  Experiment implementations and runners
src/relational_compression/models/       Shared image models and quantizers
tests/                Unit and integration tests
paper/                LaTeX source, bibliography, and published figures
data/                 Local dataset cache (ignored by Git)
out/                  Generated experiment outputs (ignored by Git)
```

The five maintained studies are:

1. Synthetic centroid and pairwise-distortion correspondence
2. Transductive graph relational-fidelity comparison
3. Inductive normalized cut on MalNet-Tiny
4. Reconstruction-trained finite image codes on Flowers102
5. Teacher-defined finite image codes on Flowers102

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
