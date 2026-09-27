# Teacher Image Compression

This package contains the teacher-defined Flowers102 image-compression experiment.
It is separate from `teacherless_image_compression` and does not use a reconstruction
loss during representation learning.

The baseline teacher is the official DINO ViT-S/8 PyTorch Hub model:

- repository entry: `facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a`
- model entrypoint: `dino_vits8`
- checkpoint URL: `https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth`
- patch size: `8`, giving a `32x32` patch-token grid for `256x256` inputs

The DINO teacher is frozen and eval-only. Teacher downloads and torch-hub files are
cached under `RELCO_OUT_DIR/model_cache/teacher_image_compression/dino_vits8` by default.

Representation learning treats sampled frozen-DINO patch tokens as a signed
teacher graph. Row-wise softmax teacher affinities above a uniform row baseline
define positive same-partition weights, and affinities below that baseline define
negative different-partition weights. The student encoder outputs one 16-bit
zero-threshold sign code per `32x32` spatial patch; no reconstruction,
mean-group relaxation, concentration loss, or aggregate separation loss is used
during representation learning. Optional reconstruction evaluation trains a
decoder after freezing the encoder and uses hard codes only.

The baseline uses `code_temperature=0.25` for relaxed training collisions and
`teacher_temperature=0.1` for the signed DINO graph.

TensorBoard logs include a training diagnostic panel with one student input next
to the corresponding DINO patch-token cosine-similarity heatmap. Set
`--similarity-diagnostic-epochs 0` to disable it.

When `decoder_num_epochs` is positive, the runner freezes the trained
representation, trains the decoder on hard codes only, and logs validation
input/reconstruction image pairs under `decoder/validate_input_reconstruction`.
Use `--decoder-reconstruction-log-epochs 0` to disable decoder image logging.

Run the baseline experiment with:

```bash
poetry run python -m relational_compression.experiments.teacher_image_compression.run_scripts.run_flowers_exp \
  --config flowers102_dino_vits8
```

To train only the optional decoder for an existing completed representation
experiment, pass the existing experiment name and a positive decoder epoch
count. The runner will load the representation checkpoint, skip representation
epochs that are already complete, and then train or resume the decoder stage:

```bash
poetry run python -m relational_compression.experiments.teacher_image_compression.run_scripts.run_flowers_exp \
  --config flowers102_dino_vits8 \
  --experiment-name EXISTING_EXPERIMENT_NAME \
  --sampled-tokens 1024 \
  --decoder-num-epochs 150 \
  --no-download \
  --no-teacher-download
```
