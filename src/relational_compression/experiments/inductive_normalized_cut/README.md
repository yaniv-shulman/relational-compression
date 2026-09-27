# Inductive Normalized Cut on MalNet-Tiny

This experiment trains one inductive graph encoder to minimize the q_Z-normalized collision form of K-way normalized cut on cleaned MalNet-Tiny function-call graphs. It is not a graph-classification experiment: malware labels are not used as inputs, targets, auxiliary losses, or checkpoint-selection signals.

## Data

The default prepared-data location is:

```bash
$RELCO_DATA_DIR/derived/malnet-tiny/1.0_relational_compression
```

Prepare the independent cache with:

```bash
source configure.sh
poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.prepare_malnet_tiny \
  --dataset-root-dir "$RELCO_DATA_DIR/derived/malnet-tiny/1.0_relational_compression"
```

Preparation uses PyTorch Geometric's official `MalNetTiny` loader and official `train`/`val`/`test` splits. Each graph is converted to a simple undirected graph, stripped of self-loops and duplicate edges, reduced to its largest connected component, reindexed, filtered by LCC size, and cached with structural node features only.

Node features are a constant, degree, `log1p` degree, normalized degree, and a bounded-memory Hutchinson/Rademacher approximation to random-walk return probabilities `diag(P^t)` for `P = D^{-1}A`. The default uses 16 walk steps and 8 probes. These positional features are a finite-probe stochastic approximation, deterministic for a fixed preprocessing seed and graph, and are computed by sparse edge propagation without forming dense transition powers.

The default cache also appends directed structural features computed before undirected LCC conversion: original directed in-degree, out-degree, total degree, their `log1p` and graph-normalized forms, and directed PageRank relative to the uniform baseline (`N * PageRank`). These values are carried through LCC extraction and reindexing; labels remain unused.

The default filter keeps graphs with at least 512 LCC nodes and no upper size limit.

## Objective

For each graph, the encoder produces categorical probabilities q_i(k). With degree-weighted aggregate q_Z(k), the sparse training objective is:

```text
NAssoc_soft = (1 / vol(G)) sum_(i,j) W_ij sum_k q_i(k) q_j(k) / q_Z(k)
Ncut_soft   = K - NAssoc_soft
```

For one-hot assignments this equals the classical K-way normalized-cut objective. The default config adds graph-local aggregate separation:

```text
D2(q_Z || U_K) = log(K * sum_k q_Z(k)^2)
loss = mean_g Ncut_soft^(g) + separation_weight * mean_g D2(q_Z^(g) || U_K)
```

The default uses `separation_weight = 0.10`. Validation checkpointing uses mean hard K-way normalized cut, lower is better.

## Running

After preparing the cache:

```bash
source configure.sh
poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.run_malnet_tiny_exp \
  --config default \
  --dataset-root-dir "$RELCO_DATA_DIR/derived/malnet-tiny/1.0_relational_compression"
```

A network-free synthetic smoke run is available:

```bash
source configure.sh
poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.run_malnet_tiny_exp \
  --dataset-backend synthetic \
  --num-epochs 2 \
  --batch-size 2 \
  --hidden-dim 16 \
  --num-layers 2 \
  --num-partitions 3 \
  --skip-spectral \
  --disable-tensorboard
```

The spectral normalized-cut reference is evaluated only on validation/test graphs and cached under the run directory. It is a transductive reference, not supervision.

The default encoder uses GraphGPS-style blocks with local GIN message passing and global factorized-softmax Efficient Attention, explicit node-wise LayerNorm, and a residual feed-forward block. Efficient Attention is implemented locally with padding-aware key/value/query masking and does not construct a quadratic attention matrix. The default run keeps `K=8`, hidden width 128, four layers, assignment temperature 1.0, batch size 80, and learning rate `5e-4`.
