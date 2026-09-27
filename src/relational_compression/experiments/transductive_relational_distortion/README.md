# Transductive Relational Distortion

This experiment isolates relational distortion as the alignment source for a finite collision bottleneck.
It reuses the cleaned MalNet-Tiny LCC graphs from `inductive_normalized_cut`, adds small TUDataset graph collections, ignores labels, and optimizes one free graph-local `n x K` logit matrix per graph.
There is no GNN, decoder, teacher, node-feature input, or auxiliary task.

For each graph, the representation is a K-way categorical distribution

```text
q_i(z) = softmax(logits_i / tau),
k_ij = sum_z q_i(z) q_j(z).
```

The four source-defined distortion criteria are:

- `D_E`: normalized direct edge-content distortion, using edge weights `rho_e` proportional to source edge weight `w_e`.
- `D_F`: normalized graph-Fourier / resistance-weighted Dirichlet distortion, using edge weights `rho_e` proportional to `w_e R_eff^G(e)`.
- `D_C`: source collision-geometry distortion, using edge weights `rho_e` proportional to `w_ij ||P_i - P_j||_2^2` for row-normalized source neighbor distributions.
- `D_H2`: source-intrinsic collision-entropy probe distortion, using edge weights `rho_e` proportional to `w_ij (H_2(P_i)-H_2(P_j))^2`.

For unweighted graphs `H_2(P_i)=log d_i`, so the collision-entropy probe is a log-degree signal.
If its denominator is zero, `D_H2` is marked undefined for that graph and skipped for training on that graph.

All four criteria use the same normalized edge-collision distortion core:

```text
D_rho(q) = sum_e rho_e (1 - k_ij),  sum_e rho_e = 1.
```

The only anti-collapse term is graph-local marginal code-space organization:

```text
D_2(q_bar_G || U_K) = log(K sum_z q_bar_G(z)^2),
q_bar_G(z) = sum_i d_i q_i(z) / vol(G).
```

The optimized objective is:

```text
loss = D_bullet(q) + lambda_org D_2(q_bar_G || U_K).
```

`H_2` and `K_eff` are collision-effective finite-state complexity/utilization diagnostics, not operational bitrates.
Each final hard partition is cross-evaluated under every defined source distortion, plus the existing hard normalized-cut metric.
Restarts are solver initializations selected by the best target soft total objective encountered during that restart.
The final reported solver uses Adam with learning rate `0.05`, `600` steps, and assignment temperature `1.0`.
The default run includes one biased near-collapse initialization so the known low-organization collapse control is reachable, plus the configured random starts.
When multiple source criteria are run, the runner also refines each criterion from the other criteria's selected logits as additional warm-start initializations, with at most one closure round and a final target-wise candidate comparison.
Selected hard assignments, logits, and probabilities are written under `partition_artifacts/`; CSV outputs store only scalar diagnostics and artifact paths.

Additional diagnostic outputs:

- `source_rho_correlations.csv`: Pearson/Spearman/cosine and distance comparisons between all defined `rho` vectors.
- `random_partition_distortion_correlations.csv`: random balanced-partition distortion-ranking correlations.
- `cross_objective_diagnostics.csv`: complete target-objective values across criteria for detecting missed target-objective improvements.
- `criterion_exclusions.csv`: graphs where a requested criterion, usually `D_H2`, is undefined.

Run the default final sweep:

```bash
poetry run python -m relational_compression.experiments.transductive_relational_distortion.run_scripts.run_experiment
```

For a quick synthetic smoke run:

```bash
poetry run python -m relational_compression.experiments.transductive_relational_distortion.run_scripts.run_experiment \
  --dataset-backend synthetic \
  --source-collections synthetic \
  --graphs-per-collection 2 \
  --optimization-steps 50 \
  --restart-seeds 1337 \
  --experiments-dir /tmp/relational_compression_trd_smoke \
  --no-plots
```
