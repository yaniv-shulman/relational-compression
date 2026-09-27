# Synthetic collision-centroid equivalence

This experiment provides a controlled numerical demonstration of the finite-sample identity between centroid distortion and qZ-normalized collision-weighted pairwise distortion.

A small MLP maps points from an imbalanced two-dimensional Gaussian mixture to `b` binary logits. The logits define an exact factorized distribution over all `2**b` discrete codes. For those assignments the experiment compares

- expected squared distance to each code centroid;
- the qZ-normalized within-code pairwise squared distance;
- the same pairwise quantity without qZ normalization;
- reconstruction through learned code vectors and its exact decoder-centroid gap.

Two encoders are cloned from the same initialization. One is optimized using centroid distortion and one using the normalized pairwise distortion. The experiment records both views at every step, checks their gradient agreement, evaluates the identity over random encoder states, and trains a separate codebook decoder to verify

`decoder distortion = centroid distortion + decoder-centroid gap`.

Run the default experiment with:

```bash
poetry run python -m relational_compression.experiments.synthetic_collision_centroid.run_scripts.run_synthetic_exp \
  --config default
```

Outputs use the standard experiment/run layout: `RELCO_OUT_DIR/experiments/<experiment_name>/run_00_deterministic/` contains `result.json`, CSV histories, TensorBoard logs, and figures under `figures/`; the experiment root contains the effective config and `all_run_results.json`.
