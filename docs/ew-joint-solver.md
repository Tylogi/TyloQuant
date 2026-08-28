# Expert-Wise joint budget solver

`mfq solve-ew` converts an expert-importance table and a rate-distortion
candidate table into an `mfq.calibration-scheme.v3` recipe.
`mfq quantize --ew-scheme` reads that recipe.

```text
mfq solve-ew \
  --importance importance.json \
  --candidates candidates.json \
  --budget budget.json \
  --output-scheme scheme.json \
  --report allocation-report.json
```

The solver uses `scipy.optimize.milp/HiGHS`. Candidate quantization and
distortion measurement may run on GPU before producing the candidate table;
the discrete allocation runs on CPU.

## Importance table

Format: `mfq.ew-importance.v1`.

Score mode preserves the supplied magnitudes. A score without `projection`
is shared by Gate, Up, and Down. An entry with `projection` overrides the
shared score for that projection. An exact `tensor` entry has highest
precedence.

```json
{
  "format": "mfq.ew-importance.v1",
  "mode": "score",
  "score_normalization": "none",
  "entries": [
    {"layer": 0, "expert_id": 0, "score": 0.812},
    {"layer": 0, "expert_id": 0, "projection": "down", "score": 1.337}
  ],
  "metadata": {"metric": "router-weighted expert output energy"}
}
```

Supported score normalizations are `none`, `global_sum`, `layer_sum`,
`layer_projection_sum`, and `tensor_sum`. `none` is the default and retains
absolute differences between layers.

Rank-only mode requires unique contiguous ranks `1..N` within every supplied
scope. Since ranks contain no magnitude, `rank_weighting` explicitly records
the synthetic weight used in the distortion objective:

```json
{
  "format": "mfq.ew-importance.v1",
  "mode": "rank",
  "rank_weighting": "linear_percentile",
  "entries": [
    {"layer": 0, "expert_id": 0, "rank": 1},
    {"layer": 0, "expert_id": 1, "rank": 2}
  ]
}
```

The available policies are:

- `linear_percentile`: `(N-rank+1)/N`;
- `reciprocal`: `1/rank`;
- `uniform`: distortion is weighted uniformly; ranks are retained in the
  input metadata only.

## Candidate table

Format: `mfq.ew-candidates.v1`.

Each candidate supplies four independent quantities:

- `distortion`: unweighted quality loss for this expert projection;
- `effective_bpw`: the rate used by peak and histogram constraints;
- `variable_storage_bits`: storage added by selecting this expert;
- `pool_storage_bits`: storage paid once when any expert activates `pool_key`.

`fixed_storage_bits` on a tensor is always paid. These three storage terms
must reproduce the serialized expert blob exactly. Non-expert and model
container storage belongs in the budget document.

```json
{
  "format": "mfq.ew-candidates.v1",
  "artifact_root": "artifacts",
  "tensors": [
    {
      "name": "blk.0.ffn_down_exps.weight",
      "group": "layer.0.down",
      "layer": 0,
      "projection": "down",
      "n_experts": 1,
      "rows_per_expert": 4096,
      "columns": 2048,
      "fixed_storage_bits": 160,
      "reference_bpw": [3.0],
      "candidates": [
        {
          "expert_id": 0,
          "profile": "nint4",
          "precision": {
            "family": "NINT4",
            "nint_spec": {"bits": 4, "groupsize": 24, "sub_bits": 6}
          },
          "variable_storage_bits": 35000000,
          "pool_key": "nint4",
          "pool_storage_bits": 512,
          "distortion": 0.0012,
          "validation_distortion": 0.0013,
          "effective_bpw": 4.173
        }
      ]
    }
  ]
}
```

`reference_bpw` must contain one value per expert. Experts may expose different
candidate sets, allowing a protected expert to exclude low-precision choices.
Within one tensor, candidates sharing a `pool_key` must have identical
precision descriptors and `pool_storage_bits`. Pool charges are scoped to
that tensor.

## Budget and TPQ-shape constraints

Format: `mfq.ew-budget.v1`.

Every rate scope accepts either `target_bits`/`tolerance_bits`,
`target_bpw`/`tolerance_bpw`, or explicit `min_*` and `max_*` bounds. Layer
budgets are optional. Projection budgets prevent a whole-model optimizer from
removing too much precision from Down.

```json
{
  "format": "mfq.ew-budget.v1",
  "target_profile": "EW-S",
  "model_weight_count": 100000000000,
  "model_fixed_storage_bits": 160000000000,
  "total": {"target_bpw": 2.2, "tolerance_bpw": 0.00001},
  "projections": {
    "down": {"min_bpw": 1.950891, "max_bpw": 2.05}
  },
  "layers": {
    "0": {"min_bpw": 1.9, "max_bpw": 2.3}
  },
  "shape_constraints": [
    {
      "name": "down-tpq-spikes",
      "projection": "down",
      "peak_reference_min_bpw": 3.0,
      "peak_selected_min_bpw": 3.0,
      "min_peak_retention": 1.0,
      "min_contrast_ratio": 1.0,
      "histogram": [
        {
          "side": "ge",
          "threshold_bpw": 3.0,
          "relative_tolerance": 0.02,
          "absolute_tolerance": 0
        },
        {
          "side": "le",
          "threshold_bpw": 1.375,
          "relative_tolerance": 0.02,
          "absolute_tolerance": 0
        }
      ]
    }
  ]
}
```

A shape constraint selects all matching experts by `projection` and optional
`layer`:

- peak retention keeps a requested fraction of reference-peak identities at
  or above a selected BPW threshold;
- contrast requires the selected peak/body mean-BPW difference to retain a
  fraction of the reference difference;
- histogram constraints retain the number of experts above or below a BPW
  threshold within relative and absolute count tolerances.

The allocation report records the planned whole-model, routed, projection,
and layer BPW, selected family counts, peak retention, contrast ratio, and
reference/selected histogram counts. It also stores SHA-256 identities for
all three inputs and the output scheme.
