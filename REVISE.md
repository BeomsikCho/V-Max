# Evaluation memory revisions

## Why this change was made

`vmax/scripts/evaluate/evaluate.py` previously retained every scenario's metric in
Python lists until all 85,126 scenarios had completed. Some values could remain
as JAX arrays, retaining device-buffer objects. RSS was observed growing rapidly
during `bscho_evaluate.sh`, even after Waymax prefetch was changed to `prefetch(1)`.

## Changes

- Transfer each completed metric batch to host memory once with `jax.device_get`.
- Convert every stored metric and episode length to a Python `float` or `int`.
- Append each batch immediately to `evaluation_episodes.csv`.
- Retain only per-metric running sums and counts for the final summary.
- Preserve the existing CSV columns, scenario index, `evaluation_results.txt`, and
  console summary.

The separate Waymax checkout was already manually changed from
`prefetch(AUTOTUNE)` to `prefetch(1)` and is not part of this repository's diff.

## How to revert

Revert the changes to:

- `vmax/scripts/evaluate/evaluate.py`
- `vmax/scripts/evaluate/utils.py`

Then remove this `REVISE.md` file. If the Waymax prefetch change should also be
reverted, restore this line in
`/workspace/winners-plan/libs/waymax/waymax/dataloader/dataloader_utils.py`:

```python
return dataset.prefetch(AUTOTUNE)
```

## Validation

Run syntax/tests first, then execute `./bscho_evaluate.sh` while sampling the
evaluation Python process RSS. Record the observed result here after validation.

### Validation run 1: streaming metrics only

- Command: `timeout --signal=INT 180s ./bscho_evaluate.sh`
- Batch size: 64; Waymax final prefetch: 1; upstream parallel calls: AUTOTUNE.
- Processed 29,440 scenarios before the intentional timeout.
- RSS grew from 20,823,080 KiB at 9 seconds to 30,414,140 KiB at 178 seconds.
- Conclusion: streaming successfully wrote results incrementally, but metric retention
  was not the main source of the large RSS growth.

### Additional Waymax data-pipeline change

The remaining upstream queues were limited for a controlled comparison:

- TFRecord `interleave`: `num_parallel_calls=1`, `cycle_length=1`
- preprocessing `map`: `num_parallel_calls=1`
- `batch`: `num_parallel_calls=1`
- outer dataset `interleave`: `num_parallel_calls=1`
- final `prefetch`: remains `1`

To revert these external changes, replace the fixed parallel-call values with
`AUTOTUNE` and restore `cycle_length=AUTOTUNE` in the Waymax file named above.

### Validation run 2: streaming metrics plus fixed pipeline parallelism

- Command: `timeout --signal=INT 180s ./bscho_evaluate.sh`
- Processed 32,576 scenarios before the intentional timeout.
- Incremental CSV contained 32,576 data rows plus its header after interruption.
- RSS warmed from 8,835,272 KiB at 7 seconds to roughly 13 GiB.
- From 130 through 160 seconds it stayed approximately flat: 13,224,804,
  13,231,072, 13,307,788, and 13,215,624 KiB.
- Compared with run 1, peak observed RSS fell from 30,414,140 KiB to
  13,374,680 KiB, and the continuous upward trend stopped after warmup.
- Throughput remained about 200 scenarios/second after warmup.

Both validation runs were intentionally interrupted after three minutes, so they
produced a valid partial `evaluation_episodes.csv` but no final aggregate result.
