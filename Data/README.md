# Fixed experiment inputs

- `math.json`: the existing 500-problem MATH subset, now versionable so fresh
  checkouts use the same input. Its problem set was checked against the
  archived main and matched-control outputs. Source: the
  `qwedsacf/competition_math` Hugging Face mirror of MATH. The upstream
  dataset is distributed under the MIT License; see [MATH_LICENSE](MATH_LICENSE).
  Selection is deterministic and proportionally stratified by topic and
  difficulty. This is not a new random sample on each run.
- `dataset_b.json`: the ten fixed author-written student messages.

Input SHA-256 checksums:

```text
b8163e13343dbe05dfe504a7a22b8ab1d460d891cfcc97340bc7145168df6dd9  math.json
4edc72322b8767761f9e8c60d4d977d22a8d4bf32505d88fb3ba01fa4751ed24  dataset_b.json
```

`scripts/prepare_math_dataset.py` can rebuild a dataset from the mirror, but
the remote source is not revision-pinned. Prefer the bundled input. Use
`--output /path/to/new-math.json` when investigating a rebuild rather than
overwriting the reference input. A rebuilt file should be checked before use.
