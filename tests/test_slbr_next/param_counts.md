# `NAFConvNeXtBlock` parameter counts

Per-block counts, `dw_kernel_size=5`, SCA without bias. Generated from

```text
spatial-only = 4C^2 + 2Ck^2 + 7C
with mixer   = 7C^2 + 2Ck^2 + 12C
```

and asserted by `tests/test_slbr_next/test_blocks.py::test_parameter_count_report`.

| channels | channel_mixer | parameters |
|---------:|:-------------:|-----------:|
| 32       | no            | 5,920      |
| 32       | yes           | 9,152      |
| 64       | no            | 20,032     |
| 64       | yes           | 32,640     |
| 128      | no            | 72,832     |
| 128      | yes           | 122,624    |

For the T4 full-resolution 3x3 depthwise variant, subtract
`2C(5^2 - 3^2) = 32C` from the spatial-only count:

| channels | channel_mixer | k=3 parameters |
|---------:|:-------------:|---------------:|
| 32       | no            | 4,896          |
| 32       | yes           | 8,128          |
| 64       | no            | 17,984         |
| 64       | yes           | 30,592         |
| 128      | no            | 68,736         |
| 128      | yes           | 118,528        |
