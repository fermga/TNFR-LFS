# Benchmark guide

The TNFR × LFS benchmark suite helps track the performance of ΔNFR and ν_f
caches before shipping changes.

## Installation

Install the optional extras or use the Makefile helper:

```bash
python -m pip install -e .[benchmark]
# or
make benchmark-delta-cache
```

## Running the delta cache benchmark

Execute `python -m benchmarks.delta_cache_benchmark` (or `make` target above) to
measure cached versus uncached runs against the bundled Replay Analyzer capture
`data/test1.zip`. The default profile samples 128 records and performs two
passes per repeat, yielding the current baseline of roughly 0.0106 s per
uncached pass versus 0.0089 s with warm caches (≈×1.2 speed-up).

Use `--sample-limit` to stretch the test—each additional sample increases the
spectral analysis workload linearly. The CLI accepts CSV, RAF or Replay Analyzer
bundles, mirroring the behaviour of the standard commands, so you can validate
optimisations against your own telemetry recordings.

## More performance tools

* [Makefile helpers](../Makefile)
* [`benchmarks/` package](../benchmarks)
* [Development workflow](DEVELOPMENT.md)
