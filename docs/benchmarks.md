# Benchmark guide

The TNFR × LFS benchmark suite helps track the performance of ΔNFR and ν_f
caches before shipping changes.

## Installation

Install the optional extras or use the Makefile helper. Include the ``spectral``
extra when you want to exercise the Goertzel backend shipped with the dominant
frequency utilities:

```bash
python -m pip install -e .[benchmark]
# Dominant frequency acceleration (SciPy Goertzel helper)
python -m pip install -e .[benchmark,spectral]
# or
make benchmark-delta-cache
```

## Running the delta cache benchmark

Execute `python -m benchmarks.delta_cache_benchmark` (or `make` target above) to
measure cached versus uncached runs against the bundled Replay Analyzer capture
`data/test1.zip`. The default profile samples 128 records and performs two
passes per repeat, yielding the current baseline of roughly 0.0106 s per
uncached pass versus 0.0089 s with warm caches (≈×1.2 speed-up).

Use `--sample-limit` to stretch the test—each additional sample now feeds an
``O(n \log n)`` FFT-based spectral analysis pipeline, delivering sub-quadratic
scaling even as the window grows. The CLI accepts CSV, RAF or Replay Analyzer
bundles, mirroring the behaviour of the standard commands, so you can validate
optimisations against your own telemetry recordings.

## Dominant frequency accelerator

The ``benchmarks/spectrum_goertzel_benchmark.py`` module compares the dense FFT
cross-spectrum against the sparse Goertzel helper introduced in
``tnfr_lfs.core.spectrum``. When the analysis only needs the dominant bin, the
Goertzel path evaluates a fixed list of candidate frequencies in ``O(n·k)``
operations. With ``k`` constant, this becomes linear in the number of samples,
which translates into faster feedback for high-resolution stints. Install the
``spectral`` extra before running the benchmark so SciPy's implementation is
available:

```bash
python -m pytest benchmarks/spectrum_goertzel_benchmark.py -k goertzel
```

## More performance tools

* [Makefile helpers](../Makefile)
* [`benchmarks/` package](../benchmarks)
* [Development workflow](DEVELOPMENT.md)
