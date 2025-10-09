# Examples gallery

The scripts under `examples/` demonstrate typical TNFR × LFS workflows.

## Quickstart script

`examples/quickstart.sh` orchestrates the baseline → analysis → report flow using
the bundled datasets. It is the same target invoked by `make quickstart`.

## Ingest and recommend

`examples/ingest_and_recommend.py` loads a telemetry file, runs the analysis
pipeline and prints the recommended setup adjustments. Use it as a starting point
for automation.

## Export to CSV

`examples/export_to_csv.py` converts TNFR telemetry into CSV, showcasing how to
reuse the reader and exporter utilities from your own scripts.

## Next steps

* [Tutorials](tutorials.md)
* [API reference](api_reference.md)
* [CLI guide](cli.md)
