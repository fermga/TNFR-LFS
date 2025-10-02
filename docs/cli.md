# Command Line Interface

Install the project in editable mode and run:

```
pip install -e .
tnfr-lfs telemetry.csv --export csv
```

The CLI ingests the telemetry file, computes EPI metrics, and prints the
selected export format to stdout.
