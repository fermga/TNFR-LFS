"""Example that exports the computed EPI series to CSV."""

from __future__ import annotations

from io import StringIO

from tnfr_lfs.ingestion.live import OutSimClient
from tnfr_lfs.core import EPIExtractor
from tnfr_lfs.exporters import csv_exporter

DATA = """timestamp,vertical_load,slip_ratio,lateral_accel,longitudinal_accel,yaw,pitch,roll,brake_pressure,locking,nfr,si
0.0,7000,0.01,1.1,0.5,0.01,0.0,0.0,0.18,0.0,550,0.82
0.1,7100,0.03,1.0,0.6,0.02,0.0,0.0,0.17,0.0,555,0.80
0.2,7150,0.04,1.2,0.7,0.03,0.0,0.0,0.19,0.0,560,0.78
"""


def main() -> None:
    client = OutSimClient()
    records = client.ingest(StringIO(DATA))
    extractor = EPIExtractor()
    results = extractor.extract(records)
    csv_output = csv_exporter({"series": results})
    print(csv_output)


if __name__ == "__main__":
    main()
