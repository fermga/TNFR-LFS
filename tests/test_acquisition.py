from io import StringIO

import pytest

from tnfr_lfs.acquisition.outsim_client import OutSimClient, TelemetryFormatError


def test_ingest_validates_header():
    client = OutSimClient()
    data = StringIO(
        "timestamp,vertical_load,slip_ratio,lateral_accel,longitudinal_accel,nfr,si\n"
        "0.0,6000,0.05,1.2,0.4,520,0.82\n"
    )
    records = client.ingest(data)
    assert len(records) == 1


def test_ingest_rejects_invalid_header():
    client = OutSimClient()
    data = StringIO("bad,columns\n1,2\n")
    with pytest.raises(TelemetryFormatError):
        client.ingest(data)
