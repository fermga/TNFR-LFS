"""Quick UDP sniffer for OutSim/OutGauge — independent of our pipeline.

Run with LFS already streaming:
    .\.venv\Scripts\python.exe scripts\sniff_udp.py
Press Ctrl-C to stop. Should print packet sizes within ~1 second.
"""
from __future__ import annotations

import select
import socket
import sys
import time

OUTSIM_PORT = 30000
OUTGAUGE_PORT = 30001


def open_udp(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    s.bind(("0.0.0.0", port))
    return s


def main() -> int:
    sims = open_udp(OUTSIM_PORT)
    gauges = open_udp(OUTGAUGE_PORT)
    seen = {OUTSIM_PORT: {"count": 0, "sizes": set()},
            OUTGAUGE_PORT: {"count": 0, "sizes": set()}}
    print(f"Listening on UDP {OUTSIM_PORT} (OutSim) and "
          f"{OUTGAUGE_PORT} (OutGauge). Ctrl-C to stop.")
    t_start = time.time()
    t_last = t_start
    try:
        while True:
            ready, _, _ = select.select([sims, gauges], [], [], 0.5)
            for s in ready:
                data, _addr = s.recvfrom(4096)
                port = s.getsockname()[1]
                seen[port]["count"] += 1
                seen[port]["sizes"].add(len(data))
            now = time.time()
            if now - t_last >= 1.0:
                t_last = now
                el = now - t_start
                os_n = seen[OUTSIM_PORT]["count"]
                og_n = seen[OUTGAUGE_PORT]["count"]
                print(f"[{el:5.1f}s] OutSim: {os_n:5d} pkts "
                      f"sizes={sorted(seen[OUTSIM_PORT]['sizes'])} | "
                      f"OutGauge: {og_n:5d} pkts "
                      f"sizes={sorted(seen[OUTGAUGE_PORT]['sizes'])}")
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
