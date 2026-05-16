"""LTTB decimation, lifted from the Dash viewer's ``app.figures`` module.

Same algorithm, same constants, same NaN handling — kept independent so
the studio package has zero dependency on the Dash app code (which we
will delete once the Studio reaches parity).
"""

from __future__ import annotations

import numpy as np

DECIMATE_THRESHOLD = 6_000
DECIMATE_TARGET = 1_500


def lttb(
    x: np.ndarray, y: np.ndarray, n_out: int = DECIMATE_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsample, NaN-safe.

    Preserves visual peaks/troughs better than naive striding (matters
    for slip spikes, brake bites, gear-shift edges). Bucket means are
    computed once via ``np.add.reduceat`` so the Python loop only does
    the per-bucket triangle-area argmax.
    """
    n = x.size
    if n_out >= n or n_out < 3:
        return x, y
    yf = y.astype(float, copy=True)
    mask = np.isnan(yf)
    if mask.any():
        idx = np.arange(n)
        if (~mask).any():
            yf[mask] = np.interp(idx[mask], idx[~mask], yf[~mask])
        else:
            yf[mask] = 0.0
    xf = x.astype(float, copy=False)
    bucket_size = (n - 2) / (n_out - 2)
    edges = (np.arange(n_out - 1) * bucket_size).astype(np.intp) + 1
    full_edges = np.concatenate((edges, np.array([n], dtype=np.intp)))
    counts = np.diff(full_edges).astype(float)
    bucket_x = np.add.reduceat(xf, full_edges[:-1]) / counts
    bucket_y = np.add.reduceat(yf, full_edges[:-1]) / counts

    sampled_x = np.empty(n_out, dtype=x.dtype)
    sampled_y = np.empty(n_out, dtype=y.dtype)
    sampled_x[0] = x[0]
    sampled_y[0] = y[0]
    a = 0
    last = n_out - 2
    for i in range(last):
        cur_lo = int(full_edges[i])
        cur_hi = int(full_edges[i + 1])
        if i + 1 < last:
            avg_x = bucket_x[i + 1]
            avg_y = bucket_y[i + 1]
        else:
            avg_x = float(x[-1])
            avg_y = float(yf[-1])
        ax = float(xf[a])
        ay = float(yf[a])
        xs = xf[cur_lo:cur_hi]
        ys = yf[cur_lo:cur_hi]
        area = np.abs((ax - avg_x) * (ys - ay) - (ax - xs) * (avg_y - ay))
        a = cur_lo + int(np.argmax(area))
        sampled_x[i + 1] = x[a]
        sampled_y[i + 1] = y[a]
    sampled_x[-1] = x[-1]
    sampled_y[-1] = y[-1]
    return sampled_x, sampled_y


def maybe_decimate(
    x: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply LTTB only when the series exceeds the threshold."""
    if x.size <= DECIMATE_THRESHOLD:
        return x, y
    return lttb(x, y, DECIMATE_TARGET)


__all__ = ["DECIMATE_TARGET", "DECIMATE_THRESHOLD", "lttb", "maybe_decimate"]
