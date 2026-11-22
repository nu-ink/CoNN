"""Paired streams derived from one seed.

Offsets name the stream. They are not per-condition seeds. Every condition
in a paired seed uses the same four integers, so mask draws cannot shift
initialization or the dataset.
"""

from __future__ import annotations

STREAM_OFFSET = {
    "data": 0,
    "init": 1_000_003,
    "order": 2_000_003,
    "mask": 3_000_003,
}


def paired_seed(seed: int, stream: str) -> int:
    if stream not in STREAM_OFFSET:
        raise KeyError(stream)
    return int(seed) + STREAM_OFFSET[stream]
