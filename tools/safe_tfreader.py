"""Minimal TFRecord + tensorboard Event parser (no tensorboard/protobuf dependency).

TensorBoard event files are TFRecord streams:
    uint64 length | uint32 crc32c(length) | bytes data | uint32 crc32c(data)
`data` is a serialized `Event` protobuf:
    field 1 (double) wall_time
    field 2 (int64)  step
    field 5 (message) Summary { repeated Value value = 1 }
`Summary.Value`:
    field 1 (string) tag
    field 2 (float)  simple_value
    field 3 (message) histo
    field 8 (float)  tensor-ish scalars in newer writers (not needed here)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        field = key >> 3
        wire = key & 0x7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, value


def _read_records(path: Path, limit_bytes: int | None = None):
    with path.open("rb") as fh:
        consumed = 0
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return
            (length,) = struct.unpack("<Q", head)
            if length == 0 or length > 64 << 20:
                return
            crc = fh.read(4)
            if len(crc) < 4:
                return
            data = fh.read(length)
            if len(data) < length:
                return
            tail = fh.read(4)
            if len(tail) < 4:
                return
            consumed += 16 + length
            yield data
            if limit_bytes is not None and consumed >= limit_bytes:
                return


def read_scalars(path: Path, limit_bytes: int | None = None) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} for simple-value scalars."""
    out: dict[str, list[tuple[int, float]]] = {}
    for data in _read_records(path, limit_bytes):
        step = 0
        wall = 0.0
        summary = None
        for field, wire, value in _fields(data):
            if field == 1 and wire == 1:
                (wall,) = struct.unpack("<d", value)
            elif field == 2 and wire == 0:
                step = value
            elif field == 5 and wire == 2:
                summary = value
        if summary is None:
            continue
        for field, wire, value in _fields(summary):
            if field != 1 or wire != 2:
                continue
            tag = None
            simple = None
            for vf, vw, vv in _fields(value):
                if vf == 1 and vw == 2:
                    tag = vv.decode("utf-8", "replace")
                elif vf == 2 and vw == 5:
                    (simple,) = struct.unpack("<f", vv)
            if tag is not None and simple is not None:
                out.setdefault(tag, []).append((step, simple))
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        scalars = read_scalars(p)
        print(f"# {p}")
        print(f"  records of tags: {len(scalars)}")
        for tag, series in sorted(scalars.items()):
            steps = [s for s, _ in series]
            vals = [v for _, v in series]
            print(
                f"  {tag:40s} n={len(series):5d} step=[{min(steps)},{max(steps)}] "
                f"last={vals[-1]:.4f} min={min(vals):.4f} max={max(vals):.4f}"
            )
