"""Generate a tiny interface fixture, not training or benchmark evaluation data."""
import argparse
import json
from pathlib import Path
import struct
import zlib


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def fixture_png(index):
    width, height = 32, 24
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            square = 2 + 2 * index <= x < 10 + 2 * index and 8 <= y < 16
            rows.extend((240, 80, 30) if square else (20, 30, 50))
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(bytes(rows))) + _chunk(b"IEND", b""))


def create_fixture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    for index in range(9):
        (output / "frames" / f"{index:06d}.png").write_bytes(fixture_png(index))
    anchors, targets = [0, 8], list(range(1, 8))
    sample = {"id": "interface-fixture", "schema_version": 2, "width": 32, "height": 24,
              "frame_count": 9,
              "input_frames": [{"index": i, "path": f"frames/{i:06d}.png"} for i in anchors],
              "target_indices": targets,
              "reference_frames": [{"index": i, "path": f"frames/{i:06d}.png"} for i in targets],
              "provenance": {"kind": "procedural interface fixture", "benchmark_data": False}}
    path = output / "samples.jsonl"
    path.write_text(json.dumps(sample) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory, outside data/ to avoid accidental publication")
    args = parser.parse_args(argv)
    print(create_fixture(args.output))


if __name__ == "__main__":
    main()
