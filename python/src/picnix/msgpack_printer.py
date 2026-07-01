#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgpack


def json_default(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def print_msgpack(filename):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("rb") as fp:
        unpacker = msgpack.Unpacker(fp, raw=False, strict_map_key=False)
        for data in unpacker:
            print(json.dumps(data, indent=4, default=json_default))


def main(argv=None):
    parser = argparse.ArgumentParser(description="MessagePack Pretty Printer")
    parser.add_argument("filename", help="msgpack format file")

    args = parser.parse_args(argv)
    print_msgpack(args.filename)


if __name__ == "__main__":
    main()
