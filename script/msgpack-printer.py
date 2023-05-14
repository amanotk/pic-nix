#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import msgpack


def doit(filename):
    with open(filename, "rb") as fp:
        stream = fp.read()
        unpacker = msgpack.Unpacker(None, max_buffer_size=len(stream))
        unpacker.feed(stream)
        for data in unpacker:
            print(json.dumps(data, indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MessagePack Pretty Printer")
    parser.add_argument("filename", nargs=1, help="msgpack format file")

    args = parser.parse_args()
    doit(args.filename[0])
