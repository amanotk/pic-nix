#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from logging import basicConfig, getLogger
from pathlib import Path

import toml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

NODEDIR_FORMAT = "node{:06d}"

logger = getLogger(__name__)


class OutputHandler(FileSystemEventHandler):
    def __init__(self, src, dst, executor):
        self.src_dir = Path(src)
        self.dst_dir = Path(dst)
        self.executor = executor
        self.future = []

    def cleanup_future(self):
        pending = []
        for src, dst, future in self.future:
            if future.done():
                logger.info("File successfully moved from %s to %s", src, dst)
            else:
                pending.append((src, dst, future))
        self.future = pending

    def on_closed(self, event):
        src_path = Path(event.src_path)
        if not event.is_directory:
            logger.info("File created: %s", src_path)
            rel_path = src_path.relative_to(self.src_dir)
            dst_path = self.dst_dir / rel_path
            future = self.executor.submit(copy_file, src_path, dst_path)
            self.future.append((src_path, dst_path, future))

    def on_created(self, event):
        src_path = Path(event.src_path)
        if event.is_directory:
            logger.info("Directory created: %s", src_path)
            rel_path = src_path.relative_to(self.src_dir)
            dst_path = self.dst_dir / rel_path
            dst_path.mkdir(parents=True, exist_ok=True)


def copy_file(src_path, dst_path):
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dst_path)
        if src_path.is_file():
            src_path.unlink()
    except Exception as error:
        logger.error("Error copying %s to %s: %s", src_path, dst_path, error)


def load_config(filename):
    path = Path(filename)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    if path.suffix == ".toml":
        return toml.load(path)
    raise ValueError("Unsupported configuration file format")


def setup(filename, rank):
    tempdir = os.environ.get("PICNIX_TMPDIR")
    if tempdir is None:
        return None, None

    config = load_config(filename)
    orig_basedir = config["application"].get("basedir", ".")
    temp_basedir = Path(tempdir) / orig_basedir
    nodedir = NODEDIR_FORMAT.format(rank)

    return temp_basedir / nodedir, Path(orig_basedir) / nodedir


def run(args):
    if args.src is not None and args.dst is not None:
        src_dir = Path(args.src)
        dst_dir = Path(args.dst)
    else:
        src_dir, dst_dir = setup(args.config, args.rank)
        if src_dir is None or dst_dir is None:
            print("Error: PICNIX_TMPDIR is not set")
            return 1

    src_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    logfile = dst_dir / f"{args.log}{args.rank:06d}.txt"
    basicConfig(
        filename=logfile,
        format="%(asctime)s [%(levelname)s] %(message)s",
        level="INFO",
    )

    observer = Observer()

    def signal_handler(signum, frame):
        observer.stop()
        observer.join()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with ThreadPoolExecutor(max_workers=args.nthread) as executor:
        event_handler = OutputHandler(src_dir, dst_dir, executor)
        observer.schedule(event_handler, src_dir, recursive=True)
        observer.start()

        try:
            while observer.is_alive():
                observer.join(1)
                event_handler.cleanup_future()
        finally:
            observer.stop()
            observer.join()

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sync PIC-NIX node output directories")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="configuration file",
    )
    parser.add_argument("--src", help="source directory")
    parser.add_argument("--dst", help="destination directory")
    parser.add_argument("--log", default="syncdir", help="log file prefix")
    parser.add_argument("--rank", type=int, required=True, help="rank of node")
    parser.add_argument("--nthread", type=int, default=4, help="number of threads")

    args = parser.parse_args(argv)
    if (args.src is None) != (args.dst is None):
        parser.error("--src and --dst must be specified together")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
