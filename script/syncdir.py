#!/usr/bin/env python

import os
import sys
import shutil
import signal
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
from logging import basicConfig, getLogger

NODEDIR_FORMAT = "node{:06d}"

logger = getLogger(__name__)


class OutputHandler(FileSystemEventHandler):
    def __init__(self, src, dst, executor):
        self.src_dir = Path(src)
        self.dst_dir = Path(dst)
        self.executor = executor
        self.future = list()

    def cleanup_future(self):
        for item in self.future:
            src, dst, future = item
            if future.done():
                self.future.remove(item)
                logger.info(f"File successfully moved from {src} to {dst}")

    def on_closed(self, event):
        src_path = Path(event.src_path)
        if not event.is_directory:
            logger.info(f"File created: {src_path}")
            rel_path = src_path.relative_to(self.src_dir)
            dst_path = self.dst_dir / rel_path
            # run copy_file in a separate thread
            future = self.executor.submit(copy_file, src_path, dst_path)
            self.future.append((src_path, dst_path, future))

    def on_created(self, event):
        src_path = Path(event.src_path)
        if event.is_directory:
            logger.info(f"Directory created: {src_path}")
            # sync directory structure
            rel_path = src_path.relative_to(self.src_dir)
            dst_path = self.dst_dir / rel_path
            dst_path.mkdir(parents=True, exist_ok=True)


def copy_file(src_path, dst_path):
    try:
        # make sure the destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # copy file
        shutil.copy(src_path, dst_path)

        # remove the source file
        if src_path.is_file():
            src_path.unlink()
    except Exception as e:
        logger.error(f"Error copying {src_path} to {dst_path}: {e}")


def setup(filename, rank):
    import json
    import toml

    # temporary directory
    tempdir = os.environ.get("PICNIX_TMPDIR", None)
    if tempdir is None:
        return None, None

    # node directory
    nodedir = NODEDIR_FORMAT.format(rank)

    # read configuration file
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            config = json.load(f)
    elif filename.endswith(".toml"):
        with open(filename, "r") as f:
            config = toml.load(f)
    else:
        raise ValueError("Unsupported configuration file format")

    orig_basedir = config["application"].get("basedir", ".")
    temp_basedir = os.sep.join([tempdir, orig_basedir])

    src_dir = os.sep.join([temp_basedir, nodedir])
    dst_dir = os.sep.join([orig_basedir, nodedir])

    return src_dir, dst_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Configuration file",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source directory",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Destination directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="syncdir",
        help="Log file prefix",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Rank of node",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=4,
        help="Number of threads",
    )
    args = parser.parse_args()

    # source and destination directories
    if args.src is not None and args.dst is not None:
        # use the specified directories if both are provided
        src_dir = args.src
        dst_dir = args.dst
    else:
        # otherwise, use the configuration file
        src_dir, dst_dir = setup(args.config, args.rank)
        if src_dir is None or dst_dir is None:
            print("Error: PICNIX_TMPDIR is not set")
            sys.exit(1)

    # make sure that the directories exist
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # logging
    logfile = os.sep.join([dst_dir, args.log]) + "{:06d}.txt".format(args.rank)
    basicConfig(
        filename=logfile,
        format="%(asctime)s [%(levelname)s] %(message)s",
        level="INFO",
    )

    # create observer
    observer = Observer()

    # signal handler
    def signal_handler(signum, frame):
        observer.stop()
        observer.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # main event handler with thread pool
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
