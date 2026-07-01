# syncdir

`picnix-syncdir` watches a source directory and copies completed files to a
destination directory.  It was originally written for large PIC-NIX jobs where
each rank writes to node-local storage and files are later moved back to shared
storage.

## Status

This tool is experimental.  It has not been heavily used in production runs.
Prefer normal direct output to the shared filesystem unless node-local output
is clearly needed.

Use this page as a design note and starting point, not as a guarantee that the
tool is robust for every supercomputer environment.

## Basic Usage

Use explicit source and destination directories:

```sh
picnix-syncdir --rank 0 --src /tmp/run/node000000 --dst data/node000000
```

Or derive paths from a PIC-NIX config and `PICNIX_TMPDIR`:

```sh
export PICNIX_TMPDIR=/tmp/my-run
picnix-syncdir --rank 0 --config config.toml
```

In config-driven mode, the destination base directory is read from
`application.basedir`, and the source base directory is
`$PICNIX_TMPDIR/<basedir>`.

## Options

| Option | Meaning |
|--------|---------|
| `--rank` | Rank number used to choose `nodeXXXXXX`. Required. |
| `--config` | TOML or JSON config file. Default: `config.toml`. |
| `--src` | Explicit source directory. Must be used with `--dst`. |
| `--dst` | Explicit destination directory. Must be used with `--src`. |
| `--log` | Log file prefix. Default: `syncdir`. |
| `--nthread` | Number of copy worker threads. Default: `4`. |

## Behavior

When a file is closed in the source directory, `syncdir` copies it to the
matching relative path under the destination directory.  After a successful
copy, the source file is removed.

New directories under the source tree are also mirrored under the destination
tree.

## Caveats

- The tool watches file close events through `watchdog`; behavior can depend on
  the filesystem and platform.
- It removes source files after copy, so test with a small run first.
- It does not currently verify checksums after copy.
- It should be treated carefully on shared production data.
