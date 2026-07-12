#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import msgpack
import numpy as np
import tqdm

from picnix import DEFAULT_LOG_PREFIX


PHASES = ("diagnostic", "push", "rebalance")
DEFAULT_MAX_PLOT_POINTS = 5000
DEFAULT_PLOT_BINS = 100
READ_CHUNK_BYTES = 1024 * 1024


def iter_msgpack_records(filename, progress=False):
    path = Path(filename)
    progress_bar = None
    if progress:
        progress_bar = tqdm.tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {path.name}",
            file=sys.stderr,
        )

    try:
        with path.open("rb") as fp:
            unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
            while True:
                chunk = fp.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                unpacker.feed(chunk)
                if progress_bar is not None:
                    progress_bar.update(len(chunk))
                for record in unpacker:
                    if isinstance(record, dict):
                        yield record
    finally:
        if progress_bar is not None:
            progress_bar.close()


def read_msgpack_stream(filename):
    return list(iter_msgpack_records(filename))


def resolve_log_filename(filename):
    path = Path(filename)
    if path.name != "profile.msgpack":
        return path

    with path.open("rb") as fp:
        profile = msgpack.load(fp, raw=False, strict_map_key=False)

    config = profile.get("configuration", {})
    log_config = config.get("application", {}).get("log", {})
    log_path = log_config.get("path", ".")
    prefix = log_config.get("prefix", DEFAULT_LOG_PREFIX)
    return path.parent / log_path / f"{prefix}.msgpack"


def extract_timing_rows(records):
    rows = []
    for index, record in enumerate(records):
        row = {
            "index": index,
            "step": record.get("step"),
            "timestamp_unixtime": None,
            "diagnostic_elapsed": None,
            "push_elapsed": None,
            "rebalance_elapsed": None,
            "rebalance_status": None,
            "total_elapsed": 0.0,
        }

        timestamp = record.get("timestamp")
        if isinstance(timestamp, dict):
            row["timestamp_unixtime"] = timestamp.get("unixtime")

        for phase in PHASES:
            value = record.get(phase)
            if not isinstance(value, dict) or "elapsed" not in value:
                continue
            elapsed = float(value["elapsed"])
            row[f"{phase}_elapsed"] = elapsed
            row["total_elapsed"] += elapsed

        rebalance = record.get("rebalance")
        if isinstance(rebalance, dict) and "status" in rebalance:
            row["rebalance_status"] = bool(rebalance["status"])

        if row["step"] is not None or any(
            row[f"{phase}_elapsed"] is not None for phase in PHASES
        ):
            rows.append(row)

    return rows


def phase_values(rows, phase):
    key = f"{phase}_elapsed"
    return np.array(
        [row[key] for row in rows if row[key] is not None], dtype=np.float64
    )


def summarize_values(values):
    if values.size == 0:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(values.size),
        "total": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def summarize_phases(rows):
    summary = {phase: summarize_values(phase_values(rows, phase)) for phase in PHASES}
    measured_total = sum(item["total"] for item in summary.values())
    for item in summary.values():
        item["percent"] = (
            100.0 * item["total"] / measured_total if measured_total > 0 else 0.0
        )
    return summary


def summarize_rebalance(rows):
    active = np.array(
        [
            row["rebalance_elapsed"]
            for row in rows
            if row["rebalance_status"] is True and row["rebalance_elapsed"] is not None
        ],
        dtype=np.float64,
    )
    inactive = np.array(
        [
            row["rebalance_elapsed"]
            for row in rows
            if row["rebalance_status"] is False and row["rebalance_elapsed"] is not None
        ],
        dtype=np.float64,
    )
    return {"active": summarize_values(active), "inactive": summarize_values(inactive)}


def worst_phase_steps(rows, top):
    items = []
    for row in rows:
        for phase in PHASES:
            elapsed = row[f"{phase}_elapsed"]
            if elapsed is not None:
                items.append((elapsed, row["step"], phase))
    items.sort(key=lambda item: item[0], reverse=True)
    return items[:top]


def format_seconds(value):
    return f"{value:.6g}"


def format_summary_value(value):
    value = float(value)
    if value == 0.0:
        return "0.000"
    if abs(value) >= 100000.0 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.3f}"


def format_report(rows, log_filename, top=10):
    phase_summary = summarize_phases(rows)
    rebalance_summary = summarize_rebalance(rows)

    lines = [f"Log: {log_filename}", f"Records: {len(rows)}", "", "Phase Summary"]
    lines.append(
        "phase             total[s]   percent     mean[ms]   median[ms]      p90[ms]      p99[ms]      max[ms]      count"
    )
    for phase in PHASES:
        item = phase_summary[phase]
        lines.append(
            f"{phase:<12}"
            f"{format_seconds(item['total']):>14}"
            f"{item['percent']:>9.2f}"
            f"{format_summary_value(1000.0 * item['mean']):>13}"
            f"{format_summary_value(1000.0 * item['median']):>13}"
            f"{format_summary_value(1000.0 * item['p90']):>12}"
            f"{format_summary_value(1000.0 * item['p99']):>12}"
            f"{format_summary_value(1000.0 * item['max']):>12}"
            f"{item['count']:>11}"
        )

    lines.extend(["", "Rebalance Summary"])
    lines.append("kind            total[s]     mean[ms]      max[ms]      count")
    for kind in ("active", "inactive"):
        item = rebalance_summary[kind]
        lines.append(
            f"{kind:<10}"
            f"{format_seconds(item['total']):>14}"
            f"{format_summary_value(1000.0 * item['mean']):>13}"
            f"{format_summary_value(1000.0 * item['max']):>12}"
            f"{item['count']:>11}"
        )

    if top > 0:
        lines.extend(["", f"Worst {top} Phase Timings"])
        lines.append("step       phase        elapsed[s]")
        for elapsed, step, phase in worst_phase_steps(rows, top):
            lines.append(f"{str(step):<11}{phase:<12}{format_seconds(elapsed):>10}")

    return "\n".join(lines)


def write_csv(rows, filename):
    fields = [
        "index",
        "step",
        "timestamp_unixtime",
        "diagnostic_elapsed",
        "push_elapsed",
        "rebalance_elapsed",
        "rebalance_status",
        "total_elapsed",
    ]
    with Path(filename).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rolling_mean(values, window):
    if window <= 1 or values.size < window:
        return None
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def get_step_axis(rows):
    return np.array(
        [row["step"] if row["step"] is not None else row["index"] for row in rows],
        dtype=np.float64,
    )


def get_phase_axis(rows, phase):
    if phase == "total":
        return np.array([row["total_elapsed"] for row in rows], dtype=np.float64)
    return np.array(
        [
            np.nan if row[f"{phase}_elapsed"] is None else row[f"{phase}_elapsed"]
            for row in rows
        ],
        dtype=np.float64,
    )


def get_step_range(rows):
    x = get_step_axis(rows)
    return float(np.min(x)), float(np.max(x))


def binned_timing(rows, phases=("total", *PHASES), bins=DEFAULT_PLOT_BINS):
    x = np.array(
        [row["step"] if row["step"] is not None else row["index"] for row in rows],
        dtype=np.float64,
    )
    phase_values_by_name = {phase: get_phase_axis(rows, phase) for phase in phases}

    groups = np.array_split(np.arange(len(rows)), min(bins, len(rows)))
    result = {
        "x": [],
        "active_rebalance": [],
    }
    for phase in phases:
        result[phase] = {"p10": [], "median": [], "p90": [], "max": []}

    for group in groups:
        result["x"].append(float(np.median(x[group])))
        for phase in phases:
            values = phase_values_by_name[phase][group]
            values = values[np.isfinite(values)]
            if values.size == 0:
                p10 = median = p90 = max_value = np.nan
            else:
                p10 = float(np.percentile(values, 10))
                median = float(np.median(values))
                p90 = float(np.percentile(values, 90))
                max_value = float(np.max(values))
            result[phase]["p10"].append(p10)
            result[phase]["median"].append(median)
            result[phase]["p90"].append(p90)
            result[phase]["max"].append(max_value)
        result["active_rebalance"].append(
            any(rows[int(index)]["rebalance_status"] is True for index in group)
        )

    converted = {
        "x": np.array(result["x"]),
        "active_rebalance": np.array(result["active_rebalance"]),
    }
    for phase in phases:
        converted[phase] = {
            key: np.array(value) for key, value in result[phase].items()
        }
    return converted


def binned_total_timing(rows, bins=DEFAULT_PLOT_BINS):
    timing = binned_timing(rows, phases=("total",), bins=bins)
    total = timing["total"]
    return {
        "x": timing["x"],
        "p10": total["p10"],
        "median": total["median"],
        "p90": total["p90"],
        "max": total["max"],
        "active_rebalance": timing["active_rebalance"],
    }


def style_timing_axis(ax, xmin, xmax):
    ax.set_xlim(xmin, xmax)
    ax.minorticks_on()
    ax.tick_params(which="major", length=5, width=0.8)
    ax.tick_params(which="minor", length=3, width=0.6)
    ax.grid(True, which="major", alpha=0.24, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)


def plot_binned_rows(rows, filename):
    import matplotlib.pyplot as plt

    timing = binned_timing(rows)
    xmin, xmax = get_step_range(rows)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].fill_between(
        timing["x"],
        timing["total"]["p10"],
        timing["total"]["p90"],
        color="black",
        alpha=0.2,
        label="total p10-p90",
    )
    axes[0].plot(
        timing["x"], timing["total"]["median"], color="black", label="total median"
    )
    axes[0].plot(
        timing["x"],
        timing["total"]["max"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="total max",
    )
    axes[0].plot(
        timing["x"], timing["push"]["median"], color="tab:blue", label="push median"
    )
    axes[0].plot(
        timing["x"],
        timing["push"]["max"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.2,
        label="push max",
    )

    axes[1].plot(
        timing["x"],
        timing["diagnostic"]["median"],
        color="tab:green",
        label="diagnostic median",
    )
    axes[1].plot(
        timing["x"],
        timing["diagnostic"]["max"],
        color="tab:green",
        linestyle="--",
        linewidth=1.2,
        label="diagnostic max",
    )

    axes[2].plot(
        timing["x"],
        timing["rebalance"]["median"],
        color="tab:orange",
        label="rebalance median",
    )
    axes[2].plot(
        timing["x"],
        timing["rebalance"]["max"],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="rebalance max",
    )

    for ax in axes:
        ax.set_ylabel("elapsed [s]")
        style_timing_axis(ax, xmin, xmax)

    axes[0].set_title(f"PIC-NIX timing ({len(timing['x'])} bins)")
    axes[2].set_xlabel("step")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_rows(rows, filename, rolling=50, max_points=DEFAULT_MAX_PLOT_POINTS):
    if len(rows) > max_points:
        plot_binned_rows(rows, filename)
        return

    import matplotlib.pyplot as plt

    x = get_step_axis(rows)
    xmin, xmax = get_step_range(rows)
    total = get_phase_axis(rows, "total")
    push = get_phase_axis(rows, "push")
    diagnostic = get_phase_axis(rows, "diagnostic")
    rebalance = get_phase_axis(rows, "rebalance")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(x, total, label="total", color="black", linewidth=1.0)
    axes[0].plot(x, push, label="push", color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel("elapsed [s]")
    axes[0].set_title("PIC-NIX phase timing")

    axes[1].plot(x, diagnostic, label="diagnostic", color="tab:green", linewidth=1.0)

    axes[2].plot(x, rebalance, label="rebalance", color="tab:orange", linewidth=1.0)
    smoothed = rolling_mean(total, rolling)
    if smoothed is not None:
        axes[0].plot(
            x[rolling - 1 :],
            smoothed,
            label=f"total rolling mean ({rolling})",
            linewidth=1.5,
        )
    axes[1].set_ylabel("elapsed [s]")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("elapsed [s]")

    for ax in axes:
        style_timing_axis(ax, xmin, xmax)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze PIC-NIX log timing records.")
    parser.add_argument("input", help="log.msgpack or profile.msgpack")
    parser.add_argument("--csv", help="write per-step timing table to CSV")
    parser.add_argument("--plot", help="write timing plot to an image file")
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_PLOT_POINTS,
        help="maximum records for raw line plot before using binned plot (default: 5000)",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=50,
        help="rolling mean window for --plot (default: 50)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of worst phase timings to print (default: 10)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable msgpack read progress bar",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    log_filename = resolve_log_filename(args.input)
    records = iter_msgpack_records(
        log_filename, progress=not args.no_progress and sys.stderr.isatty()
    )
    rows = extract_timing_rows(records)

    if not rows:
        raise SystemExit(f"No timing records found in {log_filename}")

    print(format_report(rows, log_filename, top=args.top))

    if args.csv:
        write_csv(rows, args.csv)
    if args.plot:
        plot_rows(rows, args.plot, rolling=args.rolling, max_points=args.max_points)


if __name__ == "__main__":
    main()
