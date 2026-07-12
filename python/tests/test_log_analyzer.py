#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
from copy import deepcopy

import msgpack
import pytest

from picnix import log_analyzer


def write_msgpack_stream(path, records):
    with path.open("wb") as fp:
        for record in records:
            msgpack.pack(record, fp, use_bin_type=True)


def sample_records():
    return [
        {
            "rank": 0,
            "step": 0,
            "diagnostic": {"elapsed": 0.10},
            "push": {"elapsed": 1.00},
            "rebalance": {"elapsed": 0.01, "status": False},
            "timestamp": {"unixtime": 100.0},
        },
        {
            "rank": 0,
            "step": 1,
            "diagnostic": {"elapsed": 0.20},
            "push": {"elapsed": 1.20},
            "rebalance": {"elapsed": 0.50, "status": True},
            "timestamp": {"unixtime": 101.0},
        },
        {
            "rank": 0,
            "step": 2,
            "diagnostic": {"elapsed": 0.30},
            "push": {"elapsed": 1.40},
            "rebalance": {"elapsed": 0.02, "status": False},
            "timestamp": {"unixtime": 102.0},
        },
    ]


def test_extract_timing_rows_and_summaries(tmp_path):
    log = tmp_path / "log.msgpack"
    write_msgpack_stream(log, sample_records())

    records = log_analyzer.iter_msgpack_records(log)
    rows = log_analyzer.extract_timing_rows(records)
    phase_summary = log_analyzer.summarize_phases(rows)
    rebalance_summary = log_analyzer.summarize_rebalance(rows)

    assert [row["step"] for row in rows] == [0, 1, 2]
    assert rows[1]["total_elapsed"] == pytest.approx(1.90)
    assert phase_summary["diagnostic"]["total"] == pytest.approx(0.60)
    assert phase_summary["push"]["total"] == pytest.approx(3.60)
    assert phase_summary["rebalance"]["total"] == pytest.approx(0.53)
    assert rebalance_summary["active"]["count"] == 1
    assert rebalance_summary["active"]["total"] == pytest.approx(0.50)
    assert rebalance_summary["inactive"]["count"] == 2
    assert rebalance_summary["inactive"]["total"] == pytest.approx(0.03)


def test_read_msgpack_stream_keeps_list_compatibility(tmp_path):
    log = tmp_path / "log.msgpack"
    write_msgpack_stream(log, sample_records())

    records = log_analyzer.read_msgpack_stream(log)

    assert [record["step"] for record in records] == [0, 1, 2]


def test_main_streams_msgpack_records(tmp_path, monkeypatch, capsys):
    log = tmp_path / "log.msgpack"
    write_msgpack_stream(log, sample_records())

    def fail_read_msgpack_stream(*args, **kwargs):
        raise AssertionError("main should stream records instead of materializing them")

    monkeypatch.setattr(log_analyzer, "read_msgpack_stream", fail_read_msgpack_stream)

    log_analyzer.main([str(log), "--no-progress"])

    assert "Records: 3" in capsys.readouterr().out


def test_resolve_log_filename_from_profile(tmp_path):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    profile = data / "profile.msgpack"
    with profile.open("wb") as fp:
        msgpack.pack(
            {
                "configuration": {
                    "application": {
                        "log": {
                            "path": "logs",
                            "prefix": "timing",
                        }
                    }
                }
            },
            fp,
            use_bin_type=True,
        )

    assert log_analyzer.resolve_log_filename(profile) == logs / "timing.msgpack"


def test_write_csv(tmp_path):
    rows = log_analyzer.extract_timing_rows(sample_records())
    output = tmp_path / "timing.csv"

    log_analyzer.write_csv(rows, output)

    with output.open(newline="") as fp:
        reader = csv.DictReader(fp)
        csv_rows = list(reader)

    assert reader.fieldnames == [
        "index",
        "step",
        "timestamp_unixtime",
        "diagnostic_elapsed",
        "push_elapsed",
        "rebalance_elapsed",
        "rebalance_status",
        "total_elapsed",
    ]
    assert csv_rows[1]["step"] == "1"
    assert csv_rows[1]["rebalance_status"] == "True"
    assert float(csv_rows[1]["total_elapsed"]) == pytest.approx(1.90)


def test_plot_rows(tmp_path):
    rows = log_analyzer.extract_timing_rows(sample_records())
    output = tmp_path / "timing.png"

    log_analyzer.plot_rows(rows, output, rolling=2)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_binned_total_timing_and_plot(tmp_path):
    records = [deepcopy(record) for _ in range(40) for record in sample_records()]
    for index, record in enumerate(records):
        record["step"] = index
    rows = log_analyzer.extract_timing_rows(records)
    timing = log_analyzer.binned_total_timing(rows, bins=4)
    multi_phase_timing = log_analyzer.binned_timing(rows, bins=4)
    output = tmp_path / "binned.png"

    assert timing["x"].size == 4
    assert timing["median"].size == 4
    assert timing["max"].size == 4
    assert timing["active_rebalance"].any()
    assert multi_phase_timing["push"]["median"].size == 4
    assert multi_phase_timing["diagnostic"]["max"].size == 4
    assert multi_phase_timing["rebalance"]["max"].size == 4
    assert log_analyzer.get_step_range(rows) == (0.0, 119.0)

    log_analyzer.plot_rows(rows, output, max_points=10)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_parser_has_max_points_option():
    parser = log_analyzer.build_parser()
    args = parser.parse_args(["log.msgpack", "--max-points", "100", "--no-progress"])

    assert args.max_points == 100
    assert args.no_progress is True


def test_format_report_contains_expected_sections():
    rows = log_analyzer.extract_timing_rows(sample_records())
    report = log_analyzer.format_report(rows, "log.msgpack", top=2)

    assert "Phase Summary" in report
    assert "Rebalance Summary" in report
    assert "Worst 2 Phase Timings" in report
    assert "diagnostic" in report
    assert "push" in report
    assert "rebalance" in report


def test_format_report_keeps_large_values_separated():
    rows = log_analyzer.extract_timing_rows(
        [
            {
                "step": index,
                "diagnostic": {"elapsed": 219.477725 if index == 99 else 0.000081},
                "push": {"elapsed": 25.008171 if index == 99 else 0.110201},
                "rebalance": {"elapsed": 17.105017 if index == 99 else 0.067230},
            }
            for index in range(100)
        ]
    )

    report = log_analyzer.format_report(rows, "log.msgpack", top=0)

    assert re.search(r"diagnostic\s+.*\s+\d+\.\d{3}\s+2\.195e\+05\s+100", report)
    assert re.search(r"push\s+.*\s+\d+\.\d{3}\s+25008\.171\s+100", report)
    assert re.search(r"rebalance\s+.*\s+\d+\.\d{3}\s+17105\.017\s+100", report)


def test_format_summary_value_uses_scientific_for_extremes():
    assert log_analyzer.format_summary_value(64.286) == "64.286"
    assert log_analyzer.format_summary_value(219477.725) == "2.195e+05"
    assert log_analyzer.format_summary_value(0.0004) == "4.000e-04"
