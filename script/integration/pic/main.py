#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manual integration workflow for PIC simulations.

Subcommands
-----------
  build          Configure and build example executables.
  run            Run a simulation case with deterministic settings.
  analyze        Parse simulation output and produce a numeric summary.
  compare        Compare summary against golden data.
  update-golden  Regenerate golden data from the current run output.
  plots          Generate PNG plots for manual review.
  all            Execute build, run, analyze, and compare in sequence.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import msgpack
import numpy as np
import toml

from cases import CASES, DEFAULT_CASE_NAME, IntegrationCase


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_BASE = REPO_ROOT / "run-integration-pic"

COMPILER_PROFILES = {
    "gcc": {
        "build_dir": RUN_BASE / "build-gcc",
        "cache": REPO_ROOT / "cmake" / "linux-gcc.cmake",
    },
}


def cmd_build(args):
    case = _resolve_case(args.case)
    compiler = _get_compiler_profile(args.compiler)
    build_dir = compiler["build_dir"]
    executable = _get_executable(case, args.compiler)

    print(f"[build] Configuring in {build_dir} ...")
    build_dir.parent.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        "cmake",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(build_dir),
        "-C",
        str(compiler["cache"]),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
    ]
    _run(configure_cmd, cwd=REPO_ROOT)

    print(f"[build] Building target '{case.target}' ...")
    _run(
        ["cmake", "--build", str(build_dir), "--target", case.target, "--parallel"],
        cwd=REPO_ROOT,
    )

    if not executable.exists():
        print(
            f"[build] ERROR: expected executable not found: {executable}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[build] OK: {executable}")


def cmd_run(args):
    case = _resolve_case(args.case)
    executable = _get_executable(case, args.compiler)
    run_dir = case.run_dir
    tmax = args.tmax if args.tmax is not None else case.tmax
    nproc = args.nproc if args.nproc is not None else case.nproc

    if not executable.exists():
        print(f"[run] ERROR: executable not found: {executable}", file=sys.stderr)
        print("[run] Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_case_config(case)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    mpi_cmd = [
        "mpiexec",
        "-n",
        str(nproc),
        str(executable),
        "-c",
        str(config_path),
        "-t",
        str(tmax),
    ]

    print(f"[run] Running {case.name} in {run_dir} ...")
    print(f"[run]   config: {config_path}")
    print(f"[run]   cmd: {' '.join(mpi_cmd)}")
    print("[run]   OMP_NUM_THREADS=1")

    _run(mpi_cmd, cwd=run_dir, env=env)

    data_dir = run_dir / "data"
    if data_dir.exists():
        print(f"[run] OK: output in {data_dir}")
    else:
        print(
            f"[run] WARNING: expected data directory not found: {data_dir}",
            file=sys.stderr,
        )


def cmd_analyze(args):
    case = _resolve_case(args.case)
    summary = analyze_run(case.run_dir, case.Ns)

    out = args.output or str(case.run_dir / "summary.json")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_ndarray_to_list(summary), f, indent=2)
    print(f"[analyze] Wrote {out_path}")


def analyze_run(run_dir, Ns):
    history_path = run_dir / "data" / "history.txt"
    if not history_path.exists():
        print(f"[analyze] ERROR: {history_path} not found", file=sys.stderr)
        sys.exit(1)

    data = _parse_history(history_path, Ns)

    num_steps = len(data["step"])
    total_energy = np.array(data["ene_e"]) + np.array(data["ene_b"])
    for key in data["particle_keys"]:
        total_energy += np.array(data[key])

    peak_index = int(np.argmax(data["ene_e"]))
    peak_ene_e = float(np.max(data["ene_e"]))
    peak_ene_e_step = int(data["step"][peak_index])

    if num_steps >= 2:
        drift = abs(total_energy[-1] - total_energy[0]) / (abs(total_energy[0]) + 1e-30)
    else:
        drift = 0.0

    summary = {
        "step": data["step"],
        "time": data["time"],
        "div_e": data["div_e"],
        "div_b": data["div_b"],
        "ene_e": data["ene_e"],
        "ene_b": data["ene_b"],
        "total_energy": total_energy.tolist(),
        "peak_ene_e": peak_ene_e,
        "peak_ene_e_step": peak_ene_e_step,
        "final_energy_drift": float(drift),
        "Ns": Ns,
        "num_steps": num_steps,
        "particle_keys": data["particle_keys"],
    }
    for key in data["particle_keys"]:
        summary[key] = data[key]

    snapshot = _read_field_snapshot(run_dir)
    if snapshot is not None:
        summary["snapshot"] = snapshot

    return summary


def _read_field_snapshot(run_dir):
    field_dir = run_dir / "data" / "field"
    if not field_dir.is_dir():
        return None

    json_files = sorted(field_dir.glob("*.json"))
    if not json_files:
        return None

    final_json = json_files[-1]
    with open(final_json) as f:
        meta = json.load(f)

    rawfile = meta["meta"]["rawfile"]
    data_path = field_dir / rawfile
    if not data_path.exists():
        return None

    datasets = meta["dataset"]
    if "uf" not in datasets or "um" not in datasets:
        return None

    raw = np.fromfile(str(data_path), dtype=np.float64)

    uf_info = datasets["uf"]
    uf = raw[
        uf_info["offset"] // 8 : uf_info["offset"] // 8 + uf_info["size"] // 8
    ].reshape(uf_info["shape"])

    um_info = datasets["um"]
    um = raw[
        um_info["offset"] // 8 : um_info["offset"] // 8 + um_info["size"] // 8
    ].reshape(um_info["shape"])

    snapshot = {
        "step": meta["meta"]["step"],
        "uf_shape": list(uf_info["shape"]),
        "uf": uf.tolist(),
    }

    n_species = um_info["shape"][-2]
    for ispec in range(n_species):
        density = um[..., ispec, 0]
        snapshot[f"density_{ispec}_shape"] = list(density.shape)
        snapshot[f"density_{ispec}"] = density.tolist()

    return snapshot


def _parse_history(path, Ns):
    rows = []
    for line in path.read_text().strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        rows.append([float(x) for x in stripped.split()])

    if not rows:
        print(f"[analyze] ERROR: no data rows in {path}", file=sys.stderr)
        sys.exit(1)

    data = np.array(rows)
    result = {
        "step": data[:, 0].astype(int).tolist(),
        "time": data[:, 1].tolist(),
        "div_e": data[:, 2].tolist(),
        "div_b": data[:, 3].tolist(),
        "ene_e": data[:, 4].tolist(),
        "ene_b": data[:, 5].tolist(),
    }

    particle_keys = []
    for ispec in range(Ns):
        key = f"ene_p{ispec:02d}"
        col = 6 + ispec
        if col >= data.shape[1]:
            continue
        result[key] = data[:, col].tolist()
        particle_keys.append(key)

    result["particle_keys"] = particle_keys
    return result


def cmd_compare(args):
    case = _resolve_case(args.case)
    golden_msgpack = case.golden_dir / "summary.msgpack"
    if not golden_msgpack.exists():
        print(f"[compare] ERROR: golden not found: {golden_msgpack}", file=sys.stderr)
        print("[compare] Run 'update-golden' first.", file=sys.stderr)
        sys.exit(1)

    with open(golden_msgpack, "rb") as f:
        golden = msgpack.load(f)

    current = analyze_run(case.run_dir, case.Ns)
    rtol = args.rtol
    atol = args.atol

    ok = True
    keys_checked = 0

    for key in ["div_e", "div_b", "ene_e", "ene_b", "total_energy"]:
        if key not in golden or key not in current:
            continue
        g = np.array(golden[key])
        c = np.array(current[key])
        if g.shape != c.shape:
            print(
                f"[compare] FAIL {key}: shape mismatch golden={g.shape} current={c.shape}"
            )
            ok = False
            continue
        if not np.allclose(c, g, rtol=rtol, atol=atol):
            diff = np.max(np.abs(c - g) / (np.abs(g) + atol))
            print(f"[compare] FAIL {key}: max relative diff = {diff:.6e}")
            ok = False
        else:
            print(f"[compare] OK   {key}")
        keys_checked += 1

    for key in current.get("particle_keys", []):
        if key not in golden or key not in current:
            continue
        g = np.array(golden[key])
        c = np.array(current[key])
        if g.shape != c.shape:
            print(f"[compare] FAIL {key}: shape mismatch")
            ok = False
            continue
        if not np.allclose(c, g, rtol=rtol, atol=atol):
            diff = np.max(np.abs(c - g) / (np.abs(g) + atol))
            print(f"[compare] FAIL {key}: max relative diff = {diff:.6e}")
            ok = False
        else:
            print(f"[compare] OK   {key}")
        keys_checked += 1

    for key in ["peak_ene_e", "peak_ene_e_step", "final_energy_drift", "num_steps"]:
        if key not in golden or key not in current:
            continue
        golden_value = golden[key]
        current_value = current[key]
        if key in {"peak_ene_e_step", "num_steps"}:
            match = golden_value == current_value
        else:
            match = abs(current_value - golden_value) < atol + rtol * abs(golden_value)
        if not match:
            print(
                f"[compare] FAIL {key}: golden={golden_value} current={current_value}"
            )
            ok = False
        else:
            print(f"[compare] OK   {key}: {current_value}")
        keys_checked += 1

    golden_snap = golden.get("snapshot")
    current_snap = current.get("snapshot")
    if golden_snap is not None and current_snap is not None:
        snap_ok, snap_checked = _compare_snapshot(golden_snap, current_snap, rtol, atol)
        if not snap_ok:
            ok = False
        keys_checked += snap_checked

    if ok:
        print(f"[compare] PASS ({keys_checked} keys checked)")
        return

    print(f"[compare] FAIL ({keys_checked} keys checked)")
    sys.exit(1)


def _compare_snapshot(golden_snap, current_snap, rtol, atol):
    ok = True
    checked = 0

    golden_step = golden_snap.get("step")
    current_step = current_snap.get("step")
    if golden_step != current_step:
        print(
            f"[compare] FAIL snapshot: step mismatch golden={golden_step} current={current_step}"
        )
        ok = False
    else:
        print(f"[compare] OK   snapshot: step={current_step}")
    checked += 1

    g_uf = np.array(golden_snap["uf"])
    c_uf = np.array(current_snap["uf"])
    g_shape = golden_snap.get("uf_shape", list(g_uf.shape))
    c_shape = current_snap.get("uf_shape", list(c_uf.shape))
    if g_shape != c_shape:
        print(
            f"[compare] FAIL snapshot.uf: shape mismatch golden={g_shape} current={c_shape}"
        )
        ok = False
    elif not np.allclose(c_uf, g_uf, rtol=rtol, atol=atol):
        diff = np.max(np.abs(c_uf - g_uf) / (np.abs(g_uf) + atol))
        print(f"[compare] FAIL snapshot.uf: max relative diff = {diff:.6e}")
        ok = False
    else:
        print("[compare] OK   snapshot.uf")
    checked += 1

    for key in sorted(current_snap.keys()):
        if not key.startswith("density_") or key.endswith("_shape"):
            continue
        base_key = key
        g = np.array(golden_snap.get(base_key, []))
        c = np.array(current_snap[base_key])
        g_shape = golden_snap.get(f"{base_key}_shape", list(g.shape))
        c_shape = current_snap.get(f"{base_key}_shape", list(c.shape))
        if g_shape != c_shape:
            print(
                f"[compare] FAIL snapshot.{base_key}: shape mismatch golden={g_shape} current={c_shape}"
            )
            ok = False
        elif not np.allclose(c, g, rtol=rtol, atol=atol):
            diff = np.max(np.abs(c - g) / (np.abs(g) + atol))
            print(f"[compare] FAIL snapshot.{base_key}: max relative diff = {diff:.6e}")
            ok = False
        else:
            print(f"[compare] OK   snapshot.{base_key}")
        checked += 1

    return ok, checked


def cmd_update_golden(args):
    case = _resolve_case(args.case)
    summary = _ndarray_to_list(analyze_run(case.run_dir, case.Ns))

    case.golden_dir.mkdir(parents=True, exist_ok=True)
    msgpack_path = case.golden_dir / "summary.msgpack"

    with open(msgpack_path, "wb") as f:
        msgpack.dump(summary, f)
    print(f"[update-golden] Wrote {msgpack_path}")


def cmd_plots(args):
    case = _resolve_case(args.case)
    data_dir = case.run_dir / "data"
    if not data_dir.exists():
        print(f"[plots] ERROR: {data_dir} not found", file=sys.stderr)
        print("[plots] Run 'run' first.", file=sys.stderr)
        sys.exit(1)

    if case.generate_plots is None:
        print(f"[plots] No plot hook defined for case '{case.name}'")
        return

    out_dir = case.run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_run(case.run_dir, case.Ns)
    case.generate_plots(case, case.run_dir, data_dir, out_dir, summary, REPO_ROOT)


def cmd_all(args):
    case = _resolve_case(args.case)

    cmd_build(argparse.Namespace(case=case.name, compiler=args.compiler))
    cmd_run(
        argparse.Namespace(
            case=case.name,
            compiler=args.compiler,
            tmax=args.tmax,
            nproc=args.nproc,
        )
    )
    cmd_analyze(argparse.Namespace(case=case.name, output=None))
    cmd_compare(
        argparse.Namespace(
            case=case.name,
            rtol=args.rtol,
            atol=args.atol,
        )
    )
    cmd_plots(argparse.Namespace(case=case.name, compiler=args.compiler))


def _resolve_case(name):
    case_name = DEFAULT_CASE_NAME if name is None else name
    if case_name not in CASES:
        available = ", ".join(sorted(CASES.keys()))
        print(f"Unknown case: {case_name}. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return CASES[case_name]


def _get_compiler_profile(name):
    if name not in COMPILER_PROFILES:
        available = ", ".join(sorted(COMPILER_PROFILES.keys()))
        print(
            f"Unknown compiler profile: {name}. Available: {available}", file=sys.stderr
        )
        sys.exit(1)
    return COMPILER_PROFILES[name]


def _get_executable(case: IntegrationCase, compiler):
    build_dir = _get_compiler_profile(compiler)["build_dir"]
    return build_dir / case.executable_relpath


def _materialize_case_config(case: IntegrationCase):
    with open(case.base_config, "r") as f:
        config = toml.load(f)
    _deep_update(config, case.config_overrides)
    if case.config_patch is not None:
        case.config_patch(config)

    output_path = case.run_dir / "config.toml"
    with open(output_path, "w") as f:
        toml.dump(config, f)

    return output_path


def _deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_update(base[key], value)
            continue
        base[key] = copy.deepcopy(value)
    return base


def _run(cmd, cwd=None, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def _ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _ndarray_to_list(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(value) for value in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="PIC integration test workflow")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    case_help = f"Case name (default: {DEFAULT_CASE_NAME})"

    p = sub.add_parser("build", help="Configure and build example executables")
    p.add_argument("case", nargs="?", default=None, help=case_help)
    p.add_argument(
        "--compiler",
        choices=sorted(COMPILER_PROFILES.keys()),
        default="gcc",
        help="Compiler profile (default: gcc)",
    )

    p = sub.add_parser("run", help="Run simulation with deterministic settings")
    p.add_argument("case", nargs="?", default=None, help=case_help)
    p.add_argument(
        "--compiler",
        choices=sorted(COMPILER_PROFILES.keys()),
        default="gcc",
        help="Compiler profile (default: gcc)",
    )
    p.add_argument("--tmax", type=float, default=None, help="Maximum physical time")
    p.add_argument("--nproc", type=int, default=None, help="Number of MPI ranks")

    p = sub.add_parser("analyze", help="Analyze simulation output")
    p.add_argument("case", nargs="?", default=None, help=case_help)
    p.add_argument("-o", "--output", default=None, help="Write JSON summary to file")

    p = sub.add_parser("compare", help="Compare against golden data")
    p.add_argument("case", nargs="?", default=None, help=case_help)
    p.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance")
    p.add_argument("--atol", type=float, default=1e-30, help="Absolute tolerance")

    p = sub.add_parser("update-golden", help="Regenerate golden data from current run")
    p.add_argument("case", nargs="?", default=None, help=case_help)

    p = sub.add_parser("plots", help="Generate PNG plots for manual review")
    p.add_argument("case", nargs="?", default=None, help=case_help)
    p.add_argument(
        "--compiler",
        choices=sorted(COMPILER_PROFILES.keys()),
        default="gcc",
        help="Compiler profile (default: gcc)",
    )

    p = sub.add_parser("all", help="Run build, run, analyze, compare")
    p.add_argument("case", nargs="?", default=None, help=case_help)
    p.add_argument(
        "--compiler",
        choices=sorted(COMPILER_PROFILES.keys()),
        default="gcc",
        help="Compiler profile (default: gcc)",
    )
    p.add_argument("--tmax", type=float, default=None, help="Maximum physical time")
    p.add_argument("--nproc", type=int, default=None, help="Number of MPI ranks")
    p.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance")
    p.add_argument("--atol", type=float, default=1e-30, help="Absolute tolerance")

    args = parser.parse_args()
    dispatch = {
        "build": cmd_build,
        "run": cmd_run,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "update-golden": cmd_update_golden,
        "plots": cmd_plots,
        "all": cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
