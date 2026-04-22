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
  images         Generate PNG images for manual review.
  all            Execute build, run, analyze, and compare in sequence.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import msgpack
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parents[0]
CONFIG_DIR = SCRIPT_DIR / "config"
GOLDEN_DIR = SCRIPT_DIR / "golden"
RUN_BASE = REPO_ROOT / "run-integration-pic"

COMPILER_PROFILES = {
    "gcc": {
        "build_dir": REPO_ROOT / "build-integration-pic-gcc",
        "cache": REPO_ROOT / "cmake" / "linux-gcc.cmake",
    },
}

# ---------------------------------------------------------------------------
# Case registry (extend here for new cases)
# ---------------------------------------------------------------------------

CASES = {
    "twostream": {
        "target": "beam",
        "executable_relpath": Path("pic") / "example" / "beam" / "main.out",
        "config": CONFIG_DIR / "twostream.toml",
        "golden_dir": GOLDEN_DIR / "twostream",
        "run_dir": RUN_BASE / "twostream",
        "tmax": 30.0,
        "nproc": 8,
        "Ns": 3,
        "snapshot_times": [10.0, 20.0, 30.0],
    },
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def cmd_build(args):
    case_name = _resolve_case(args.case)
    case = _get_case(case_name, args.compiler)
    compiler = _get_compiler_profile(args.compiler)
    target = case["target"]
    build_dir = compiler["build_dir"]

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

    print(f"[build] Building target '{target}' ...")
    _run(
        ["cmake", "--build", str(build_dir), "--target", target, "--parallel"],
        cwd=REPO_ROOT,
    )

    exe = case["executable"]
    if not exe.exists():
        print(f"[build] ERROR: expected executable not found: {exe}", file=sys.stderr)
        sys.exit(1)
    print(f"[build] OK: {exe}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def cmd_run(args):
    case_name = _resolve_case(args.case)
    case = _get_case(case_name, args.compiler)

    exe = case["executable"]
    config = case["config"]
    run_dir = case["run_dir"]
    tmax = args.tmax if args.tmax is not None else case["tmax"]
    nproc = args.nproc if args.nproc is not None else case["nproc"]

    if not exe.exists():
        print(f"[run] ERROR: executable not found: {exe}", file=sys.stderr)
        print("[run] Run 'build' first.", file=sys.stderr)
        sys.exit(1)

    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    mpi_cmd = [
        "mpiexec",
        "-n",
        str(nproc),
        str(exe),
        "-c",
        str(config),
        "-t",
        str(tmax),
    ]

    print(f"[run] Running {case_name} in {run_dir} ...")
    print(f"[run]   cmd: {' '.join(mpi_cmd)}")
    print(f"[run]   OMP_NUM_THREADS=1")

    _run(mpi_cmd, cwd=run_dir, env=env)

    data_dir = run_dir / "data"
    if data_dir.exists():
        print(f"[run] OK: output in {data_dir}")
    else:
        print(
            f"[run] WARNING: expected data directory not found: {data_dir}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


def cmd_analyze(args):
    case_name = _resolve_case(args.case)
    case = _get_case(case_name)
    run_dir = case["run_dir"]
    Ns = case["Ns"]

    summary = analyze_run(run_dir, Ns)

    out = args.output
    if out is None:
        json.dump(_ndarray_to_list(summary), sys.stdout, indent=2)
        print()
    else:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(_ndarray_to_list(summary), f, indent=2)
        print(f"[analyze] Wrote {out}")


def analyze_run(run_dir, Ns):
    history_path = run_dir / "data" / "history.txt"
    if not history_path.exists():
        print(f"[analyze] ERROR: {history_path} not found", file=sys.stderr)
        sys.exit(1)

    data = _parse_history(history_path, Ns)

    n = len(data["step"])
    total_energy = np.array(data["ene_e"]) + np.array(data["ene_b"])
    for k in data["particle_keys"]:
        total_energy += np.array(data[k])

    peak_ene_e = float(np.max(data["ene_e"]))
    peak_ene_e_step = int(data["step"][np.argmax(data["ene_e"])])

    if n >= 2:
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
        "num_steps": n,
    }
    for k in data["particle_keys"]:
        summary[k] = data[k]

    return summary


def _parse_history(path, Ns):
    lines = path.read_text().strip().split("\n")
    header = None
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            header = stripped
            continue
        if not stripped:
            continue
        rows.append([float(x) for x in stripped.split()])

    if not rows:
        print(f"[analyze] ERROR: no data rows in {path}", file=sys.stderr)
        sys.exit(1)

    data = np.array(rows)
    n = data.shape[0]

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
        if col < data.shape[1]:
            result[key] = data[:, col].tolist()
            particle_keys.append(key)

    result["particle_keys"] = particle_keys
    return result


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def cmd_compare(args):
    case_name = _resolve_case(args.case)
    case = _get_case(case_name)
    Ns = case["Ns"]

    golden_dir = case["golden_dir"]
    golden_msgpack = golden_dir / "summary.msgpack"
    if not golden_msgpack.exists():
        print(f"[compare] ERROR: golden not found: {golden_msgpack}", file=sys.stderr)
        print("[compare] Run 'update-golden' first.", file=sys.stderr)
        sys.exit(1)

    with open(golden_msgpack, "rb") as f:
        golden = msgpack.load(f)

    current = analyze_run(case["run_dir"], Ns)

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
        g = golden[key]
        c = current[key]
        if key == "peak_ene_e_step" or key == "num_steps":
            match = g == c
        else:
            match = abs(c - g) < atol + rtol * abs(g)
        if not match:
            print(f"[compare] FAIL {key}: golden={g} current={c}")
            ok = False
        else:
            print(f"[compare] OK   {key}: {c}")
        keys_checked += 1

    if ok:
        print(f"[compare] PASS ({keys_checked} keys checked)")
    else:
        print(f"[compare] FAIL ({keys_checked} keys checked)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Update golden
# ---------------------------------------------------------------------------


def cmd_update_golden(args):
    case_name = _resolve_case(args.case)
    case = _get_case(case_name)
    Ns = case["Ns"]

    summary = analyze_run(case["run_dir"], Ns)
    summary_list = _ndarray_to_list(summary)

    golden_dir = case["golden_dir"]
    golden_dir.mkdir(parents=True, exist_ok=True)

    msgpack_path = golden_dir / "summary.msgpack"
    json_path = golden_dir / "summary.json"

    with open(msgpack_path, "wb") as f:
        msgpack.dump(summary_list, f)
    print(f"[update-golden] Wrote {msgpack_path}")

    with open(json_path, "w") as f:
        json.dump(summary_list, f, indent=2)
    print(f"[update-golden] Wrote {json_path}")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def cmd_images(args):
    case_name = _resolve_case(args.case)
    case = _get_case(case_name)
    run_dir = case["run_dir"]

    data_dir = run_dir / "data"
    if not data_dir.exists():
        print(f"[images] ERROR: {data_dir} not found", file=sys.stderr)
        print("[images] Run 'run' first.", file=sys.stderr)
        sys.exit(1)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out_dir = run_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    stale_divergence_plot = out_dir / "divergence_history.png"
    if stale_divergence_plot.exists():
        stale_divergence_plot.unlink()

    summary = analyze_run(run_dir, case["Ns"])
    time_arr = np.array(summary["time"])

    # -- energy history -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_arr, np.array(summary["ene_e"]), label=r"$E^2/2$")
    ax.plot(time_arr, np.array(summary["ene_b"]), label=r"$B^2/2$")
    for ispec in range(summary["Ns"]):
        k = f"ene_p{ispec:02d}"
        ax.plot(time_arr, np.array(summary[k]), label=k)
    ax.plot(time_arr, np.array(summary["total_energy"]), "k--", label="total")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.legend(fontsize=8)
    ax.set_title("Energy history")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_history.png", dpi=150)
    plt.close(fig)
    print(f"[images] Wrote {out_dir / 'energy_history.png'}")

    # -- snapshot plots (batch.py style) --------------------------------------
    try:
        sys.path.insert(0, str(REPO_ROOT / "script"))
        import picnix

        profile_path = data_dir / "profile.msgpack"
        if not profile_path.exists():
            for candidate in sorted(data_dir.glob("profile*.msgpack")):
                profile_path = candidate
                break

        if not profile_path.exists():
            print(
                "[images] No profile.msgpack found; skipping field/particle snapshots"
            )
            return

        run = picnix.Run(str(profile_path))
        field_steps = run.get_step("field")
        particle_steps = run.get_step("particle")

        snapshot_times = case.get("snapshot_times", [10.0, 20.0, 30.0])

        for target_time in snapshot_times:
            target_step = int(round(target_time / run.delt))
            closest_field = int(
                field_steps[np.argmin(np.abs(field_steps - target_step))]
            )
            closest_particle = int(
                particle_steps[np.argmin(np.abs(particle_steps - target_step))]
            )

            outpath = out_dir / f"snapshot_{closest_field:08d}.png"
            _generate_snapshot(run, closest_field, closest_particle, outpath)
            print(
                f"[images] Wrote {outpath.name}  (t={target_time:.1f}, "
                f"step={closest_field})"
            )

    except ImportError:
        print("[images] picnix module not available; skipping field/particle snapshots")
    except Exception as e:
        print(f"[images] Warning: could not generate field/particle snapshots: {e}")


def _generate_snapshot(run, field_step, particle_step, outpath):
    import sys

    import matplotlib
    import matplotlib as mpl

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    sys.path.insert(0, str(REPO_ROOT / "script"))
    import picnix

    plt.rcParams.update({"font.size": 12})

    field = run.read_at("field", field_step)
    particle = run.read_at("particle", particle_step)

    xc = field["xc"]
    uf = field["uf"]
    um = field["um"]

    Nx = run.Nx
    delh = run.delh
    xlim = (0, Nx * delh)

    me = 1.0
    mi = 1.0e2

    ebinx = (0, Nx, Nx + 1)
    ebiny = (-30, +30, 61)
    ibinx = (0, Nx, Nx + 1)
    ibiny = (-5, +5, 61)

    up = [particle[f"up{s:02d}"] for s in range(run.Ns)]

    # 4-panel figure matching batch.py layout
    fig = plt.figure(figsize=(6.4, 6.4), dpi=120)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.12,
        right=0.85,
        hspace=0.25,
        wspace=0.02,
    )
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[50, 1])
    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    # panel 0: density
    ne = (um[..., 0, 0] + um[..., 1, 0]).mean(axis=(0, 1)) / me
    ni = um[..., 2, 0].mean(axis=(0, 1)) / mi
    plt.sca(axs[0])
    plt.plot(xc, ne, "b-")
    plt.plot(xc, ni, "r-")
    axs[0].set_ylabel(r"$N$")
    axs[0].set_ylim(0.5, 2.5)
    axs[0].set_xlim(xlim)

    # panel 1: electric field
    ex = uf[..., 0].mean(axis=(0, 1))
    plt.sca(axs[1])
    plt.plot(xc, ex, "k-")
    axs[1].set_ylabel(r"$E_x$")
    axs[1].set_ylim(-10, +10)
    axs[1].set_xlim(xlim)

    # panel 2: electron phase space (species 0 + 1 combined)
    fvx0 = picnix.Histogram2D(up[0][:, 0], up[0][:, 3], ebinx, ebiny)
    fvx1 = picnix.Histogram2D(up[1][:, 0], up[1][:, 3], ebinx, ebiny)
    Xe, Ye, Ze = fvx0.pcolormesh_args()
    _, _, Z1 = fvx1.pcolormesh_args()
    Ze = Ze + Z1
    plt.sca(axs[2])
    plt.pcolormesh(Xe, Ye, Ze, shading="nearest")
    axs[2].set_ylabel(r"$v_x$")
    axs[2].set_xlim(xlim)
    fmt = mpl.ticker.FormatStrFormatter("%4.0e")
    cax = fig.add_subplot(gs[2, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_e(x, v_x)$")

    # panel 3: ion phase space (species 2)
    fvx2 = picnix.Histogram2D(up[2][:, 0], up[2][:, 3], ibinx, ibiny)
    Xi, Yi, Zi = fvx2.pcolormesh_args()
    plt.sca(axs[3])
    plt.pcolormesh(Xi, Yi, Zi, shading="nearest")
    axs[3].set_xlabel(r"$x$")
    axs[3].set_ylabel(r"$v_x$")
    axs[3].set_xlim(xlim)
    fmt = mpl.ticker.FormatStrFormatter("%4.0e")
    cax = fig.add_subplot(gs[3, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_i(x, v_x)$")

    tt = run.get_time_at("particle", particle_step)
    fig.suptitle(rf"$t = {tt:6.2f}$")

    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# All
# ---------------------------------------------------------------------------


def cmd_all(args):
    case_name = _resolve_case(args.case)

    build_args = argparse.Namespace(case=case_name, compiler=args.compiler)
    cmd_build(build_args)

    run_args = argparse.Namespace(
        case=case_name,
        compiler=args.compiler,
        tmax=args.tmax,
        nproc=args.nproc,
    )
    cmd_run(run_args)

    analyze_args = argparse.Namespace(case=case_name, output=None)
    cmd_analyze(analyze_args)

    compare_args = argparse.Namespace(
        case=case_name,
        rtol=args.rtol,
        atol=args.atol,
    )
    cmd_compare(compare_args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_case(name):
    if name is None:
        return "twostream"
    if name not in CASES:
        print(
            f"Unknown case: {name}. Available: {', '.join(CASES.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    return name


def _get_compiler_profile(name):
    if name not in COMPILER_PROFILES:
        print(
            f"Unknown compiler profile: {name}. Available: {', '.join(COMPILER_PROFILES.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    return COMPILER_PROFILES[name]


def _get_case(case_name, compiler="gcc"):
    case = dict(CASES[case_name])
    build_dir = _get_compiler_profile(compiler)["build_dir"]
    case["build_dir"] = build_dir
    case["executable"] = build_dir / case["executable_relpath"]
    return case


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
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="PIC integration test workflow",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # -- build --
    p = sub.add_parser("build", help="Configure and build example executables")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )
    p.add_argument(
        "--compiler",
        choices=sorted(COMPILER_PROFILES.keys()),
        default="gcc",
        help="Compiler profile (default: gcc)",
    )

    # -- run --
    p = sub.add_parser("run", help="Run simulation with deterministic settings")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )
    p.add_argument(
        "--compiler",
        choices=sorted(COMPILER_PROFILES.keys()),
        default="gcc",
        help="Compiler profile (default: gcc)",
    )
    p.add_argument("--tmax", type=float, default=None, help="Maximum physical time")
    p.add_argument("--nproc", type=int, default=None, help="Number of MPI ranks")

    # -- analyze --
    p = sub.add_parser("analyze", help="Analyze simulation output")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )
    p.add_argument("-o", "--output", default=None, help="Write JSON summary to file")

    # -- compare --
    p = sub.add_parser("compare", help="Compare against golden data")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )
    p.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance")
    p.add_argument("--atol", type=float, default=1e-30, help="Absolute tolerance")

    # -- update-golden --
    p = sub.add_parser("update-golden", help="Regenerate golden data from current run")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )

    # -- images --
    p = sub.add_parser("images", help="Generate PNG images for manual review")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )

    # -- all --
    p = sub.add_parser("all", help="Run build, run, analyze, compare")
    p.add_argument(
        "case", nargs="?", default=None, help="Case name (default: twostream)"
    )
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
        "images": cmd_images,
        "all": cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
