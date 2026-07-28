#!/usr/bin/env python3
"""Analysis plots for the hybrid beam simulation.

Reads nixio MPI-IO diagnostic output (field/*.json + field/*.data) and
profile.msgpack to assemble global arrays from chunked data.

Produces:
  (1) Snapshot plots: By(x), Vy ions, Vy beam at ~12 evenly-spaced times
  (2) Fourier mode plot: |FFT(By + i*Bz)| for modes 4,5,6 vs time (log scale)

Usage:
  python analysis.py [--basedir data] [--outdir plots] [--select N]
"""

import argparse
import json
import msgpack
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_profile(basedir: Path) -> dict:
    with open(basedir / "profile.msgpack", "rb") as f:
        content = f.read()
    u = msgpack.Unpacker(None, max_buffer_size=len(content))
    u.feed(content)
    return next(iter(u))


def load_config(basedir: Path) -> dict:
    """Extract delh and grid parameters from the profile."""
    profile = load_profile(basedir)
    p = profile["configuration"]["parameter"]
    return {
        "delh": float(p["delh"]),
        "Nx": int(p["Nx"]),
        "Cx": int(p["Cx"]),
    }


def load_all_steps(basedir: Path) -> dict:
    """Return {step: json_path} for all field snapshots."""
    field_dir = basedir / "field"
    result = {}
    for f in sorted(field_dir.glob("*.json")):
        obj = json.loads(f.read_text())
        result[obj["meta"]["step"]] = f
    return result


def read_field_moment(json_path: Path, rawfile: Path):
    """Read field_cell and moment datasets from one snapshot."""
    obj = json.loads(json_path.read_text())

    fi = obj["dataset"]["field_cell"]
    mi = obj["dataset"]["moment"]
    nc, nz, ny, cnx = fi["shape"][:4]

    with open(rawfile, "rb") as fp:
        fp.seek(fi["offset"])
        field = np.fromfile(fp, "<f8", int(np.prod(fi["shape"])))
        fp.seek(mi["offset"])
        moment = np.fromfile(fp, "<f8", int(np.prod(mi["shape"])))

    field = field.reshape(fi["shape"])  # (nc, nz, ny, cnx, 6)
    moment = moment.reshape(mi["shape"])  # (nc, nz, ny, cnx, Ns, 10)
    return field, moment


def assemble_global(field, moment):
    """Stitch chunked data into global 1D arrays at z=0, y=0."""
    nc, nz, ny, cnx = field.shape[:4]
    gnx = nc * cnx

    By = np.zeros(gnx)
    Bz = np.zeros(gnx)
    Vy_ions = np.zeros(gnx)
    Vy_beam = np.zeros(gnx)

    for ic in range(nc):
        xs, xe = ic * cnx, (ic + 1) * cnx
        By[xs:xe] = field[ic, 0, 0, :, 4]
        Bz[xs:xe] = field[ic, 0, 0, :, 5]
        rho0 = moment[ic, 0, 0, :, 0, 0] + 1e-30
        rho1 = moment[ic, 0, 0, :, 1, 0] + 1e-30
        Vy_ions[xs:xe] = moment[ic, 0, 0, :, 0, 2] / rho0
        Vy_beam[xs:xe] = moment[ic, 0, 0, :, 1, 2] / rho1

    return By, Bz, Vy_ions, Vy_beam


def make_snapshot_plots(basedir: Path, outdir: Path, n_select: int,
                        config: dict, steps_all: list[int]):
    """Generate snapshot plots at evenly-spaced steps."""
    outdir.mkdir(parents=True, exist_ok=True)
    field_dir = basedir / "field"
    files_by_step = load_all_steps(basedir)

    idx = np.linspace(0, len(steps_all) - 1, min(n_select, len(steps_all)), dtype=int)
    select = [steps_all[i] for i in idx]

    delh = config["delh"]
    gnx = config["Nx"]
    xc = delh * (np.arange(gnx) + 0.5)

    for step in select:
        fname = files_by_step[step]
        obj = json.loads(fname.read_text())
        rawfile = field_dir / obj["meta"]["rawfile"]
        field, moment = read_field_moment(fname, rawfile)
        By, Bz, Vy_ions, Vy_beam = assemble_global(field, moment)
        t = step * 0.01

        bmax = max(np.abs(By).max(), np.abs(Bz).max())
        vmax = max(np.abs(Vy_ions).max(), np.abs(Vy_beam).max())

        fig, axs = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

        axs[0].plot(xc, By, "C0-", lw=0.6)
        axs[0].set_ylabel("By")
        axs[0].set_title(f"By(x)  step={step}  t={t:.1f}  max|By|={bmax:.3f}")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(xc, Vy_ions, "C2-", lw=0.6)
        axs[1].set_ylabel("Vy ions")
        axs[1].set_title(f"Vy ions(x)  max|Vy|={vmax:.3f}")
        axs[1].grid(True, alpha=0.3)
        if vmax > 0:
            axs[1].set_ylim(-vmax * 1.15, vmax * 1.15)

        axs[2].plot(xc, Vy_beam, "C3-", lw=0.6)
        axs[2].set_ylabel("Vy beam")
        axs[2].set_xlabel("x [c/wp]")
        axs[2].set_title(f"Vy beam(x)  max|Vy|={np.abs(Vy_beam).max():.3f}")
        axs[2].grid(True, alpha=0.3)
        if vmax > 0:
            axs[2].set_ylim(-vmax * 1.15, vmax * 1.15)

        plt.tight_layout()
        fout = outdir / f"snapshot_step_{step:06d}.png"
        fig.savefig(fout, dpi=120)
        plt.close(fig)
        print(f"  {fout.name}  |By|={bmax:.4f}")

    print(f"Saved {len(select)} snapshot plots to {outdir}")


def make_fourier_plot(basedir: Path, outdir: Path, steps_all: list[int]):
    """Plot |FFT(By+i*Bz)| for modes 4,5,6 (positive and negative) vs time."""
    outdir.mkdir(parents=True, exist_ok=True)
    field_dir = basedir / "field"
    files_by_step = load_all_steps(basedir)

    times = []
    power_pos = {m: [] for m in [4, 5, 6]}
    power_neg = {m: [] for m in [4, 5, 6]}

    for step in steps_all:
        fname = files_by_step[step]
        obj = json.loads(fname.read_text())
        rawfile = field_dir / obj["meta"]["rawfile"]
        field, _moment = read_field_moment(fname, rawfile)

        nc, _nz, _ny, cnx = field.shape[:4]
        gnx = nc * cnx
        By = np.zeros(gnx)
        Bz = np.zeros(gnx)
        for ic in range(nc):
            xs, xe = ic * cnx, (ic + 1) * cnx
            By[xs:xe] = field[ic, 0, 0, :, 4]
            Bz[xs:xe] = field[ic, 0, 0, :, 5]

        complex_b = np.zeros(gnx, dtype=np.complex128)
        complex_b.real = By
        complex_b.imag = Bz
        fft = np.abs(np.fft.fft(complex_b))
        t = step * 0.01
        times.append(t)
        for m in [4, 5, 6]:
            power_pos[m].append(fft[m])             # positive wavenumber
            power_neg[m].append(fft[gnx - m])       # negative wavenumber

    times = np.array(times)
    colors = {4: "C0", 5: "C1", 6: "C2"}

    # Positive modes
    fig, ax = plt.subplots(figsize=(14, 6))
    for m in [4, 5, 6]:
        ax.plot(times, np.array(power_pos[m]), color=colors[m], lw=1, label=f"mode +{m}")
    ax.set_yscale("log")
    ax.set_xlabel("Time [c/wp]")
    ax.set_ylabel("|FFT(By + i Bz)|")
    ax.set_title("Positive Wavenumber Modes  |FFT(By + i Bz)|")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(outdir / "fourier_modes_pos.png", dpi=120)
    plt.close(fig)

    # Negative modes
    fig, ax = plt.subplots(figsize=(14, 6))
    for m in [4, 5, 6]:
        ax.plot(times, np.array(power_neg[m]), color=colors[m], lw=1, label=f"mode -{m}")
    ax.set_yscale("log")
    ax.set_xlabel("Time [c/wp]")
    ax.set_ylabel("|FFT(By + i Bz)|")
    ax.set_title("Negative Wavenumber Modes  |FFT(By + i Bz)|")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(outdir / "fourier_modes_neg.png", dpi=120)
    plt.close(fig)

    print(f"Saved {outdir / 'fourier_modes_pos.png'}")
    print(f"Saved {outdir / 'fourier_modes_neg.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analysis plots for hybrid beam simulation."
    )
    parser.add_argument(
        "--basedir",
        type=Path,
        default=Path("data"),
        help="Diagnostic output directory (default: data)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--select",
        type=int,
        default=12,
        help="Number of snapshot steps to plot (default: 12)",
    )
    args = parser.parse_args()

    if not args.basedir.is_dir():
        print(f"Error: basedir not found: {args.basedir}", file=sys.stderr)
        return 1

    config = load_config(args.basedir)
    steps_all = sorted(load_all_steps(args.basedir))
    if not steps_all:
        print(f"Error: no field snapshots in {args.basedir}/field/", file=sys.stderr)
        return 1

    print(
        f"Config: Nx={config['Nx']} Cx={config['Cx']} delh={config['delh']}"
        f"  snapshots: {steps_all[0]}..{steps_all[-1]} ({len(steps_all)})"
    )

    make_snapshot_plots(
        args.basedir, args.outdir, args.select, config, steps_all
    )
    make_fourier_plot(args.basedir, args.outdir, steps_all)

    print(f"\nDone. Plots in {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
