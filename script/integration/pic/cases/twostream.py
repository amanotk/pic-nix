from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import IntegrationCase


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parents[1]


CASE = IntegrationCase(
    name="twostream",
    target="beam",
    executable_relpath=Path("pic") / "example" / "beam" / "main.out",
    base_config=REPO_ROOT / "pic" / "example" / "beam" / "twostream" / "config.toml",
    golden_dir=SCRIPT_DIR / "golden" / "twostream",
    run_dir=REPO_ROOT / "run-integration-pic" / "twostream",
    Ns=3,
    tmax=30.0,
    nproc=8,
    snapshot_times=(10.0, 20.0, 30.0),
    config_overrides={
        "application": {
            "rebalance": {"interval": 999999999, "loglevel": 0},
            "option": {"seed_type": "fixed"},
        }
    },
    generate_plots=lambda case,
    run_dir,
    data_dir,
    out_dir,
    summary,
    repo_root: generate_plots(case, data_dir, out_dir, summary, repo_root),
)


def generate_plots(case, data_dir, out_dir, summary, repo_root):
    stale_plot = out_dir / "divergence_history.png"
    if stale_plot.exists():
        stale_plot.unlink()

    _plot_energy_history(
        summary,
        out_dir / "energy_history.png",
        ("ele1", "ele2", "ion"),
        xlim=(0.0, case.tmax),
        ylim=(1.0e-4, 2.0e0),
    )
    print(f"[plots] Wrote {out_dir / 'energy_history.png'}")

    try:
        picnix = _load_picnix(repo_root)
        profile_path = _get_profile_path(data_dir)
        if profile_path is None:
            print("[plots] No profile.msgpack found; skipping field/particle snapshots")
            return

        run = picnix.Run(str(profile_path))
        field_steps = run.get_step("field")
        particle_steps = run.get_step("particle")

        for target_time in case.snapshot_times:
            target_step = int(round(target_time / run.delt))
            closest_field = int(
                field_steps[np.argmin(np.abs(field_steps - target_step))]
            )
            closest_particle = int(
                particle_steps[np.argmin(np.abs(particle_steps - target_step))]
            )
            outpath = out_dir / f"snapshot_{closest_field:08d}.png"
            _generate_snapshot(run, picnix, closest_field, closest_particle, outpath)
            print(
                f"[plots] Wrote {outpath.name}  (t={target_time:.1f}, step={closest_field})"
            )
    except ImportError:
        print("[plots] picnix module not available; skipping field/particle snapshots")
    except Exception as exc:
        print(f"[plots] Warning: could not generate field/particle snapshots: {exc}")


def _plot_energy_history(summary, outpath, species_labels, xlim, ylim):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    time_arr = np.array(summary["time"])
    total_energy = np.array(summary["total_energy"])
    safe_total = np.where(total_energy != 0.0, total_energy, np.nan)

    fig, (ax_energy, ax_drift) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    ax_energy.plot(
        time_arr, _normalized(summary["ene_e"], safe_total), label=r"$E^2/2$"
    )
    ax_energy.plot(
        time_arr, _normalized(summary["ene_b"], safe_total), label=r"$B^2/2$"
    )
    for key, label in zip(summary.get("particle_keys", []), species_labels):
        ax_energy.plot(time_arr, _normalized(summary[key], safe_total), label=label)
    ax_energy.plot(time_arr, np.ones_like(total_energy), "k--", label="total")
    ax_energy.set_yscale("log")
    ax_energy.set_ylabel("normalized energy")
    ax_energy.legend(fontsize=8)
    ax_energy.set_title("Energy history")
    ax_energy.set_xlim(*xlim)
    ax_energy.set_ylim(*ylim)

    initial_total = total_energy[0]
    if initial_total == 0.0:
        relative_drift = total_energy - initial_total
    else:
        relative_drift = (total_energy - initial_total) / initial_total
    ax_drift.plot(time_arr, relative_drift, color="k")
    ax_drift.axhline(0.0, color="k", linewidth=0.8)
    ax_drift.set_yscale("symlog", linthresh=1.0e-6)
    ax_drift.set_xlabel("time")
    ax_drift.set_ylabel("total energy drift")
    ax_drift.set_xlim(*xlim)
    ax_drift.set_ylim(-1.0e-4, 1.0e-4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _generate_snapshot(run, picnix, field_step, particle_step, outpath):
    import matplotlib
    import matplotlib as mpl

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 12})

    field = run.read_at("field", field_step)
    particle = run.read_at("particle", particle_step)
    xc = field["xc"]
    uf = field["uf"]
    um = field["um"]
    up = [particle[f"up{species:02d}"] for species in range(run.Ns)]

    xlim = (0, run.Nx * run.delh)
    me = 1.0
    mi = 1.0e2
    ebinx = (0, run.Nx, run.Nx + 1)
    ebiny = (-30, +30, 31)
    ibinx = (0, run.Nx, run.Nx + 1)
    ibiny = (-5, +5, 31)
    electron_range = (0.0, 20.0)
    ion_range = (0.0, 70.0)

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
    axes = [fig.add_subplot(gs[index, 0]) for index in range(4)]

    ne = (um[..., 0, 0] + um[..., 1, 0]).mean(axis=(0, 1)) / me
    ni = um[..., 2, 0].mean(axis=(0, 1)) / mi
    plt.sca(axes[0])
    plt.plot(xc, ne, "b-")
    plt.plot(xc, ni, "r-")
    axes[0].set_ylabel(r"$N$")
    axes[0].set_ylim(0.5, 2.5)
    axes[0].set_xlim(xlim)

    ex = uf[..., 0].mean(axis=(0, 1))
    plt.sca(axes[1])
    plt.plot(xc, ex, "k-")
    axes[1].set_ylabel(r"$E_x$")
    axes[1].set_ylim(-10, +10)
    axes[1].set_xlim(xlim)

    fvx0 = picnix.Histogram2D(up[0][:, 0], up[0][:, 3], ebinx, ebiny)
    fvx1 = picnix.Histogram2D(up[1][:, 0], up[1][:, 3], ebinx, ebiny)
    xe, ye, ze = fvx0.pcolormesh_args()
    _, _, z1 = fvx1.pcolormesh_args()
    plt.sca(axes[2])
    plt.pcolormesh(
        xe,
        ye,
        ze + z1,
        shading="nearest",
        vmin=electron_range[0],
        vmax=electron_range[1],
    )
    axes[2].set_ylabel(r"$v_x$")
    axes[2].set_xlim(xlim)
    fmt = mpl.ticker.FormatStrFormatter("%4.0e")
    cax = fig.add_subplot(gs[2, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_e(x, v_x)$")

    fvx2 = picnix.Histogram2D(up[2][:, 0], up[2][:, 3], ibinx, ibiny)
    xi, yi, zi = fvx2.pcolormesh_args()
    plt.sca(axes[3])
    plt.pcolormesh(
        xi,
        yi,
        zi,
        shading="nearest",
        vmin=ion_range[0],
        vmax=ion_range[1],
    )
    axes[3].set_xlabel(r"$x$")
    axes[3].set_ylabel(r"$v_x$")
    axes[3].set_xlim(xlim)
    cax = fig.add_subplot(gs[3, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_i(x, v_x)$")

    tt = run.get_time_at("particle", particle_step)
    fig.suptitle(rf"$t = {tt:6.2f}$")
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def _normalized(values, safe_total):
    data = np.array(values) / safe_total
    return np.clip(data, 1.0e-30, None)


def _get_profile_path(data_dir):
    profile_path = data_dir / "profile.msgpack"
    if profile_path.exists():
        return profile_path
    for candidate in sorted(data_dir.glob("profile*.msgpack")):
        return candidate
    return None


def _load_picnix(repo_root):
    script_dir = str(repo_root / "script")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import picnix

    return picnix
