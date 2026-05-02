from __future__ import annotations

from pathlib import Path

import numpy as np

from . import IntegrationCase


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _patch_diagnostic_intervals(config):
    for diag in config.get("diagnostic", []):
        if diag.get("name") == "particle":
            diag["interval"] = 800


CASE = IntegrationCase(
    name="shock",
    target="shock",
    executable_relpath=Path("pic") / "example" / "shock" / "main.out",
    base_config=REPO_ROOT / "pic" / "example" / "shock" / "config.toml",
    golden_dir=SCRIPT_DIR / "golden" / "shock",
    run_dir=REPO_ROOT / "run-integration-pic" / "shock",
    Ns=2,
    tmax=160.0,
    nproc=8,
    snapshot_times=(40.0, 80.0, 120.0, 160.0),
    config_overrides={
        "application": {
            "option": {"seed_type": "fixed"},
        },
        "parameter": {
            "Nx": 256,
            "Ny": 1,
            "Nz": 1,
            "Cx": 16,
            "Cy": 1,
            "Cz": 1,
            "nppc": 8,
            "mime": 16,
        },
    },
    config_patch=_patch_diagnostic_intervals,
    generate_plots=lambda case, run_dir, data_dir, out_dir, summary, repo_root: (
        generate_plots(case, data_dir, out_dir, summary, repo_root)
    ),
)


def generate_plots(case, data_dir, out_dir, summary, repo_root):
    try:
        import picnix

        profile_path = _get_profile_path(data_dir)
        if profile_path is None:
            print("[plots] No profile.msgpack found; skipping field snapshots")
            return

        run = picnix.Run(str(profile_path))
        field_steps = run.get_step("field")
        particle_steps = run.get_step("particle")

        mime = run.config["parameter"]["mime"]
        sigma = run.config["parameter"]["sigma"]
        u0 = run.config["parameter"]["u0"]
        b0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
        roe = 1.0
        roi = 1.0 * mime

        ebinx = (0, run.Nx, run.Nx + 1)
        ebiny = (-1.0, +1.0, 41)
        ibinx = (0, run.Nx, run.Nx + 1)
        ibiny = (-0.2, +0.2, 41)
        xlim = (0, run.Nx * run.delh)

        for target_time in case.snapshot_times:
            target_step = int(round(target_time / run.delt))
            closest_field = int(
                field_steps[np.argmin(np.abs(field_steps - target_step))]
            )
            closest_particle = int(
                particle_steps[np.argmin(np.abs(particle_steps - target_step))]
            )
            outpath = out_dir / f"snapshot_{closest_field:08d}.png"
            _generate_snapshot(
                run,
                picnix,
                closest_field,
                closest_particle,
                outpath,
                xlim=xlim,
                roe=roe,
                roi=roi,
                b0=b0,
                ebinx=ebinx,
                ebiny=ebiny,
                ibinx=ibinx,
                ibiny=ibiny,
            )
            print(
                f"[plots] Wrote {outpath.name}  (t={target_time:.1f}, step={closest_field})"
            )
    except ImportError:
        print("[plots] picnix module not available; skipping field snapshots")
    except Exception as exc:
        print(f"[plots] Warning: could not generate field snapshots: {exc}")


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
    ax_drift.set_ylim(-1.0e-2, 1.0e-2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _generate_snapshot(
    run,
    picnix,
    field_step,
    particle_step,
    outpath,
    *,
    xlim,
    roe,
    roi,
    b0,
    ebinx,
    ebiny,
    ibinx,
    ibiny,
):
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
    up = [particle["up00"], particle["up01"]]
    tt = run.get_time_at("field", field_step)

    fig = plt.figure(figsize=(6.4, 6.4), dpi=120)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.12,
        right=0.85,
        hspace=0.25,
        wspace=0.02,
    )
    gridspec = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[50, 1])
    axs = [fig.add_subplot(gridspec[i, 0]) for i in range(4)]

    ne = um[..., 0, 0].mean(axis=(0, 1)) / roe
    ni = um[..., 1, 0].mean(axis=(0, 1)) / roi
    plt.sca(axs[0])
    plt.plot(xc, ne, "b-", lw=1.0)
    plt.plot(xc, ni, "r-", lw=1.0)
    axs[0].set_ylabel(r"$N$")
    axs[0].set_ylim(0.5, 6.5)

    bx = uf[..., 3].mean(axis=(0, 1)) / b0
    by = uf[..., 4].mean(axis=(0, 1)) / b0
    bz = uf[..., 5].mean(axis=(0, 1)) / b0
    plt.sca(axs[1])
    plt.plot(xc, bx, "r-", lw=1.0)
    plt.plot(xc, by, "g-", lw=1.0)
    plt.plot(xc, bz, "b-", lw=1.0)
    axs[1].set_ylabel(r"$B$")
    axs[1].set_ylim(-1, +10)

    electron_range = (0, 1000)
    ion_range = (0, 4000)

    fvxe = picnix.Histogram2D(up[0][:, 0], up[0][:, 3], ebinx, ebiny)
    xe, ye, ze = fvxe.pcolormesh_args()
    plt.sca(axs[2])
    plt.pcolormesh(
        xe, ye, ze, shading="nearest", vmin=electron_range[0], vmax=electron_range[1]
    )
    axs[2].set_ylabel(r"$v_x$")
    fmt = mpl.ticker.FormatStrFormatter("%4.0e")
    cax = fig.add_subplot(gridspec[2, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_e(x, v_x)$")

    fvxi = picnix.Histogram2D(up[1][:, 0], up[1][:, 3], ibinx, ibiny)
    xi, yi, zi = fvxi.pcolormesh_args()
    plt.sca(axs[3])
    plt.pcolormesh(xi, yi, zi, shading="nearest", vmin=ion_range[0], vmax=ion_range[1])
    axs[3].set_xlabel(r"$x$")
    axs[3].set_ylabel(r"$v_x$")
    cax = fig.add_subplot(gridspec[3, 1])
    plt.colorbar(cax=cax, format=fmt, label=r"$f_i(x, v_x)$")

    for i in range(4):
        axs[i].set_xlim(xlim)
    fig.align_ylabels(axs)

    coord = np.array(run.chunkmap["coord"])
    rank = run.get_chunk_rank(field_step)
    cdelx = run.delh * (run.Nx // run.Cx)
    for i in range(4):
        picnix.plot_chunk_dist1d(axs[i], coord, rank, cdelx, colors="gray")

    fig.suptitle(rf"$t = {tt:6.2f}$")
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def _get_profile_path(data_dir):
    profile_path = data_dir / "profile.msgpack"
    if profile_path.exists():
        return profile_path
    for candidate in sorted(data_dir.glob("profile*.msgpack")):
        return candidate
    return None


def _normalized(values, safe_total):
    data = np.array(values) / safe_total
    return np.clip(data, 1.0e-30, None)
