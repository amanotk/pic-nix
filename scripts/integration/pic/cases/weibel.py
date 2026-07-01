from __future__ import annotations

from pathlib import Path

import numpy as np

from . import IntegrationCase


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parents[1]


_SHARED_OVERRIDES = {
    "parameter": {
        "Nx": 32,
        "Ny": 32,
        "Cx": 4,
        "Cy": 4,
    },
}


def _PLOT_HOOK(case, run_dir, data_dir, out_dir, summary, repo_root):
    return generate_plots(case, data_dir, out_dir, summary, repo_root)


def _make_case(order: int) -> IntegrationCase:
    return IntegrationCase(
        name=f"weibel-order{order}",
        target="beam",
        executable_relpath=Path("pic") / "example" / "beam" / "main.out",
        base_config=REPO_ROOT / "pic" / "example" / "beam" / "weibel" / "config.toml",
        golden_dir=SCRIPT_DIR / "golden" / f"weibel-order{order}",
        run_dir=REPO_ROOT / "run-integration-pic" / f"weibel-order{order}",
        Ns=4,
        tmax=30.0,
        nproc=8,
        snapshot_times=(10.0, 20.0, 30.0),
        config_overrides={
            "application": {
                "option": {"seed_type": "fixed", "order": order},
            },
            **_SHARED_OVERRIDES,
        },
        generate_plots=_PLOT_HOOK,
    )


CASES = {f"weibel-order{order}": _make_case(order) for order in (1, 2, 3, 4)}


def generate_plots(case, data_dir, out_dir, summary, repo_root):
    _plot_energy_history(
        summary,
        out_dir / "energy_history.png",
        (r"$e^{-}_1$", r"$e^{-}_2$", r"$e^{+}_1$", r"$e^{+}_2$"),
        xlim=(0.0, case.tmax),
        ylim=(1.0e-5, 2.0e0),
    )
    print(f"[plots] Wrote {out_dir / 'energy_history.png'}")

    try:
        import picnix

        profile_path = _get_profile_path(data_dir)
        if profile_path is None:
            print("[plots] No profile.msgpack found; skipping field snapshots")
            return

        run = picnix.Run(str(profile_path))
        field_steps = run.get_step("field")
        for target_time in case.snapshot_times:
            target_step = int(round(target_time / run.delt))
            closest_field = int(
                field_steps[np.argmin(np.abs(field_steps - target_step))]
            )
            outpath = out_dir / f"snapshot_{closest_field:08d}.png"
            _generate_snapshot(run, picnix, closest_field, outpath)
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
    ax_energy.legend(fontsize=8, ncol=2)
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


def _generate_snapshot(run, picnix, field_step, outpath):
    import matplotlib
    import matplotlib as mpl

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 12})

    field = run.read_at("field", field_step)
    qm = [particle["qm"] for particle in run.config["parameter"]["particle"]]
    xc = field["xc"]
    yc = field["yc"]
    uf = field["uf"]
    um = field["um"]
    tt = run.get_time_at("field", field_step)
    xlim = (0, run.Nx * run.delh)
    ylim = (0, run.Ny * run.delh)
    density_range = (0.75, 3.25)
    magnetic_range = (0.0, 40.0)
    current_range = (-50.0, 50.0)

    fig = plt.figure(figsize=(9.6, 3.6), dpi=120)
    fig.subplots_adjust(
        top=0.80,
        bottom=0.14,
        left=0.075,
        right=0.95,
        hspace=0.05,
        wspace=0.35,
    )
    gridspec = fig.add_gridspec(2, 3, height_ratios=[2, 50], width_ratios=[1, 1, 1])
    axes = []
    color_axes = []
    for index in range(3):
        axes.append(fig.add_subplot(gridspec[1, index]))
        axes[index].set_aspect("equal")
        color_axes.append(fig.add_subplot(gridspec[0, index]))

    x_grid, y_grid = np.broadcast_arrays(xc[None, :], yc[:, None])

    plt.sca(axes[0])
    density = um[..., 0].sum(axis=(0, 3)) / um.shape[0] / 2
    plt.pcolormesh(
        x_grid,
        y_grid,
        density,
        shading="nearest",
        vmin=density_range[0],
        vmax=density_range[1],
    )
    plt.colorbar(cax=color_axes[0], orientation="horizontal")
    color_axes[0].xaxis.set_ticks_position("top")
    color_axes[0].set_title(r"$N$")

    plt.sca(axes[1])
    magnetic_strength = (uf[..., 3] ** 2 + uf[..., 4] ** 2 + uf[..., 5] ** 2).mean(
        axis=0
    )
    plt.pcolormesh(
        x_grid,
        y_grid,
        magnetic_strength,
        shading="nearest",
        vmin=magnetic_range[0],
        vmax=magnetic_range[1],
    )
    plt.colorbar(cax=color_axes[1], orientation="horizontal")
    color_axes[1].xaxis.set_ticks_position("top")
    color_axes[1].set_title(r"$|B|$")

    plt.sca(axes[2])
    current = np.zeros((yc.size, xc.size))
    for index, charge_to_mass in enumerate(qm):
        current += charge_to_mass * um[..., index, 3].mean(axis=0)
    plt.pcolormesh(
        x_grid,
        y_grid,
        current,
        shading="nearest",
        vmin=current_range[0],
        vmax=current_range[1],
    )
    plt.colorbar(cax=color_axes[2], orientation="horizontal")
    color_axes[2].xaxis.set_ticks_position("top")
    color_axes[2].set_title(r"$J_z$")

    major_x = max(1, int(round(run.Nx / 4)))
    major_y = max(1, int(round(run.Ny / 4)))
    minor_x = max(1, int(round(major_x / 2)))
    minor_y = max(1, int(round(major_y / 2)))
    for index in range(3):
        axes[index].xaxis.set_major_locator(mpl.ticker.MultipleLocator(major_x))
        axes[index].yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_y))
        axes[index].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(minor_x))
        axes[index].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(minor_y))
        axes[index].set_xlim(xlim)
        axes[index].set_ylim(ylim)
        axes[index].set_xlabel(r"$x / c/\omega_{pe}$")
        axes[index].set_ylabel(r"$y / c/\omega_{pe}$")
        ax_pos = axes[index].get_position()
        cx_pos = color_axes[index].get_position()
        color_axes[index].set_position(
            [ax_pos.x0, cx_pos.y0, ax_pos.width, cx_pos.height]
        )

    coord = np.array(run.chunkmap["coord"])
    rank = run.get_chunk_rank(field_step)
    cdelx = run.delh * run.Nx // run.Cx
    cdely = run.delh * run.Ny // run.Cy
    for axis in axes:
        picnix.plot_chunk_dist2d(axis, coord, rank, cdelx, cdely, colors="white")

    fig.suptitle(r"$\omega_{{pe}} t = {:6.2f}$".format(tt), x=0.5, y=0.99)
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
