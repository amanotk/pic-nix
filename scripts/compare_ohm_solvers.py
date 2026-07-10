#!/usr/bin/env python3
"""Compare curl-curl and Gauss-reduced Ohm solvers against PIC field data."""

import argparse
import time

import numpy as np

VALID_PRECONDITIONERS = {"fft", "cg", "amg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", help="PIC-NIX profile.msgpack path")
    parser.add_argument("steps", type=int, nargs="+", help="Field diagnostic steps")
    parser.add_argument("--decimate", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--qm",
        help="Comma-separated per-species q/m values; required for profiles without qm metadata.",
    )
    parser.add_argument(
        "--c", type=float, help="Override speed of light from parameter.cc"
    )
    parser.add_argument(
        "--preconditioner",
        nargs="+",
        choices=list(VALID_PRECONDITIONERS),
        default=["fft"],
        help="Gauss-reduced preconditioners to compare (default: fft). "
        "Multiple values run each variant.",
    )
    return parser.parse_args()


def parse_qm(value, run):
    if value is not None:
        return np.fromstring(value, sep=",", dtype=np.float64)
    if getattr(run, "qm", None) is not None:
        return np.asarray(run.qm, dtype=np.float64)
    raise ValueError("profile has no qm metadata; pass --qm=q1,q2,... explicitly")


def average_2d_periodic(values, decimate):
    if decimate < 1:
        raise ValueError(f"decimate must be positive, got {decimate}")
    if decimate == 1:
        return np.asarray(values)

    from scipy import ndimage

    filtered = ndimage.uniform_filter(values, size=decimate, axes=(0, 1), mode="wrap")
    return filtered[decimate // 2 :: decimate, decimate // 2 :: decimate]


def build_source(B, M, delta, c):
    gamma = M[..., 1:4]
    pxx, pyy = M[..., 4], M[..., 5]
    pxy, pyz, pzx = M[..., 7], M[..., 8], M[..., 9]

    def ddx(values):
        return (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (
            2.0 * delta
        )

    def ddy(values):
        return (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (
            2.0 * delta
        )

    div_pi = np.stack(
        [ddx(pxx) + ddy(pxy), ddx(pxy) + ddy(pyy), ddx(pzx) + ddy(pyz)], axis=-1
    )
    return -np.cross(gamma, B, axis=-1) / c + div_pi


def divergence(E, delta):
    return (
        np.roll(E[..., 0], -1, axis=1)
        - np.roll(E[..., 0], 1, axis=1)
        + np.roll(E[..., 1], -1, axis=0)
        - np.roll(E[..., 1], 1, axis=0)
    ) / (2.0 * delta)


def field_errors(E, reference, rho, delta):
    eps = np.finfo(np.float64).eps
    difference = E - reference
    l2 = np.linalg.norm(difference.reshape((-1, 3)), axis=0)
    l2 /= np.maximum(np.linalg.norm(reference.reshape((-1, 3)), axis=0), eps)
    linf = np.max(np.abs(difference), axis=(0, 1))
    linf /= np.maximum(np.max(np.abs(reference), axis=(0, 1)), eps)
    gauss = np.linalg.norm(divergence(E, delta) - rho) / max(np.linalg.norm(rho), eps)
    return l2, linf, gauss


def print_result(name, elapsed, E, info, reference, rho, delta):
    l2, linf, gauss = field_errors(E, reference, rho, delta)
    if name.startswith("curl-curl"):
        status = (info["status_1"], info["status_2"])
        niter = (info["niter_1"], info["niter_2"])
    else:
        status = info["status"]
        niter = info["niter"]
    print(
        f"  {name:18s} solve={elapsed:7.3f}s status={status} niter={niter} "
        f"rel_l2={np.array2string(l2, precision=3)} "
        f"rel_linf={np.array2string(linf, precision=3)} gauss={gauss:.3e}"
    )


def main():
    args = parse_args()
    if args.decimate < 1:
        raise ValueError("--decimate must be at least one")
    if args.rtol <= 0.0:
        raise ValueError("--rtol must be positive")

    import picnix
    from picnix import ohm

    run = picnix.Run(args.profile)
    qm = parse_qm(args.qm, run)
    c = float(run.config["parameter"].get("cc", 1.0) if args.c is None else args.c)
    if not np.isfinite(c) or c == 0.0:
        raise ValueError(f"c must be finite and non-zero, got {c}")

    print(
        f"profile={args.profile} qm={qm.tolist()} c={c:g} decimate={args.decimate} "
        f"rtol={args.rtol:g}"
    )
    for step in args.steps:
        read_begin = time.perf_counter()
        data = run.read_at("field", step)
        read_time = time.perf_counter() - read_begin
        if data["uf"].shape[0] != 1 or data["um"].shape[0] != 1:
            raise ValueError(
                "comparison supports 2D diagnostics with a singleton z axis"
            )
        if data["um"].shape[-2] != qm.size:
            raise ValueError(
                f"qm has {qm.size} species but um has {data['um'].shape[-2]}"
            )

        uf = average_2d_periodic(data["uf"][0], args.decimate)
        um = average_2d_periodic(data["um"][0], args.decimate)
        delta = float(run.delh) * args.decimate
        M = ohm.transform_moments(um, qm)
        rho = np.sum(um[..., :, 0] * qm, axis=-1)
        source = build_source(uf[..., 3:6], M, delta, c)
        E_pic = uf[..., :3]

        begin = time.perf_counter()
        E_curl, curl_info = ohm.solve_ohm_2d(
            M[..., 0],
            source,
            delta,
            c=c,
            rtol=args.rtol,
            maxiter=args.maxiter,
            return_info=True,
        )
        curl_time = time.perf_counter() - begin

        need_amg = "amg" in args.preconditioner
        gauss_matrix = None
        if need_amg:
            gauss_matrix = ohm.assemble_ohm_gauss_matrix_2d(M[..., 0], delta, c=c)

        reduced_times = {}
        reduced_results = {}
        for precond in args.preconditioner:
            if precond == "amg":
                solver = "cg"
                pc = "amg"
                mat = gauss_matrix
            elif precond == "cg":
                solver = "cg"
                pc = None
                mat = None
            else:  # fft
                solver = "cg"
                pc = "fft"
                mat = None

            label = f"gauss-{precond}"
            begin = time.perf_counter()
            E_reduced, reduced_info = ohm.solve_ohm_2d_gauss_reduced(
                M[..., 0],
                source,
                rho,
                delta,
                c=c,
                solver=solver,
                preconditioner=pc,
                matrix=mat,
                validate_matrix=False,
                rtol=args.rtol,
                maxiter=args.maxiter,
                return_info=True,
            )
            reduced_times[label] = time.perf_counter() - begin
            reduced_results[label] = (E_reduced, reduced_info)

        pic_gauss = np.linalg.norm(divergence(E_pic, delta) - rho) / max(
            np.linalg.norm(rho), np.finfo(np.float64).eps
        )
        print(
            f"step={step} grid={E_pic.shape[:2]} read={read_time:.3f}s "
            f"rho_std={rho.std():.3e} pic_gauss={pic_gauss:.3e}"
        )
        print_result("curl-curl", curl_time, E_curl, curl_info, E_pic, rho, delta)
        for label in args.preconditioner:
            precond_label = f"gauss-{label}"
            elapsed, (E, info) = (
                reduced_times[precond_label],
                reduced_results[precond_label],
            )
            print_result(precond_label, elapsed, E, info, E_pic, rho, delta)


if __name__ == "__main__":
    main()
