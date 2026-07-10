#!/usr/bin/env python3
"""Benchmark the 2D Gauss-law-reduced Ohm solver."""

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 96, 128, 192, 256])
    parser.add_argument(
        "--cases", choices=["constant", "variable", "both"], default="both"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["cg", "amg-cg", "fft-pcg", "fft"],
        default=["cg", "amg-cg", "fft-pcg", "fft"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-pin-threads", action="store_true")
    return parser.parse_args()


def pin_threads():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"


def build_case(n, seed, variable):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.arange(n)
    y = np.arange(n)
    xx = np.broadcast_to(x, (n, n))
    yy = np.broadcast_to(y[:, None], (n, n))
    if variable:
        L = 1.0 + 0.2 * np.cos(2.0 * np.pi * xx / n)
        L += 0.1 * np.sin(2.0 * np.pi * yy / n)
    else:
        L = np.ones((n, n))
    rho = 0.1 * np.sin(2.0 * np.pi * xx / n) * np.cos(2.0 * np.pi * yy / n)
    S = rng.normal(size=(n, n, 3))
    return L, S, rho


def build_preconditioner(ohm, method, L, matrix):
    if method not in {"amg-cg", "fft-pcg"}:
        return None, 0.0
    kind = "amg" if method == "amg-cg" else "fft"
    t0 = time.perf_counter()
    preconditioner = ohm.build_ohm_gauss_preconditioner_2d(
        L,
        1.0,
        kind=kind,
        matrix=matrix if kind == "amg" else None,
        validate_matrix=False,
    )
    return preconditioner, time.perf_counter() - t0


def run_method(ohm, method, L, S, rho, matrix, preconditioner, args):
    solver = "fft" if method == "fft" else "cg"
    solver_matrix = None if method == "fft" else matrix
    times = []
    info = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        ohm.solve_ohm_2d_gauss_reduced(
            L,
            S,
            rho,
            1.0,
            solver=solver,
            preconditioner=preconditioner,
            matrix=solver_matrix,
            validate_matrix=False,
            rtol=args.rtol,
            maxiter=args.maxiter,
        )
        times.append(time.perf_counter() - t0)
    _, info = ohm.solve_ohm_2d_gauss_reduced(
        L,
        S,
        rho,
        1.0,
        solver=solver,
        preconditioner=preconditioner,
        matrix=solver_matrix,
        validate_matrix=False,
        rtol=args.rtol,
        maxiter=args.maxiter,
        return_info=True,
    )
    return times, info


def print_header():
    print(
        f"{'case':>8} {'grid':>9} {'method':>9} {'matrix':>9} {'setup':>9} "
        f"{'solve':>9} {'iterations':>14} {'status':>8} {'residual':>10} {'nnz':>10}"
    )
    print("-" * 117)


def print_result(case, n, method, matrix_time, setup_time, times, info, nnz):
    import numpy as np

    iterations = "/".join(str(value) for value in info["niter"])
    status = "/".join(str(value) for value in info["status"])
    print(
        f"{case:>8} {n:4d}x{n:<4d} {method:>9} {matrix_time:9.4f} {setup_time:9.4f} "
        f"{np.mean(times):9.4f} {iterations:>14} {status:>8} "
        f"{max(info['relative_residual']):10.2e} {nnz:10d}"
    )


def main():
    args = parse_args()
    if not args.no_pin_threads:
        pin_threads()

    from picnix import ohm

    cases = [args.cases] if args.cases != "both" else ["constant", "variable"]
    print_header()
    for case in cases:
        for n in args.sizes:
            L, S, rho = build_case(n, args.seed, variable=case == "variable")
            matrix = None
            matrix_time = 0.0
            if any(method != "fft" for method in args.methods):
                t0 = time.perf_counter()
                matrix = ohm.assemble_ohm_gauss_matrix_2d(L, 1.0)
                matrix_time = time.perf_counter() - t0
            for method in args.methods:
                if method == "fft" and case == "variable":
                    continue
                preconditioner, setup_time = build_preconditioner(
                    ohm, method, L, matrix
                )
                times, info = run_method(
                    ohm, method, L, S, rho, matrix, preconditioner, args
                )
                reported_matrix_time = 0.0 if method == "fft" else matrix_time
                nnz = 0 if method == "fft" else matrix.nnz
                print_result(
                    case,
                    n,
                    method,
                    reported_matrix_time,
                    setup_time,
                    times,
                    info,
                    nnz,
                )


if __name__ == "__main__":
    main()
