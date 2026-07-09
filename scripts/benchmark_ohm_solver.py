#!/usr/bin/env python3
"""Benchmark picnix Ohm solver scaling on synthetic periodic 2D cases."""

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 96, 128, 192, 256],
        help="Square grid sizes to benchmark.",
    )
    parser.add_argument(
        "--rtol", type=float, default=1.0e-10, help="CG relative tolerance."
    )
    parser.add_argument(
        "--maxiter", type=int, default=1000, help="Maximum CG iterations."
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--compare-amg",
        action="store_true",
        help="Also time externally supplied PyAMG preconditioners.",
    )
    parser.add_argument(
        "--no-pin-threads",
        action="store_true",
        help="Do not force common BLAS/OpenMP thread counts to one.",
    )
    return parser.parse_args()


def pin_threads():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def build_case(n, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.arange(n)
    y = np.arange(n)
    xx = np.broadcast_to(x, (n, n))
    yy = np.broadcast_to(y[:, None], (n, n))
    L = 1.0 + 0.1 * np.cos(2.0 * np.pi * xx / n) + 0.1 * np.sin(2.0 * np.pi * yy / n)
    S = rng.normal(size=(n, n, 3))
    return L, S


def run_default(ohm, n, L, S, args):
    t0 = time.perf_counter()
    _, info = ohm.solve_ohm_2d(
        L,
        S,
        1.0,
        rtol=args.rtol,
        maxiter=args.maxiter,
        return_info=True,
    )
    return time.perf_counter() - t0, 0.0, info


def run_amg(ohm, n, L, S, args):
    import pyamg

    base1, base2 = ohm.build_ohm_bases_from_grid(n, n, c=1.0, delta=1.0)
    A1 = ohm.assemble_matrix_1(n, n, L, 1.0, 0.25, base=base1)
    A2 = ohm.assemble_matrix_2(n, n, L, 1.0, base=base2)

    t0 = time.perf_counter()
    M1 = pyamg.smoothed_aggregation_solver(A1).aspreconditioner(cycle="V")
    M2 = pyamg.smoothed_aggregation_solver(A2).aspreconditioner(cycle="V")
    setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, info = ohm.solve_ohm_2d(
        L,
        S,
        1.0,
        rtol=args.rtol,
        maxiter=args.maxiter,
        M1=M1,
        M2=M2,
        base1=base1,
        base2=base2,
        return_info=True,
    )
    return time.perf_counter() - t0, setup, info


def main():
    args = parse_args()
    if not args.no_pin_threads:
        pin_threads()

    from picnix import ohm

    print(
        f"{'grid':>8} {'mode':>12} {'setup [s]':>10} {'solve [s]':>10} "
        f"{'total [s]':>10} {'it1':>5} {'it2':>5} {'status':>8}"
    )
    print("-" * 84)

    for n in args.sizes:
        L, S = build_case(n, args.seed)
        solve, setup, info = run_default(ohm, n, L, S, args)
        print_result(n, "cg", setup, solve, info)

        if args.compare_amg:
            solve, setup, info = run_amg(ohm, n, L, S, args)
            print_result(n, "amg-cg", setup, solve, info)


def print_result(n, mode, setup, solve, info):
    print(
        f"{n}x{n:<4} {mode:>12} {setup:10.4f} {solve:10.4f} "
        f"{setup + solve:10.4f} {info['niter_1']:5d} {info['niter_2']:5d} "
        f"{info['status_1']},{info['status_2']}"
    )


if __name__ == "__main__":
    main()
