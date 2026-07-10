#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generalized Ohm's law solver for picnix.

Solves

.. math::

    (\\Lambda + c^2 \\nabla \\times \\nabla \\times) \\boldsymbol{E}
    = -\\frac{\\Gamma}{c} \\times \\boldsymbol{B} + \\nabla \\cdot \\Pi,

in Lorentz-Heaviside units, with periodic boundary conditions in all
spatial directions. The quantities

.. math::

    \\Lambda = \\sum_s \\left(\\frac{q_s}{m_s}\\right)^2
                   \\int f_s\\, d\\boldsymbol{v}, \\qquad
    \\Gamma = \\sum_s \\left(\\frac{q_s}{m_s}\\right)^2
                   \\int \\boldsymbol{v}\\, f_s\\, d\\boldsymbol{v}, \\qquad
    \\Pi    = \\sum_s \\frac{q_s}{m_s}
                   \\int \\boldsymbol{v}\\boldsymbol{v}\\, f_s\\, d\\boldsymbol{v},

are obtained from the per-species moment data emitted by PIC-NIX via
:func:`transform_moments`.

Two solvers are provided:

* :func:`solve_ohm_1d` for 1D (``\\partial/\\partial y = \\partial/\\partial z = 0``)
* :func:`solve_ohm_2d` for 2D in the x-y plane (``\\partial/\\partial z = 0``)

In 1D the curl-curl identity reduces to ``(\\nabla \\times \\nabla \\times \\boldsymbol{E})_x = 0``
and ``(\\nabla \\times \\nabla \\times \\boldsymbol{E})_y = -\\partial_x^2 E_y``
(the same for ``z``), so the ``E_x`` equation is the pointwise division
``E_x = S_x / \\Lambda`` and the ``E_y``/``E_z`` equations are identical
circulant systems.

In 2D, the standard finite-difference discretization of
``\\nabla \\times \\nabla \\times`` couples ``(E_x, E_y)`` through a
``2N \\times 2N`` block matrix (``N = N_x N_y``); ``E_z`` is decoupled in
an ``N \\times N`` matrix. Both are solved by conjugate gradient with
optional external preconditioning.
"""

import inspect

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu


_CG_TOL_KWARG = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"


__all__ = [
    "solve_ohm_1d",
    "solve_ohm_2d",
    "solve_ohm_2d_gauss_reduced",
    "calc_e_ohm_1d",
    "calc_e_ohm_2d",
    "transform_moments",
    "qm_per_species_from_config",
    "build_ohm_bases_from_grid",
    "build_ohm_bases",
    "assemble_matrix_1",
    "assemble_matrix_2",
    "build_ohm_gauss_base_2d",
    "assemble_ohm_gauss_matrix_2d",
    "build_ohm_gauss_preconditioner_2d",
]


# -*- Periodic difference operators -*-


def _periodic_first_difference_matrix(n):
    """Periodic 1D first-difference matrix (n x n), no 1/(2*dx) factor."""
    D = sparse.lil_matrix((n, n))
    D.setdiag(1.0, k=1)
    D.setdiag(-1.0, k=-1)
    D[0, n - 1] = -1.0
    D[n - 1, 0] = 1.0
    return D.tocsr()


def _periodic_second_difference_matrix(n):
    """Periodic 1D second-difference matrix (n x n), no 1/dx^2 factor."""
    D = sparse.lil_matrix((n, n))
    D.setdiag(-2.0)
    D.setdiag(1.0, k=-1)
    D.setdiag(1.0, k=1)
    D[0, n - 1] = 1.0
    D[n - 1, 0] = 1.0
    return D.tocsr()


def build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4):
    """Build Lambda-independent base matrices for the 2D Ohm solver."""
    Ix = sparse.eye(Nx, format="csr")
    Iy = sparse.eye(Ny, format="csr")
    Dx1 = _periodic_first_difference_matrix(Nx)
    Dxx = _periodic_second_difference_matrix(Nx)
    Dy1 = _periodic_first_difference_matrix(Ny)
    Dyy = _periodic_second_difference_matrix(Ny)
    Lx = sparse.kron(Iy, Dxx, format="csr")
    Ly = sparse.kron(Dyy, Ix, format="csr")
    Cxy = sparse.kron(Dy1, Dx1, format="csr")

    base1 = sparse.bmat(
        [[(-c2_dx2) * Ly, c2_dx4 * Cxy], [c2_dx4 * Cxy.T, (-c2_dx2) * Lx]],
        format="csr",
    )
    base2 = (-c2_dx2) * (Lx + Ly)
    return base1, base2


def build_ohm_bases_from_grid(Nx, Ny, c, delta):
    """Build Lambda-independent 2D base matrices from grid parameters."""
    c2_dx2 = c * c / (delta * delta)
    c2_dx4 = c2_dx2 / 4.0
    return build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)


def assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=None):
    """Assemble the coupled ``(E_x, E_y)`` sparse matrix."""
    if base is None:
        base, _ = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
    expected_shape = (2 * Nx * Ny, 2 * Nx * Ny)
    if base.shape != expected_shape:
        raise ValueError(f"base shape {base.shape} must be {expected_shape}")

    L_flat = L.flatten(order="C")
    return base + sparse.diags(np.concatenate((L_flat, L_flat)), format="csr")


def assemble_matrix_2(Nx, Ny, L, c2_dx2, base=None):
    """Assemble the decoupled ``E_z`` sparse matrix."""
    if base is None:
        _, base = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx2 / 4.0)
    expected_shape = (Nx * Ny, Nx * Ny)
    if base.shape != expected_shape:
        raise ValueError(f"base shape {base.shape} must be {expected_shape}")

    L_flat = L.flatten(order="C")
    return base + sparse.diags(L_flat, format="csr")


def _validate_grid_parameters(delta, c):
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be finite and positive, got {delta}")
    if not np.isfinite(c):
        raise ValueError(f"c must be finite, got {c}")


def build_ohm_gauss_base_2d(Nx, Ny, delta, *, c=1.0):
    """Build the Lambda-independent periodic ``-c^2 Laplacian`` matrix."""
    if Nx < 3 or Ny < 3:
        raise ValueError(f"Nx and Ny must both be at least 3, got Nx={Nx}, Ny={Ny}")
    _validate_grid_parameters(delta, c)

    Ix = sparse.eye(Nx, format="csr")
    Iy = sparse.eye(Ny, format="csr")
    Dxx = _periodic_second_difference_matrix(Nx)
    Dyy = _periodic_second_difference_matrix(Ny)
    laplacian = sparse.kron(Iy, Dxx, format="csr") + sparse.kron(Dyy, Ix, format="csr")
    return (-(c * c) / (delta * delta)) * laplacian


def _validate_lambda_2d(L, min_lambda):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError(f"L must be 2D with shape (Ny, Nx), got {L.shape}")
    if L.shape[0] < 3 or L.shape[1] < 3:
        raise ValueError(f"L axes must both contain at least 3 points, got {L.shape}")
    if not np.all(np.isfinite(L)):
        raise ValueError("L must contain only finite values")
    if not np.isfinite(min_lambda) or min_lambda < 0.0:
        raise ValueError(
            f"min_lambda must be finite and non-negative, got {min_lambda}"
        )
    if np.min(L) <= min_lambda:
        raise ValueError(f"L must be greater than min_lambda={min_lambda}")
    return L


def assemble_ohm_gauss_matrix_2d(L, delta, *, c=1.0, base=None, min_lambda=0.0):
    """Assemble ``diag(Lambda) - c^2 Laplacian`` for a periodic 2D grid."""
    L = _validate_lambda_2d(L, min_lambda)
    _validate_grid_parameters(delta, c)
    Ny, Nx = L.shape
    if base is None:
        base = build_ohm_gauss_base_2d(Nx, Ny, delta, c=c)
    else:
        base = _validate_gauss_base_2d(base, L.shape, delta, c)
    return base + sparse.diags(L.flatten(order="C"), format="csr")


def _validate_gauss_base_2d(base, shape, delta, c):
    Ny, Nx = shape
    expected = build_ohm_gauss_base_2d(Nx, Ny, delta, c=c)
    return _validate_sparse_operator(base, expected, "base")


def _validate_gauss_matrix_2d(matrix, L, delta, c):
    Ny, Nx = L.shape
    expected = build_ohm_gauss_base_2d(Nx, Ny, delta, c=c)
    expected = expected + sparse.diags(L.flatten(order="C"), format="csr")
    return _validate_sparse_operator(matrix, expected, "matrix")


def _validate_sparse_operator(operator, expected, name):
    if not sparse.issparse(operator):
        raise TypeError(f"{name} must be a SciPy sparse matrix")
    if operator.shape != expected.shape:
        raise ValueError(f"{name} shape {operator.shape} must be {expected.shape}")
    operator_csr = operator.tocsr()
    if not np.all(np.isfinite(operator_csr.data)):
        raise ValueError(f"{name} must contain only finite values")

    difference = operator_csr - expected.tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        scale = max(float(np.max(np.abs(expected.data))), 1.0)
        if np.max(np.abs(difference.data)) > 1.0e-12 * scale:
            raise ValueError(
                f"{name} does not match the requested finite-difference operator"
            )
    return operator


def _fft_denominator(shape, delta, c, lambda0):
    Ny, Nx = shape
    fy = np.fft.fftfreq(Ny)
    fx = np.fft.rfftfreq(Nx)
    kappa2 = (4.0 / (delta * delta)) * (
        np.sin(np.pi * fy[:, None]) ** 2 + np.sin(np.pi * fx[None, :]) ** 2
    )
    return lambda0 + c * c * kappa2


def _build_fft_preconditioner(shape, delta, c, lambda0):
    denominator = _fft_denominator(shape, delta, c, lambda0)
    size = shape[0] * shape[1]

    def matvec(vector):
        values = np.asarray(vector, dtype=np.float64).reshape(shape, order="C")
        transformed = np.fft.rfftn(values, axes=(0, 1))
        result = np.fft.irfftn(transformed / denominator, s=shape, axes=(0, 1))
        return result.flatten(order="C")

    return LinearOperator((size, size), matvec=matvec, dtype=np.float64)


def build_ohm_gauss_preconditioner_2d(
    L,
    delta,
    *,
    c=1.0,
    kind="amg",
    matrix=None,
    validate_matrix=True,
    fft_lambda=None,
    min_lambda=0.0,
):
    """Build a reusable AMG or constant-coefficient FFT preconditioner."""
    L = _validate_lambda_2d(L, min_lambda)
    _validate_grid_parameters(delta, c)
    kind = str(kind).lower()

    if kind == "amg":
        if matrix is None:
            matrix = assemble_ohm_gauss_matrix_2d(L, delta, c=c, min_lambda=min_lambda)
        elif validate_matrix:
            matrix = _validate_gauss_matrix_2d(matrix, L, delta, c)
        else:
            expected_shape = (L.size, L.size)
            if not sparse.issparse(matrix):
                raise TypeError("matrix must be a SciPy sparse matrix")
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"matrix shape {matrix.shape} must be {expected_shape}"
                )
        import pyamg

        return pyamg.smoothed_aggregation_solver(matrix).aspreconditioner(cycle="V")

    if kind == "fft":
        if matrix is not None:
            raise ValueError("kind='fft' does not use a sparse matrix")
        if fft_lambda is None:
            fft_lambda = float(np.mean(L))
        if not np.isfinite(fft_lambda) or fft_lambda <= 0.0:
            raise ValueError(
                f"fft_lambda must be finite and positive, got {fft_lambda}"
            )
        return _build_fft_preconditioner(L.shape, delta, c, fft_lambda)

    raise ValueError(f"unknown preconditioner {kind!r}; expected 'amg' or 'fft'")


# -*- Solver cores -*-


def _cg_solve(A, b, rtol, maxiter, M):
    niter = 0

    def count_iteration(_):
        nonlocal niter
        niter += 1

    x, status = cg(
        A,
        b,
        M=M,
        **{_CG_TOL_KWARG: rtol},
        atol=0.0,
        maxiter=maxiter,
        callback=count_iteration,
    )
    return x, int(status), niter


def _build_gauss_reduced_rhs_2d(S, rho, delta, c):
    """Return ``S - c^2 grad(rho)`` using centered periodic differences."""
    d_rho_dx = (np.roll(rho, -1, axis=-1) - np.roll(rho, 1, axis=-1)) / (2.0 * delta)
    d_rho_dy = (np.roll(rho, -1, axis=-2) - np.roll(rho, 1, axis=-2)) / (2.0 * delta)
    rhs = np.array(S, dtype=np.float64, copy=True)
    rhs[..., 0] -= c * c * d_rho_dx
    rhs[..., 1] -= c * c * d_rho_dy
    return rhs


def _apply_gauss_operator_2d(L, values, delta, c):
    laplacian = (
        np.roll(values, -1, axis=1)
        + np.roll(values, 1, axis=1)
        + np.roll(values, -1, axis=0)
        + np.roll(values, 1, axis=0)
        - 4.0 * values
    ) / (delta * delta)
    multiplier = L if values.ndim == 2 else L[..., None]
    return multiplier * values - c * c * laplacian


def _relative_residuals(L, rhs, solution, delta, c):
    eps = np.finfo(np.float64).eps
    residual = rhs - _apply_gauss_operator_2d(L, solution, delta, c)
    residuals = []
    for component in range(3):
        b = rhs[..., component].flatten(order="C")
        component_residual = residual[..., component].flatten(order="C")
        residuals.append(
            float(np.linalg.norm(component_residual) / max(np.linalg.norm(b), eps))
        )
    return tuple(residuals)


def _solve_gauss_fft(rhs, L, delta, c, constant_rtol, constant_atol):
    lambda0 = float(L.flat[0])
    if not np.allclose(L, lambda0, rtol=constant_rtol, atol=constant_atol):
        raise ValueError(
            "solver='fft' requires L to be constant within the requested tolerance"
        )
    denominator = _fft_denominator(L.shape, delta, c, lambda0)
    transformed = np.fft.rfftn(rhs, axes=(0, 1))
    return np.fft.irfftn(transformed / denominator[..., None], s=L.shape, axes=(0, 1))


def _solve_ohm_1d_core(L, S_yz, c2_dx2, rtol, maxiter, M):
    """Solve the circulant 1D system ``(Lambda - c^2 D^2) E = S`` once.

    Used for both the ``E_y`` and ``E_z`` components.
    """
    N = L.shape[0]
    D2 = _periodic_second_difference_matrix(N)
    A = (-c2_dx2) * D2 + sparse.diags(L, format="csr")
    return _cg_solve(A, S_yz, rtol, maxiter, M)


def _build_ohm_matrices_2d(Nx, Ny, L, c2_dx2, c2_dx4):
    """Build the coupled (Ex, Ey) and decoupled Ez sparse matrices."""
    base1, base2 = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
    return (
        assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=base1),
        assemble_matrix_2(Nx, Ny, L, c2_dx2, base=base2),
    )


def _solve_ohm_2d_core(L, S, c2_dx2, c2_dx4, rtol, maxiter, M1, M2, base1, base2):
    """Solve the 2D periodic system, returning ``E`` of shape ``(Ny, Nx, 3)``."""
    Ny, Nx = L.shape
    N = Nx * Ny
    if base1 is None or base2 is None:
        built_base1, built_base2 = build_ohm_bases(Nx, Ny, c2_dx2, c2_dx4)
        if base1 is None:
            base1 = built_base1
        if base2 is None:
            base2 = built_base2
    A1 = assemble_matrix_1(Nx, Ny, L, c2_dx2, c2_dx4, base=base1)
    A2 = assemble_matrix_2(Nx, Ny, L, c2_dx2, base=base2)

    S1 = np.concatenate((S[..., 0].flatten(order="C"), S[..., 1].flatten(order="C")))
    S2 = S[..., 2].flatten(order="C")

    E1, status_1, niter_1 = _cg_solve(A1, S1, rtol, maxiter, M1)
    E2, status_2, niter_2 = _cg_solve(A2, S2, rtol, maxiter, M2)

    E = np.zeros((Ny, Nx, 3), dtype=np.float64)
    E[..., 0] = E1[:N].reshape((Ny, Nx), order="C")
    E[..., 1] = E1[N:].reshape((Ny, Nx), order="C")
    E[..., 2] = E2.reshape((Ny, Nx), order="C")
    return E, int(status_1), int(status_2), niter_1, niter_2


# -*- Per-species q/m resolution and moment transformation -*-


def qm_per_species_from_config(config):
    """Infer per-species charge-to-mass ratio from a picnix config.

    This is a *fallback* used by :func:`_resolve_qm` for profiles written
    by older picnix builds that do not carry the ``qm`` field. New
    profiles always include ``qm`` (accessible as ``run.qm``), so this
    helper is rarely needed.

    Supports two config schemas used in PIC-NIX examples:

    * ``[[parameter.particle]]`` (array of per-species blocks, each with
      a ``qm`` field; e.g., ``beam/twostream``, ``beam/weibel``, ``thermal``)
    * Top-level ``mime``, ``nppc``, ``wp`` keys, which implies a 2-species
      electron-ion pair (e.g., ``anisotropy``). Returns ``[-wp, +wp/mime]``
      in code units (matches the C++ side which sets ``me = 1/nppc``,
      ``qe = -wp/nppc`` so ``qme = qe/me = -wp``).

    For multi-species cases that are not electron-ion pairs, the qm
    cannot be inferred from the config; pass ``qm_per_species`` to
    :func:`calc_e_ohm_1d` or :func:`calc_e_ohm_2d` explicitly.
    """
    p = config["parameter"]
    Ns = p["Ns"]

    if "particle" in p:
        return np.array([float(p["particle"][s]["qm"]) for s in range(Ns)])

    if "mime" in p and "wp" in p and "nppc" in p:
        if Ns != 2:
            raise ValueError(
                "mime/wp/nppc qm inference only valid for Ns=2 (electron-ion); "
                "pass qm_per_species to calc_e_ohm_1d / calc_e_ohm_2d explicitly"
            )
        wp = float(p["wp"])
        mime = float(p["mime"])
        return np.array([-wp, +wp / mime])

    raise ValueError(
        "Cannot infer qm; pass qm_per_species to calc_e_ohm_1d / calc_e_ohm_2d"
    )


def transform_moments(um, qm):
    """Build the transformed moments M from raw per-species moments.

    ``um`` has shape ``(..., Ns, 14)`` following the PIC-NIX moment
    layout (see ``engine/moment.hpp``):

    * indices 0-3: Lambda-, Gamma_x-, Gamma_y-, Gamma_z-weights
    * indices 5-7: P_xx, P_yy, P_zz (diagonal pressure)
    * indices 11-13: P_xy, P_yz, P_zx (off-diagonal pressure)

    ``qm`` has shape ``(Ns,)``. Returns M of shape ``(..., 10)`` with axes
    ``[Lambda, Gamma_x, Gamma_y, Gamma_z, P_xx, P_yy, P_zz, P_xy, P_yz, P_zx]``.
    """
    qm = np.asarray(qm, dtype=np.float64)
    leading = um.shape[:-2]
    M = np.empty(leading + (10,), dtype=np.float64)
    M[..., 0:4] = (um[..., 0:4] * (qm[:, None] ** 2)).sum(axis=-2)
    M[..., 4:7] = (um[..., 5:8] * qm[:, None]).sum(axis=-2)  # P_xx, P_yy, P_zz
    M[..., 7:10] = (um[..., 11:14] * qm[:, None]).sum(axis=-2)  # P_xy, P_yz, P_zx
    return M


# -*- Source term from (B, M) -*-


def _build_source_term_1d(B, M, delta):
    """Compute ``S = -Gamma cross B + div Pi`` in 1D.

    ``B`` and ``M`` both have shape ``(Nx, ...)`` with the trailing
    component axis (3 for ``B``, 10 for ``M``).
    """
    Gamma = M[..., 1:4]
    Pxx, Pxy, Pzx = M[..., 4], M[..., 7], M[..., 9]

    dPxx_dx = (np.roll(Pxx, -1, axis=-1) - np.roll(Pxx, 1, axis=-1)) / (2.0 * delta)
    dPxy_dx = (np.roll(Pxy, -1, axis=-1) - np.roll(Pxy, 1, axis=-1)) / (2.0 * delta)
    dPzx_dx = (np.roll(Pzx, -1, axis=-1) - np.roll(Pzx, 1, axis=-1)) / (2.0 * delta)

    divPi = np.stack(
        [dPxx_dx, dPxy_dx, dPzx_dx],
        axis=-1,
    )
    return -np.cross(Gamma, B, axis=-1) + divPi


def _build_source_term_2d(B, M, delta):
    """Compute ``S = -Gamma cross B + div Pi`` in 2D.

    ``B`` and ``M`` both have shape ``(Ny, Nx, ...)`` with the trailing
    component axis (3 for ``B``, 10 for ``M``).
    """
    Gamma = M[..., 1:4]
    Pxx, Pyy = M[..., 4], M[..., 5]
    Pxy, Pyz, Pzx = M[..., 7], M[..., 8], M[..., 9]

    dPxx_dx = (np.roll(Pxx, -1, axis=-1) - np.roll(Pxx, 1, axis=-1)) / (2.0 * delta)
    dPxy_dx = (np.roll(Pxy, -1, axis=-1) - np.roll(Pxy, 1, axis=-1)) / (2.0 * delta)
    dPzx_dx = (np.roll(Pzx, -1, axis=-1) - np.roll(Pzx, 1, axis=-1)) / (2.0 * delta)
    dPyy_dy = (np.roll(Pyy, -1, axis=-2) - np.roll(Pyy, 1, axis=-2)) / (2.0 * delta)
    dPxy_dy = (np.roll(Pxy, -1, axis=-2) - np.roll(Pxy, 1, axis=-2)) / (2.0 * delta)
    dPyz_dy = (np.roll(Pyz, -1, axis=-2) - np.roll(Pyz, 1, axis=-2)) / (2.0 * delta)

    divPi = np.stack(
        [dPxx_dx + dPxy_dy, dPxy_dx + dPyy_dy, dPzx_dx + dPyz_dy],
        axis=-1,
    )
    return -np.cross(Gamma, B, axis=-1) + divPi


# -*- Public solvers -*-


def solve_ohm_1d(
    L,
    S,
    delta,
    *,
    c=1.0,
    rtol=1.0e-12,
    maxiter=1000,
    M=None,
    return_info=False,
):
    """Solve the 1D periodic generalized Ohm's law.

    The ``E_x`` equation is the pointwise division ``E_x = S_x / Lambda``
    (the curl-curl operator vanishes for this component in 1D). The
    ``E_y`` and ``E_z`` equations are identical circulant systems
    ``(Lambda - c^2 d^2/dx^2) E^alpha = S^alpha`` discretized on a
    uniform grid of spacing ``delta`` and solved with CG.

    ``L`` has shape ``(Nx,)``, ``S`` has shape ``(Nx, 3)``.
    """
    L = np.asarray(L, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if L.ndim != 1:
        raise ValueError(f"L must be 1D with shape (Nx,), got {L.shape}")
    if S.shape != (L.shape[0], 3):
        raise ValueError(f"S must have shape (Nx, 3) with Nx=L.shape[0], got {S.shape}")

    c2_dx2 = c * c / (delta * delta)
    E = np.empty((L.shape[0], 3), dtype=np.float64)
    E[:, 0] = S[:, 0] / L
    E[:, 1], status_y, niter_y = _solve_ohm_1d_core(
        L, S[:, 1], c2_dx2, rtol, maxiter, M
    )
    E[:, 2], status_z, niter_z = _solve_ohm_1d_core(
        L, S[:, 2], c2_dx2, rtol, maxiter, M
    )

    if not return_info:
        return E
    return E, {
        "status_yz": (status_y, status_z),
        "niter": max(niter_y, niter_z),
        "niter_yz": (niter_y, niter_z),
    }


def solve_ohm_2d(
    L,
    S,
    delta,
    *,
    c=1.0,
    rtol=1.0e-12,
    maxiter=1000,
    M1=None,
    M2=None,
    base1=None,
    base2=None,
    return_info=False,
):
    """Solve the 2D periodic generalized Ohm's law in the x-y plane.

    The ``(E_x, E_y)`` equations are coupled through a ``2N x 2N`` block
    matrix (``N = N_x N_y``); ``E_z`` is decoupled in an ``N x N``
    matrix. Both are solved by CG with optional external preconditioning.

    ``L`` has shape ``(Ny, Nx)``, ``S`` has shape ``(Ny, Nx, 3)``.
    ``base1`` and ``base2`` may be supplied from :func:`build_ohm_bases_from_grid`
    to reuse the Lambda-independent stencil across repeated solves.
    """
    L = np.asarray(L, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError(f"L must be 2D with shape (Ny, Nx), got {L.shape}")
    if S.shape != (*L.shape, 3):
        raise ValueError(f"S must have shape (Ny, Nx, 3) matching L, got {S.shape}")

    c2_dx2 = c * c / (delta * delta)
    c2_dx4 = c2_dx2 / 4.0
    E, status_1, status_2, niter_1, niter_2 = _solve_ohm_2d_core(
        L, S, c2_dx2, c2_dx4, rtol, maxiter, M1, M2, base1, base2
    )

    if not return_info:
        return E
    return E, {
        "status_1": status_1,
        "status_2": status_2,
        "niter_1": niter_1,
        "niter_2": niter_2,
    }


def solve_ohm_2d_gauss_reduced(
    L,
    S,
    rho,
    delta,
    *,
    c=1.0,
    solver="cg",
    preconditioner="fft",
    rtol=1.0e-12,
    maxiter=1000,
    base=None,
    matrix=None,
    validate_matrix=True,
    fft_lambda=None,
    min_lambda=0.0,
    constant_rtol=1.0e-12,
    constant_atol=0.0,
    return_info=False,
):
    """Solve the 2D periodic Gauss-law-reduced generalized Ohm's law.

    This solves ``(Lambda - c^2 Laplacian) E = S - c^2 grad(rho)`` with
    the same scalar finite-difference operator for all three components.
    ``solver`` may be ``"cg"``, ``"fft"`` (constant Lambda only), or
    ``"splu"`` (small-problem reference). For CG, ``preconditioner`` may
    default to ``"fft"``; it may be set to ``None``, ``"amg"``, or an
    external ``LinearOperator``. Direct FFT and sparse-LU methods ignore the
    default FFT preconditioner.
    A supplied matrix is checked against the requested grid and coefficients
    unless ``validate_matrix=False`` is selected for a trusted hot path.
    """
    L = _validate_lambda_2d(L, min_lambda)
    S = np.asarray(S, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    _validate_grid_parameters(delta, c)
    if S.shape != (*L.shape, 3):
        raise ValueError(f"S must have shape (Ny, Nx, 3) matching L, got {S.shape}")
    if rho.shape != L.shape:
        raise ValueError(f"rho must have shape (Ny, Nx) matching L, got {rho.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho must contain only finite values")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be finite and positive, got {rtol}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter}")
    if base is not None and matrix is not None:
        raise ValueError("base and matrix cannot both be supplied")
    if not np.isfinite(constant_rtol) or constant_rtol < 0.0:
        raise ValueError("constant_rtol must be finite and non-negative")
    if not np.isfinite(constant_atol) or constant_atol < 0.0:
        raise ValueError("constant_atol must be finite and non-negative")

    expected_shape = (L.size, L.size)
    if matrix is not None:
        if validate_matrix:
            matrix = _validate_gauss_matrix_2d(matrix, L, delta, c)
        else:
            if not sparse.issparse(matrix):
                raise TypeError("matrix must be a SciPy sparse matrix")
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"matrix shape {matrix.shape} must be {expected_shape}"
                )

    solver = str(solver).lower()
    if solver not in {"cg", "fft", "splu"}:
        raise ValueError(f"unknown solver {solver!r}; expected 'cg', 'fft', or 'splu'")
    if solver == "fft" and (base is not None or matrix is not None):
        raise ValueError("solver='fft' does not use base or matrix")

    rhs = _build_gauss_reduced_rhs_2d(S, rho, delta, c)
    A = matrix
    statuses = (0, 0, 0)
    iterations = (0, 0, 0)
    preconditioner_name = "none"

    if solver == "fft":
        if preconditioner not in (None, "none", "fft"):
            raise ValueError("solver='fft' does not accept a preconditioner")
        E = _solve_gauss_fft(rhs, L, delta, c, constant_rtol, constant_atol)
    else:
        if A is None:
            A = assemble_ohm_gauss_matrix_2d(
                L, delta, c=c, base=base, min_lambda=min_lambda
            )

        if solver == "splu":
            if preconditioner not in (None, "none", "fft"):
                raise ValueError("solver='splu' does not accept a preconditioner")
            factorization = splu(A.tocsc())
            solution = factorization.solve(rhs.reshape((-1, 3), order="C"))
            E = solution.reshape((*L.shape, 3), order="C")
        else:
            M = None
            if isinstance(preconditioner, str):
                preconditioner_name = preconditioner.lower()
                if preconditioner_name == "none":
                    preconditioner_name = "none"
                elif preconditioner_name in {"amg", "fft"}:
                    M = build_ohm_gauss_preconditioner_2d(
                        L,
                        delta,
                        c=c,
                        kind=preconditioner_name,
                        matrix=A if preconditioner_name == "amg" else None,
                        validate_matrix=validate_matrix,
                        fft_lambda=fft_lambda,
                        min_lambda=min_lambda,
                    )
                else:
                    raise ValueError(
                        f"unknown preconditioner {preconditioner!r}; "
                        "expected None, 'amg', or 'fft'"
                    )
            elif preconditioner is not None:
                if not isinstance(preconditioner, LinearOperator):
                    raise TypeError("external preconditioner must be a LinearOperator")
                if preconditioner.shape != expected_shape:
                    raise ValueError(
                        f"preconditioner shape {preconditioner.shape} must be {expected_shape}"
                    )
                M = preconditioner
                preconditioner_name = "external"

            E = np.empty((*L.shape, 3), dtype=np.float64)
            status_list = []
            iteration_list = []
            for component in range(3):
                solution, status, niter = _cg_solve(
                    A,
                    rhs[..., component].flatten(order="C"),
                    rtol,
                    maxiter,
                    M,
                )
                E[..., component] = solution.reshape(L.shape, order="C")
                status_list.append(status)
                iteration_list.append(niter)
            statuses = tuple(status_list)
            iterations = tuple(iteration_list)

    if not return_info:
        return E
    return E, {
        "status": statuses,
        "niter": iterations,
        "relative_residual": _relative_residuals(L, rhs, E, delta, c),
        "solver": solver,
        "preconditioner": preconditioner_name,
    }


# -*- Run-based dispatch -*-


def _resolve_qm(run, qm_per_species):
    if qm_per_species is not None:
        return np.asarray(qm_per_species, dtype=np.float64)
    if getattr(run, "qm", None) is not None:
        return np.asarray(run.qm, dtype=np.float64)
    return qm_per_species_from_config(run.config)


def calc_e_ohm_1d(run, step, *, prefix="field", c=1.0, qm_per_species=None):
    """Reconstruct the electric field from a 1D picnix run snapshot.

    Reads the field and moment data at ``step``, infers per-species
    ``q_s / m_s`` from the config (or uses ``qm_per_species`` if given),
    builds the source term, and solves the 1D Ohm's law.
    """
    data = run.read_at(prefix, step)
    uf = data["uf"]
    um = data["um"]

    # picnix stores a 1D run as ``(nc, 1, Nx, 6)`` and ``(nc, 1, Nx, Ns, 14)``.
    B = uf[0, 0, ..., 3:6]
    um_collapsed = um[0, 0]

    qm = _resolve_qm(run, qm_per_species)
    M = transform_moments(um_collapsed, qm)
    S = _build_source_term_1d(B, M, float(run.config["parameter"]["delh"]))

    return solve_ohm_1d(M[..., 0], S, float(run.config["parameter"]["delh"]), c=c)


def calc_e_ohm_2d(run, step, *, prefix="field", c=1.0, qm_per_species=None):
    """Reconstruct the electric field from a 2D picnix run snapshot.

    Reads the field and moment data at ``step``, infers per-species
    ``q_s / m_s`` from the config (or uses ``qm_per_species`` if given),
    builds the source term, and solves the 2D Ohm's law.
    """
    data = run.read_at(prefix, step)
    uf = data["uf"]
    um = data["um"]

    # picnix stores a 2D run as ``(nc, Ny, Nx, 6)`` and ``(nc, Ny, Nx, Ns, 14)``.
    B = uf.mean(axis=0)[..., 3:6]
    um_collapsed = um.mean(axis=0)

    qm = _resolve_qm(run, qm_per_species)
    M = transform_moments(um_collapsed, qm)
    S = _build_source_term_2d(B, M, float(run.config["parameter"]["delh"]))

    return solve_ohm_2d(M[..., 0], S, float(run.config["parameter"]["delh"]), c=c)
