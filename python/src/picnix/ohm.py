#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generalized Ohm's law solver for picnix.

Solves

.. math::

    (\\Lambda - c^2 \\nabla^2) \\boldsymbol{E}
    = -\\frac{\\Gamma}{c} \\times \\boldsymbol{B}
    + \\nabla \\cdot \\Pi - c^2 \\nabla \\rho,

with periodic boundary conditions, using a scalar reaction-diffusion
operator for all three electric-field components. The quantities

.. math::

    \\Lambda = \\sum_s \\left(\\frac{q_s}{m_s}\\right)^2
                   \\int f_s\\, d\\boldsymbol{v}, \\qquad
    \\Gamma = \\sum_s \\left(\\frac{q_s}{m_s}\\right)^2
                   \\int \\boldsymbol{v}\\, f_s\\, d\\boldsymbol{v}, \\qquad
    \\Pi    = \\sum_s \\frac{q_s}{m_s}
                   \\int \\boldsymbol{v}\\boldsymbol{v}\\, f_s\\, d\\boldsymbol{v},

are obtained from the per-species moment data emitted by PIC-NIX via
:func:`transform_moments`, which also returns the charge density
:math:`\\rho = \\sum_s (q_s/m_s) \\, um_{s,0}`.

Two low-level solvers are provided:

* :func:`solve_ohm_1d` for 1D
* :func:`solve_ohm_2d` for 2D in the x-y plane

Both expect a reduced source term

.. math::

    \\boldsymbol{S}_{\\text{reduced}}
    = \\boldsymbol{S}_{\\text{original}} - c^2 \\nabla \\rho

and solve the same scalar elliptic operator for all components.  The
high-level helpers :func:`calc_e_ohm_1d` and :func:`calc_e_ohm_2d`
build the full reduced source from raw moments and magnetic field
before calling the low-level solvers.

Preconditioners are built by the caller and passed as a
``LinearOperator`` (or ``None``).  The module provides FFT-based
preconditioner builders :func:`_build_fft_preconditioner_1d` and
:func:`_build_fft_preconditioner_2d` for this purpose.
"""

import inspect

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

_CG_TOL_KWARG = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"


__all__ = [
    "solve_ohm_1d",
    "solve_ohm_2d",
    "calc_e_ohm_1d",
    "calc_e_ohm_2d",
    "transform_moments",
]


# -*- Periodic difference operators -*-


def _periodic_second_difference_matrix(n):
    """Periodic 1D second-difference matrix (n x n), no 1/dx^2 factor."""
    D = sparse.lil_matrix((n, n))
    D.setdiag(-2.0)
    D.setdiag(1.0, k=-1)
    D.setdiag(1.0, k=1)
    D[0, n - 1] = 1.0
    D[n - 1, 0] = 1.0
    return D.tocsr()


# -*- Validation -*-


def _validate_grid_parameters(delta, c):
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be finite and positive, got {delta}")
    if not np.isfinite(c):
        raise ValueError(f"c must be finite, got {c}")


def _validate_lambda_1d(L, min_lambda):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 1:
        raise ValueError(f"L must be 1D with shape (Nx,), got {L.shape}")
    if L.shape[0] < 3:
        raise ValueError(f"L must contain at least 3 points, got {L.shape}")
    if not np.all(np.isfinite(L)):
        raise ValueError("L must contain only finite values")
    if not np.isfinite(min_lambda) or min_lambda < 0.0:
        raise ValueError(
            f"min_lambda must be finite and non-negative, got {min_lambda}"
        )
    if np.min(L) <= min_lambda:
        raise ValueError(f"L must be greater than min_lambda={min_lambda}")
    return L


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


def _validate_ohm_inputs_1d(L, S, delta, c, rtol, maxiter, min_lambda):
    """Validate all inputs for solve_ohm_1d."""
    _validate_grid_parameters(delta, c)
    L = _validate_lambda_1d(L, min_lambda)
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (L.shape[0], 3):
        raise ValueError(f"S must have shape (Nx, 3) with Nx=L.shape[0], got {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be finite and positive, got {rtol}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter}")
    return L, S


def _validate_ohm_inputs_2d(L, S, delta, c, rtol, maxiter, min_lambda):
    """Validate all inputs for solve_ohm_2d."""
    _validate_grid_parameters(delta, c)
    L = _validate_lambda_2d(L, min_lambda)
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (*L.shape, 3):
        raise ValueError(f"S must have shape (Ny, Nx, 3) matching L, got {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be finite and positive, got {rtol}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter}")
    return L, S


def _validate_preconditioner(M, expected_size):
    """Validate preconditioner is a LinearOperator or None with correct shape."""
    if M is None:
        return None
    if not isinstance(M, LinearOperator):
        raise TypeError(
            f"preconditioner must be a LinearOperator or None, got {type(M)}"
        )
    if M.shape != (expected_size, expected_size):
        raise ValueError(
            f"preconditioner shape {M.shape} must be ({expected_size}, {expected_size})"
        )
    return M


# -*- Laplacian matrix construction -*-


def _build_laplacian_2d(Nx, Ny, delta, *, c=1.0):
    """Build the periodic ``-c^2 Laplacian`` sparse matrix."""
    if Nx < 3 or Ny < 3:
        raise ValueError(f"Nx and Ny must both be at least 3, got Nx={Nx}, Ny={Ny}")
    _validate_grid_parameters(delta, c)

    Ix = sparse.eye(Nx, format="csr")
    Iy = sparse.eye(Ny, format="csr")
    Dxx = _periodic_second_difference_matrix(Nx)
    Dyy = _periodic_second_difference_matrix(Ny)
    laplacian = sparse.kron(Iy, Dxx, format="csr") + sparse.kron(Dyy, Ix, format="csr")
    return (-(c * c) / (delta * delta)) * laplacian


def _assemble_ohm_matrix_1d(L, delta, *, c=1.0, min_lambda=0.0):
    """Assemble ``diag(Lambda) - c^2 d^2/dx^2`` for a periodic 1D grid."""
    L = _validate_lambda_1d(L, min_lambda)
    _validate_grid_parameters(delta, c)
    N = L.shape[0]
    D2 = _periodic_second_difference_matrix(N)
    return (-(c * c) / (delta * delta)) * D2 + sparse.diags(L, format="csr")


def _assemble_ohm_matrix_2d(L, delta, *, c=1.0, min_lambda=0.0):
    """Assemble ``diag(Lambda) - c^2 Laplacian`` for a periodic 2D grid."""
    L = _validate_lambda_2d(L, min_lambda)
    _validate_grid_parameters(delta, c)
    Ny, Nx = L.shape
    base = _build_laplacian_2d(Nx, Ny, delta, c=c)
    return base + sparse.diags(L.flatten(order="C"), format="csr")


# -*- FFT preconditioners -*-


def _fft_denominator_1d(N, delta, c, lambda0):
    kx = np.fft.rfftfreq(N)
    kappa2 = (4.0 / (delta * delta)) * np.sin(np.pi * kx) ** 2
    return lambda0 + c * c * kappa2


def _build_fft_preconditioner_1d(N, delta, c, lambda0):
    denominator = _fft_denominator_1d(N, delta, c, lambda0)

    def matvec(vector):
        values = np.asarray(vector, dtype=np.float64)
        transformed = np.fft.rfft(values)
        return np.fft.irfft(transformed / denominator, n=N)

    return LinearOperator((N, N), matvec=matvec, dtype=np.float64)


def _fft_denominator_2d(shape, delta, c, lambda0):
    Ny, Nx = shape
    fy = np.fft.fftfreq(Ny)
    fx = np.fft.rfftfreq(Nx)
    kappa2 = (4.0 / (delta * delta)) * (
        np.sin(np.pi * fy[:, None]) ** 2 + np.sin(np.pi * fx[None, :]) ** 2
    )
    return lambda0 + c * c * kappa2


def _build_fft_preconditioner_2d(shape, delta, c, lambda0):
    denominator = _fft_denominator_2d(shape, delta, c, lambda0)
    size = shape[0] * shape[1]

    def matvec(vector):
        values = np.asarray(vector, dtype=np.float64).reshape(shape, order="C")
        transformed = np.fft.rfftn(values, axes=(0, 1))
        result = np.fft.irfftn(transformed / denominator, s=shape, axes=(0, 1))
        return result.flatten(order="C")

    return LinearOperator((size, size), matvec=matvec, dtype=np.float64)


# -*- CG wrapper -*-


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


# -*- Operator application -*-


def _apply_ohm_operator_1d(L, values, delta, c):
    laplacian = (np.roll(values, -1) + np.roll(values, 1) - 2.0 * values) / (
        delta * delta
    )
    multiplier = L if values.ndim == 1 else L[:, None]
    return multiplier * values - c * c * laplacian


def _apply_ohm_operator_2d(L, values, delta, c):
    laplacian = (
        np.roll(values, -1, axis=1)
        + np.roll(values, 1, axis=1)
        + np.roll(values, -1, axis=0)
        + np.roll(values, 1, axis=0)
        - 4.0 * values
    ) / (delta * delta)
    multiplier = L if values.ndim == 2 else L[..., None]
    return multiplier * values - c * c * laplacian


def _build_ohm_operator_2d(L, delta, c):
    shape = L.shape
    size = shape[0] * shape[1]

    def matvec(vector):
        values = vector.reshape(shape, order="C")
        result = _apply_ohm_operator_2d(L, values, delta, c)
        return result.flatten(order="C")

    return LinearOperator((size, size), matvec=matvec, dtype=np.float64)


# -*- Relative residuals -*-


def _relative_residuals_1d(L, rhs, solution, delta, c):
    eps = np.finfo(np.float64).eps
    residual = rhs - _apply_ohm_operator_1d(L, solution, delta, c)
    residuals = []
    for component in range(3):
        b = rhs[..., component].flatten()
        component_residual = residual[..., component].flatten()
        residuals.append(
            float(np.linalg.norm(component_residual) / max(np.linalg.norm(b), eps))
        )
    return tuple(residuals)


def _relative_residuals_2d(L, rhs, solution, delta, c):
    eps = np.finfo(np.float64).eps
    residual = rhs - _apply_ohm_operator_2d(L, solution, delta, c)
    residuals = []
    for component in range(3):
        b = rhs[..., component].flatten(order="C")
        component_residual = residual[..., component].flatten(order="C")
        residuals.append(
            float(np.linalg.norm(component_residual) / max(np.linalg.norm(b), eps))
        )
    return tuple(residuals)


# -*- Public solvers -*-


def solve_ohm_1d(
    L,
    S,
    delta,
    *,
    c=1.0,
    M=None,
    rtol=1.0e-12,
    maxiter=1000,
    min_lambda=0.0,
    return_info=False,
):
    """Solve the 1D periodic reduced Ohm's law.

    .. math::

        (\\Lambda - c^2 d^2\\!/dx^2) \\boldsymbol{E}
        = \\boldsymbol{S}

    Parameters
    ----------
    L : (Nx,) float array
        Scalar Lambda field.
    S : (Nx, 3) float array
        Reduced source term.
    delta : float
        Grid spacing.
    c : float, optional
        Speed of light (default 1.0).
    M : LinearOperator or None, optional
        Preconditioner for CG.  ``None`` uses unpreconditioned CG.
        Build an FFT preconditioner with
        :func:`_build_fft_preconditioner_1d`.
    rtol : float, optional
        Relative tolerance for CG convergence.
    maxiter : int, optional
        Maximum CG iterations.
    min_lambda : float, optional
        Reject Lambda values at or below this threshold
        (default 0.0).
    return_info : bool, optional
        If True, return ``(E, info_dict)``.

    Returns
    -------
    E : (Nx, 3) float array
        Solved electric-field components.
    info : dict, optional
        Convergence diagnostics.
    """
    L, S = _validate_ohm_inputs_1d(L, S, delta, c, rtol, maxiter, min_lambda)
    N = L.shape[0]
    M = _validate_preconditioner(M, N)

    A = _assemble_ohm_matrix_1d(L, delta, c=c)

    E = np.empty((N, 3), dtype=np.float64)
    status_list = []
    niter_list = []
    for comp in range(3):
        solution, status, niter = _cg_solve(A, S[:, comp], rtol, maxiter, M)
        E[:, comp] = solution
        status_list.append(status)
        niter_list.append(niter)

    if not return_info:
        return E
    return E, {
        "status": tuple(status_list),
        "niter": tuple(niter_list),
        "relative_residual": _relative_residuals_1d(L, S, E, delta, c),
        "solver": "cg",
        "preconditioner": "fft" if M is not None else "none",
    }


def solve_ohm_2d(
    L,
    S,
    delta,
    *,
    c=1.0,
    M=None,
    rtol=1.0e-12,
    maxiter=1000,
    min_lambda=0.0,
    return_info=False,
):
    """Solve the 2D periodic reduced Ohm's law in the x-y plane.

    .. math::

        (\\Lambda - c^2 \\nabla^2) \\boldsymbol{E}
        = \\boldsymbol{S}

    Parameters
    ----------
    L : (Ny, Nx) float array
        Scalar Lambda field.
    S : (Ny, Nx, 3) float array
        Reduced source term.
    delta : float
        Grid spacing.
    c : float, optional
        Speed of light (default 1.0).
    M : LinearOperator or None, optional
        Preconditioner for CG.  ``None`` uses unpreconditioned CG.
        Build an FFT preconditioner with
        :func:`_build_fft_preconditioner_2d`.
    rtol : float, optional
        Relative tolerance for CG convergence.
    maxiter : int, optional
        Maximum CG iterations.
    min_lambda : float, optional
        Reject Lambda values at or below this threshold
        (default 0.0).
    return_info : bool, optional
        If True, return ``(E, info_dict)``.

    Returns
    -------
    E : (Ny, Nx, 3) float array
        Solved electric-field components.
    info : dict, optional
        Convergence diagnostics.
    """
    L, S = _validate_ohm_inputs_2d(L, S, delta, c, rtol, maxiter, min_lambda)
    M = _validate_preconditioner(M, L.size)

    A = _assemble_ohm_matrix_2d(L, delta, c=c)

    E = np.empty((*L.shape, 3), dtype=np.float64)
    status_list = []
    niter_list = []
    for comp in range(3):
        solution, status, niter = _cg_solve(
            A, S[..., comp].flatten(order="C"), rtol, maxiter, M
        )
        E[..., comp] = solution.reshape(L.shape, order="C")
        status_list.append(status)
        niter_list.append(niter)

    if not return_info:
        return E
    return E, {
        "status": tuple(status_list),
        "niter": tuple(niter_list),
        "relative_residual": _relative_residuals_2d(L, S, E, delta, c),
        "solver": "cg",
        "preconditioner": "fft" if M is not None else "none",
    }


# -*- Per-species moment transformation -*-


def transform_moments(um, qm):
    """Build transformed moments and charge density from raw moments.

    ``um`` has shape ``(..., Ns, 14)`` following the PIC-NIX moment
    layout (see ``engine/moment.hpp``):

    * indices 0-3: Lambda-, Gamma_x-, Gamma_y-, Gamma_z-weights
    * indices 5-7: P_xx, P_yy, P_zz (diagonal pressure)
    * indices 11-13: P_xy, P_yz, P_zx (off-diagonal pressure)

    ``qm`` has shape ``(Ns,)``.

    Returns
    -------
    L : (...) float array
        Lambda ``sum_s (q_s/m_s)^2 * um_{s,0}``.
    G : (..., 3) float array
        Gamma ``[Gamma_x, Gamma_y, Gamma_z]``, each weighted by ``(q/m)^2``.
    P : (..., 6) float array
        Pressure tensor ``[P_xx, P_yy, P_zz, P_xy, P_yz, P_zx]``,
        each weighted by ``q/m``.
    R : (...) float array
        Charge density ``sum_s (q_s/m_s) * um_{s,0}``.
    """
    qm = np.asarray(qm, dtype=np.float64)
    qm2 = qm**2

    L = (um[..., 0] * qm2).sum(axis=-1)
    G = (um[..., 1:4] * (qm2[:, None])).sum(axis=-2)
    P = np.empty(um.shape[:-2] + (6,), dtype=np.float64)
    P[..., 0:3] = (um[..., 5:8] * qm[:, None]).sum(axis=-2)
    P[..., 3:6] = (um[..., 11:14] * qm[:, None]).sum(axis=-2)
    R = (um[..., 0] * qm).sum(axis=-1)

    return L, G, P, R


# -*- Source term from (B, M, rho) -*-


def _build_source_1d(B, G, P, R, delta, c):
    """Compute the reduced source ``S = -G x B / c + div(P) - c^2 d(R)/dx``."""
    dPxx_dx = (np.roll(P[..., 0], -1, axis=-1) - np.roll(P[..., 0], 1, axis=-1)) / (
        2.0 * delta
    )
    dPxy_dx = (np.roll(P[..., 3], -1, axis=-1) - np.roll(P[..., 3], 1, axis=-1)) / (
        2.0 * delta
    )
    dPzx_dx = (np.roll(P[..., 5], -1, axis=-1) - np.roll(P[..., 5], 1, axis=-1)) / (
        2.0 * delta
    )

    divPi = np.stack([dPxx_dx, dPxy_dx, dPzx_dx], axis=-1)
    S = -np.cross(G, B, axis=-1) / c + divPi

    dR_dx = (np.roll(R, -1, axis=-1) - np.roll(R, 1, axis=-1)) / (2.0 * delta)
    S[..., 0] -= c * c * dR_dx
    return S


def _build_source_2d(B, G, P, R, delta, c):
    """Compute the reduced source ``S = -G x B / c + div(P) - c^2 grad(R)``."""
    dPxx_dx = (np.roll(P[..., 0], -1, axis=-1) - np.roll(P[..., 0], 1, axis=-1)) / (
        2.0 * delta
    )
    dPxy_dx = (np.roll(P[..., 3], -1, axis=-1) - np.roll(P[..., 3], 1, axis=-1)) / (
        2.0 * delta
    )
    dPzx_dx = (np.roll(P[..., 5], -1, axis=-1) - np.roll(P[..., 5], 1, axis=-1)) / (
        2.0 * delta
    )
    dPyy_dy = (np.roll(P[..., 1], -1, axis=-2) - np.roll(P[..., 1], 1, axis=-2)) / (
        2.0 * delta
    )
    dPxy_dy = (np.roll(P[..., 3], -1, axis=-2) - np.roll(P[..., 3], 1, axis=-2)) / (
        2.0 * delta
    )
    dPyz_dy = (np.roll(P[..., 4], -1, axis=-2) - np.roll(P[..., 4], 1, axis=-2)) / (
        2.0 * delta
    )

    divPi = np.stack(
        [dPxx_dx + dPxy_dy, dPxy_dx + dPyy_dy, dPzx_dx + dPyz_dy],
        axis=-1,
    )
    S = -np.cross(G, B, axis=-1) / c + divPi

    dR_dx = (np.roll(R, -1, axis=-1) - np.roll(R, 1, axis=-1)) / (2.0 * delta)
    dR_dy = (np.roll(R, -1, axis=-2) - np.roll(R, 1, axis=-2)) / (2.0 * delta)
    S[..., 0] -= c * c * dR_dx
    S[..., 1] -= c * c * dR_dy
    return S


# -*- Run-based dispatch -*-


def calc_e_ohm_1d(run, step, *, prefix="field", c=1.0):
    """Reconstruct the electric field from a 1D picnix run snapshot."""
    if run.qm is None:
        raise ValueError("run.qm is None; cannot reconstruct E without q/m")
    data = run.read_at(prefix, step)
    qm = np.asarray(run.qm, dtype=np.float64)
    dh = float(run.config["parameter"]["delh"])

    B = data["uf"].mean(axis=(0, 1))[..., 3:6]
    L, G, P, R = transform_moments(data["um"].mean(axis=(0, 1)), qm)

    S = _build_source_1d(B, G, P, R, dh, c)
    M_prec = _build_fft_preconditioner_1d(L.shape[0], dh, c, float(np.mean(L)))
    return solve_ohm_1d(L, S, dh, c=c, M=M_prec)


def calc_e_ohm_2d(run, step, *, prefix="field", c=1.0):
    """Reconstruct the electric field from a 2D picnix run snapshot."""
    if run.qm is None:
        raise ValueError("run.qm is None; cannot reconstruct E without q/m")
    data = run.read_at(prefix, step)
    qm = np.asarray(run.qm, dtype=np.float64)
    dh = float(run.config["parameter"]["delh"])

    B = data["uf"].mean(axis=0)[..., 3:6]
    L, G, P, R = transform_moments(data["um"].mean(axis=0), qm)

    S = _build_source_2d(B, G, P, R, dh, c)
    M_prec = _build_fft_preconditioner_2d(L.shape, dh, c, float(np.mean(L)))
    return solve_ohm_2d(L, S, dh, c=c, M=M_prec)
