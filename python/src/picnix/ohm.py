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
:func:`transform_moments`.  Charge density is reconstructed directly
from raw moments as :math:`\\rho = \\sum_s (q_s/m_s) \\, um_{s,0}`.

Two low-level solvers are provided:

* :func:`solve_ohm_1d` for 1D
* :func:`solve_ohm_2d` for 2D in the x-y plane

Both expect a *pre-reduced* source term

.. math::

    \\boldsymbol{S}_{\\text{reduced}}
    = \\boldsymbol{S}_{\\text{original}} - c^2 \\nabla \\rho

and solve the same scalar elliptic operator for all components.  The
high-level helpers :func:`calc_e_ohm_1d` and :func:`calc_e_ohm_2d`
compute :math:`\\rho` from raw moments and build the reduced source
before calling the low-level solvers.
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
    "qm_per_species_from_config",
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


def _assemble_ohm_matrix_2d(L, delta, *, c=1.0, base=None, min_lambda=0.0):
    """Assemble ``diag(Lambda) - c^2 Laplacian`` for a periodic 2D grid."""
    L = _validate_lambda_2d(L, min_lambda)
    _validate_grid_parameters(delta, c)
    Ny, Nx = L.shape
    if base is None:
        base = _build_laplacian_2d(Nx, Ny, delta, c=c)
    else:
        base = _validate_laplacian_2d(base, L.shape, delta, c)
    return base + sparse.diags(L.flatten(order="C"), format="csr")


def _validate_laplacian_2d(base, shape, delta, c):
    Ny, Nx = shape
    expected = _build_laplacian_2d(Nx, Ny, delta, c=c)
    return _validate_sparse_operator(base, expected, "base")


def _validate_ohm_matrix_2d(matrix, L, delta, c):
    Ny, Nx = L.shape
    expected = _build_laplacian_2d(Nx, Ny, delta, c=c)
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


# -*- Reduced RHS (grad-rho correction) -*-


def _build_reduced_rhs_1d(S, rho, delta, c):
    """Return ``S - c^2 grad(rho)`` for 1D periodic grid."""
    d_rho_dx = (np.roll(rho, -1, axis=-1) - np.roll(rho, 1, axis=-1)) / (2.0 * delta)
    rhs = np.array(S, dtype=np.float64, copy=True)
    rhs[..., 0] -= c * c * d_rho_dx
    return rhs


def _build_reduced_rhs_2d(S, rho, delta, c):
    """Return ``S - c^2 grad(rho)`` for 2D periodic grid."""
    d_rho_dx = (np.roll(rho, -1, axis=-1) - np.roll(rho, 1, axis=-1)) / (2.0 * delta)
    d_rho_dy = (np.roll(rho, -1, axis=-2) - np.roll(rho, 1, axis=-2)) / (2.0 * delta)
    rhs = np.array(S, dtype=np.float64, copy=True)
    rhs[..., 0] -= c * c * d_rho_dx
    rhs[..., 1] -= c * c * d_rho_dy
    return rhs


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
    preconditioner="fft",
    rtol=1.0e-12,
    maxiter=1000,
    return_info=False,
):
    """Solve the 1D periodic reduced Ohm's law.

    .. math::

        (\\Lambda - c^2 d^2\\!/dx^2) \\boldsymbol{E}
        = \\boldsymbol{S}

    **S must be pre-reduced**: ``S_original - c^2 grad(rho)``.
    Use :func:`calc_e_ohm_1d` for high-level dispatch from a
    picnix run; it computes ``rho`` from raw moments and reduces
    the source automatically.

    Parameters
    ----------
    L : (Nx,) float array
        Scalar Lambda field.
    S : (Nx, 3) float array
        Pre-reduced source term.
    delta : float
        Grid spacing.
    c : float, optional
        Speed of light (default 1.0).
    preconditioner : ``"fft"`` or ``None``, optional
        ``"fft"`` (default) uses a constant-coefficient FFT
        preconditioner. ``None`` uses unpreconditioned CG.
    rtol : float, optional
        Relative tolerance for CG convergence.
    maxiter : int, optional
        Maximum CG iterations.
    return_info : bool, optional
        If True, return ``(E, info_dict)``.

    Returns
    -------
    E : (Nx, 3) float array
        Solved electric-field components.
    info : dict, optional
        Convergence diagnostics.
    """
    L = _validate_lambda_1d(L, 0.0)
    S = np.asarray(S, dtype=np.float64)
    _validate_grid_parameters(delta, c)
    if S.shape != (L.shape[0], 3):
        raise ValueError(f"S must have shape (Nx, 3) with Nx=L.shape[0], got {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be finite and positive, got {rtol}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter}")

    if isinstance(preconditioner, str):
        preconditioner = preconditioner.lower()
        if preconditioner not in ("fft", "none"):
            raise ValueError(
                f"unknown preconditioner {preconditioner!r}; expected 'fft' or None"
            )
    elif preconditioner is not None:
        raise TypeError("preconditioner must be 'fft', None, or a string")
    use_preconditioner = preconditioner == "fft"

    N = L.shape[0]
    A = _assemble_ohm_matrix_1d(L, delta, c=c)
    M = (
        _build_fft_preconditioner_1d(N, delta, c, float(np.mean(L)))
        if use_preconditioner
        else None
    )

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
        "preconditioner": "fft" if use_preconditioner else "none",
    }


def solve_ohm_2d(
    L,
    S,
    delta,
    *,
    c=1.0,
    preconditioner="fft",
    rtol=1.0e-12,
    maxiter=1000,
    base=None,
    matrix=None,
    validate_matrix=True,
    min_lambda=0.0,
    return_info=False,
):
    """Solve the 2D periodic reduced Ohm's law in the x-y plane.

    .. math::

        (\\Lambda - c^2 \\nabla^2) \\boldsymbol{E}
        = \\boldsymbol{S}

    **S must be pre-reduced**: ``S_original - c^2 grad(rho)``.
    Use :func:`calc_e_ohm_2d` for high-level dispatch from a
    picnix run; it computes ``rho`` from raw moments and reduces
    the source automatically.

    Parameters
    ----------
    L : (Ny, Nx) float array
        Scalar Lambda field.
    S : (Ny, Nx, 3) float array
        Pre-reduced source term.
    delta : float
        Grid spacing.
    c : float, optional
        Speed of light (default 1.0).
    preconditioner : ``"fft"`` or ``None``, optional
        ``"fft"`` (default) uses a constant-coefficient FFT
        preconditioner with the mean of Lambda. ``None`` uses
        unpreconditioned CG.
    rtol : float, optional
        Relative tolerance for CG convergence.
    maxiter : int, optional
        Maximum CG iterations.
    base : sparse matrix or None, optional
        Reusable Lambda-independent ``-c^2 Laplacian`` matrix
        from a previous solve with the same grid.
    matrix : sparse matrix or None, optional
        Reusable full matrix from a previous solve with the
        same ``(L, delta, c)``. Validated by default; set
        ``validate_matrix=False`` for a trusted hot path.
    validate_matrix : bool, optional
        Whether to verify the supplied matrix against the
        requested grid and coefficients.
    min_lambda : float, optional
        Reject Lambda values at or below this threshold.
    return_info : bool, optional
        If True, return ``(E, info_dict)``.

    Returns
    -------
    E : (Ny, Nx, 3) float array
        Solved electric-field components.
    info : dict, optional
        Convergence diagnostics.
    """
    L = _validate_lambda_2d(L, min_lambda)
    S = np.asarray(S, dtype=np.float64)
    _validate_grid_parameters(delta, c)
    if S.shape != (*L.shape, 3):
        raise ValueError(f"S must have shape (Ny, Nx, 3) matching L, got {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be finite and positive, got {rtol}")
    if not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter}")
    if base is not None and matrix is not None:
        raise ValueError("base and matrix cannot both be supplied")

    if isinstance(preconditioner, str):
        preconditioner = preconditioner.lower()
        if preconditioner not in ("fft", "none"):
            raise ValueError(
                f"unknown preconditioner {preconditioner!r}; expected 'fft' or None"
            )
    elif preconditioner is not None:
        raise TypeError("preconditioner must be 'fft', None, or a string")
    use_preconditioner = preconditioner == "fft"

    expected_shape = (L.size, L.size)
    A = matrix
    if A is not None:
        if validate_matrix:
            A = _validate_ohm_matrix_2d(A, L, delta, c)
        elif not sparse.issparse(A) or A.shape != expected_shape:
            raise ValueError(f"matrix shape {A.shape} must be {expected_shape}")

    if use_preconditioner:
        if A is None:
            A = _assemble_ohm_matrix_2d(L, delta, c=c, base=base, min_lambda=min_lambda)
        M = _build_fft_preconditioner_2d(L.shape, delta, c, float(np.mean(L)))
    else:
        if A is None:
            A = _assemble_ohm_matrix_2d(L, delta, c=c, base=base, min_lambda=min_lambda)
        M = None

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
        "preconditioner": "fft" if use_preconditioner else "none",
    }


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
    M[..., 4:7] = (um[..., 5:8] * qm[:, None]).sum(axis=-2)
    M[..., 7:10] = (um[..., 11:14] * qm[:, None]).sum(axis=-2)
    return M


# -*- Source term from (B, M) -*-


def _build_source_1d(B, M, delta, c):
    """Compute ``S = -Gamma x B / c + div(Pi)`` in 1D."""
    Gamma = M[..., 1:4]
    Pxx, Pxy, Pzx = M[..., 4], M[..., 7], M[..., 9]

    dPxx_dx = (np.roll(Pxx, -1, axis=-1) - np.roll(Pxx, 1, axis=-1)) / (2.0 * delta)
    dPxy_dx = (np.roll(Pxy, -1, axis=-1) - np.roll(Pxy, 1, axis=-1)) / (2.0 * delta)
    dPzx_dx = (np.roll(Pzx, -1, axis=-1) - np.roll(Pzx, 1, axis=-1)) / (2.0 * delta)

    divPi = np.stack([dPxx_dx, dPxy_dx, dPzx_dx], axis=-1)
    return -np.cross(Gamma, B, axis=-1) / c + divPi


def _build_source_2d(B, M, delta, c):
    """Compute ``S = -Gamma x B / c + div(Pi)`` in 2D."""
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
    return -np.cross(Gamma, B, axis=-1) / c + divPi


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
    builds the source term, computes charge density from raw moments,
    reduces the source, and solves the 1D Ohm's law.
    """
    data = run.read_at(prefix, step)
    uf = data["uf"]
    um = data["um"]

    B = uf[0, 0, ..., 3:6]
    um_collapsed = um[0, 0]

    qm = _resolve_qm(run, qm_per_species)
    rho = np.sum(um_collapsed[..., 0] * qm, axis=-1)
    M = transform_moments(um_collapsed, qm)
    S = _build_source_1d(B, M, float(run.config["parameter"]["delh"]), c)
    S_reduced = _build_reduced_rhs_1d(S, rho, float(run.config["parameter"]["delh"]), c)

    return solve_ohm_1d(
        M[..., 0], S_reduced, float(run.config["parameter"]["delh"]), c=c
    )


def calc_e_ohm_2d(run, step, *, prefix="field", c=1.0, qm_per_species=None):
    """Reconstruct the electric field from a 2D picnix run snapshot.

    Reads the field and moment data at ``step``, infers per-species
    ``q_s / m_s`` from the config (or uses ``qm_per_species`` if given),
    builds the source term, computes charge density from raw moments,
    reduces the source, and solves the 2D Ohm's law.
    """
    data = run.read_at(prefix, step)
    uf = data["uf"]
    um = data["um"]

    B = uf.mean(axis=0)[..., 3:6]
    um_collapsed = um.mean(axis=0)

    qm = _resolve_qm(run, qm_per_species)
    rho = np.sum(um_collapsed[..., 0] * qm, axis=-1)
    M = transform_moments(um_collapsed, qm)
    S = _build_source_2d(B, M, float(run.config["parameter"]["delh"]), c)
    S_reduced = _build_reduced_rhs_2d(S, rho, float(run.config["parameter"]["delh"]), c)

    return solve_ohm_2d(
        M[..., 0], S_reduced, float(run.config["parameter"]["delh"]), c=c
    )
