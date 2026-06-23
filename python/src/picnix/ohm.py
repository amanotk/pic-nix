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
optional pyamg AMG preconditioning.
"""

import numpy as np
import pyamg
from scipy import sparse


__all__ = [
    "solve_ohm_1d",
    "solve_ohm_2d",
    "calc_e_ohm_1d",
    "calc_e_ohm_2d",
    "transform_moments",
    "qm_per_species_from_config",
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


# -*- Solver cores -*-


def _solve_ohm_1d_core(L, S_yz, c2_dx2, rtol, maxiter, M):
    """Solve the circulant 1D system ``(Lambda - c^2 D^2) E = S`` once.

    Used for both the ``E_y`` and ``E_z`` components.
    """
    N = L.shape[0]
    D2 = _periodic_second_difference_matrix(N)
    A = (-c2_dx2) * D2 + sparse.diags(L, format="csr")
    if M is None:
        M = pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle="V")
    x, status = pyamg.krylov.cg(A, S_yz, M=M, tol=rtol, maxiter=maxiter)
    return x, int(status)


def _build_ohm_matrices_2d(Nx, Ny, L, c2_dx2, c2_dx4):
    """Build the coupled (Ex, Ey) and decoupled Ez sparse matrices."""
    Ix = sparse.eye(Nx, format="csr")
    Iy = sparse.eye(Ny, format="csr")
    Dx1 = _periodic_first_difference_matrix(Nx)
    Dxx = _periodic_second_difference_matrix(Nx)
    Dy1 = _periodic_first_difference_matrix(Ny)
    Dyy = _periodic_second_difference_matrix(Ny)
    Lx = sparse.kron(Iy, Dxx, format="csr")
    Ly = sparse.kron(Dyy, Ix, format="csr")
    Cxy = sparse.kron(Dy1, Dx1, format="csr")

    L_flat = L.flatten(order="C")
    A1 = sparse.bmat(
        [[(-c2_dx2) * Ly, c2_dx4 * Cxy], [c2_dx4 * Cxy.T, (-c2_dx2) * Lx]],
        format="csr",
    ) + sparse.diags(np.concatenate((L_flat, L_flat)), format="csr")
    A2 = (-c2_dx2) * (Lx + Ly) + sparse.diags(L_flat, format="csr")
    return A1, A2


def _solve_ohm_2d_core(L, S, c2_dx2, c2_dx4, rtol, maxiter, M1, M2):
    """Solve the 2D periodic system, returning ``E`` of shape ``(Ny, Nx, 3)``."""
    Ny, Nx = L.shape
    N = Nx * Ny
    A1, A2 = _build_ohm_matrices_2d(Nx, Ny, L, c2_dx2, c2_dx4)

    S1 = np.concatenate((S[..., 0].flatten(order="C"), S[..., 1].flatten(order="C")))
    S2 = S[..., 2].flatten(order="C")

    if M1 is None:
        M1 = pyamg.smoothed_aggregation_solver(A1).aspreconditioner(cycle="V")
    if M2 is None:
        M2 = pyamg.smoothed_aggregation_solver(A2).aspreconditioner(cycle="V")

    E1, status_1 = pyamg.krylov.cg(A1, S1, M=M1, tol=rtol, maxiter=maxiter)
    E2, status_2 = pyamg.krylov.cg(A2, S2, M=M2, tol=rtol, maxiter=maxiter)

    E = np.zeros((Ny, Nx, 3), dtype=np.float64)
    E[..., 0] = E1[:N].reshape((Ny, Nx), order="C")
    E[..., 1] = E1[N:].reshape((Ny, Nx), order="C")
    E[..., 2] = E2.reshape((Ny, Nx), order="C")
    return E, int(status_1), int(status_2)


# -*- Per-species q/m resolution and moment transformation -*-


def qm_per_species_from_config(config):
    """Infer per-species charge-to-mass ratio from a picnix config.

    Supports two config schemas used in PIC-NIX examples:

    * ``[[parameter.particle]]`` (array of per-species blocks, each with
      a ``qm`` field; e.g., ``beam/twostream``, ``foot/``)
    * Top-level ``mime``, ``nppc``, ``wp`` keys, which implies a 2-species
      electron-ion pair (e.g., ``anisotropy``, ``heatflux``). Returns
      ``[-wp, +wp/mime]`` in code units (matches the C++ side which
      sets ``me = 1/nppc``, ``qe = -wp/nppc`` so ``qme = qe/me = -wp``).

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

    dPxx_dx = (np.roll(Pxx, -1, axis=-2) - np.roll(Pxx, 1, axis=-2)) / (2.0 * delta)
    dPxy_dx = (np.roll(Pxy, -1, axis=-2) - np.roll(Pxy, 1, axis=-2)) / (2.0 * delta)
    dPzx_dx = (np.roll(Pzx, -1, axis=-2) - np.roll(Pzx, 1, axis=-2)) / (2.0 * delta)
    dPyy_dy = (np.roll(Pyy, -1, axis=-1) - np.roll(Pyy, 1, axis=-1)) / (2.0 * delta)
    dPxy_dy = (np.roll(Pxy, -1, axis=-1) - np.roll(Pxy, 1, axis=-1)) / (2.0 * delta)
    dPyz_dy = (np.roll(Pyz, -1, axis=-1) - np.roll(Pyz, 1, axis=-1)) / (2.0 * delta)

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
    E[:, 1], status_y = _solve_ohm_1d_core(L, S[:, 1], c2_dx2, rtol, maxiter, M)
    E[:, 2], status_z = _solve_ohm_1d_core(L, S[:, 2], c2_dx2, rtol, maxiter, M)

    if not return_info:
        return E
    return E, {"status_yz": (status_y, status_z), "niter": maxiter}


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
    return_info=False,
):
    """Solve the 2D periodic generalized Ohm's law in the x-y plane.

    The ``(E_x, E_y)`` equations are coupled through a ``2N x 2N`` block
    matrix (``N = N_x N_y``); ``E_z`` is decoupled in an ``N x N``
    matrix. Both are solved by CG with optional pyamg AMG preconditioning.

    ``L`` has shape ``(Ny, Nx)``, ``S`` has shape ``(Ny, Nx, 3)``.
    """
    L = np.asarray(L, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError(f"L must be 2D with shape (Ny, Nx), got {L.shape}")
    if S.shape != (*L.shape, 3):
        raise ValueError(f"S must have shape (Ny, Nx, 3) matching L, got {S.shape}")

    c2_dx2 = c * c / (delta * delta)
    c2_dx4 = c2_dx2 / 4.0
    E, status_1, status_2 = _solve_ohm_2d_core(
        L, S, c2_dx2, c2_dx4, rtol, maxiter, M1, M2
    )

    if not return_info:
        return E
    return E, {
        "status_1": status_1,
        "status_2": status_2,
        "niter_1": maxiter,
        "niter_2": maxiter,
    }


# -*- Run-based dispatch -*-


def _resolve_qm(run, qm_per_species):
    if qm_per_species is None:
        return qm_per_species_from_config(run.config)
    return np.asarray(qm_per_species, dtype=np.float64)


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
