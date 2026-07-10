# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import picnix
from picnix import ohm


# -*- helpers -*-


def _periodic_gradient(values, delta):
    dx = (np.roll(values, -1, axis=-1) - np.roll(values, 1, axis=-1)) / (2.0 * delta)
    dy = (np.roll(values, -1, axis=-2) - np.roll(values, 1, axis=-2)) / (2.0 * delta)
    return dx, dy


def _manufactured_case(nx=12, ny=10, *, variable_lambda=True):
    delta = 0.4
    c = 1.7
    x = np.arange(nx) * delta
    y = np.arange(ny) * delta
    xx = np.broadcast_to(x, (ny, nx))
    yy = np.broadcast_to(y[:, None], (ny, nx))

    if variable_lambda:
        L = 1.2 + 0.15 * np.cos(2.0 * np.pi * xx / (nx * delta))
        L += 0.1 * np.sin(2.0 * np.pi * yy / (ny * delta))
    else:
        L = np.full((ny, nx), 1.2)

    E = np.empty((ny, nx, 3))
    E[..., 0] = np.cos(2.0 * np.pi * xx / (nx * delta))
    E[..., 1] = 0.7 * np.sin(4.0 * np.pi * yy / (ny * delta))
    E[..., 2] = 0.4 * np.cos(
        2.0 * np.pi * xx / (nx * delta) + 2.0 * np.pi * yy / (ny * delta)
    )
    rho = 0.3 * np.sin(2.0 * np.pi * xx / (nx * delta))
    rho += 0.2 * np.cos(2.0 * np.pi * yy / (ny * delta))

    A = ohm._assemble_ohm_matrix_2d(L, delta, c=c)
    reduced_rhs = np.empty_like(E)
    for component in range(3):
        reduced_rhs[..., component] = (A @ E[..., component].ravel()).reshape(L.shape)
    d_rho_dx, d_rho_dy = _periodic_gradient(rho, delta)
    S = reduced_rhs.copy()
    S[..., 0] += c * c * d_rho_dx
    S[..., 1] += c * c * d_rho_dy
    return L, S, rho, E, delta, c, A


# -*- public API exports -*-


def test_ohm_public_api_exported():
    """All public names are accessible from the picnix top-level namespace."""
    for name in [
        "solve_ohm_1d",
        "solve_ohm_2d",
        "calc_e_ohm_1d",
        "calc_e_ohm_2d",
        "transform_moments",
        "qm_per_species_from_config",
    ]:
        assert hasattr(picnix, name), f"picnix.{name} not exported"


# -*- shape and argument validation -*-


def test_solve_ohm_2d_shape_validation():
    """solve_ohm_2d rejects invalid input shapes."""
    with pytest.raises(ValueError, match="L must be 2D"):
        ohm.solve_ohm_2d(np.ones(8), np.zeros((8, 8, 3)), 1.0)
    with pytest.raises(ValueError, match=r"S must have shape"):
        ohm.solve_ohm_2d(np.ones((8, 8)), np.zeros((8, 8, 2)), 1.0)
    with pytest.raises(ValueError, match=r"matching L"):
        ohm.solve_ohm_2d(np.ones((8, 8)), np.zeros((7, 8, 3)), 1.0)


def test_solve_ohm_2d_lambda_and_value_validation():
    L = np.ones((6, 8))
    S = np.zeros((6, 8, 3))

    for bad_L in [np.full_like(L, np.nan), np.zeros_like(L), -np.ones_like(L)]:
        with pytest.raises(ValueError, match="L must"):
            ohm.solve_ohm_2d(bad_L, S, 1.0)

    bad_S = S.copy()
    bad_S[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="S must contain only finite"):
        ohm.solve_ohm_2d(L, bad_S, 1.0)

    with pytest.raises(ValueError, match="min_lambda"):
        ohm.solve_ohm_2d(L, S, 1.0, min_lambda=1.0)


def test_solve_ohm_2d_preconditioner_validation():
    L = np.ones((6, 8))
    S = np.zeros((6, 8, 3))

    with pytest.raises(ValueError, match="unknown preconditioner"):
        ohm.solve_ohm_2d(L, S, 1.0, preconditioner="invalid")
    with pytest.raises(TypeError, match="must be"):
        ohm.solve_ohm_2d(L, S, 1.0, preconditioner=np.eye(L.size))

    matrix = ohm._assemble_ohm_matrix_2d(L, 1.0)
    matrix.data[0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        ohm.solve_ohm_2d(L, S, 1.0, matrix=matrix)


# -*- source term construction -*-


def test_build_source_2d_uses_ny_nx_axes_for_pressure_gradients():
    """Pressure divergence uses x=axis -1 and y=axis -2 for (Ny, Nx) arrays."""
    nx, ny = 7, 5
    delta = 0.25
    x = np.arange(nx)
    y = np.arange(ny)
    xx = np.broadcast_to(x, (ny, nx))
    yy = np.broadcast_to(y[:, None], (ny, nx))
    B = np.zeros((ny, nx, 3))
    M = np.zeros((ny, nx, 10))

    M[..., 4] = xx**2 + 10 * yy
    M[..., 5] = 3 * yy**2 - xx
    M[..., 7] = 2 * xx + 5 * yy**2
    M[..., 8] = yy**2 + 7 * xx
    M[..., 9] = 4 * xx**2 - 3 * yy

    def ddx(values):
        return (np.roll(values, -1, axis=-1) - np.roll(values, 1, axis=-1)) / (
            2.0 * delta
        )

    def ddy(values):
        return (np.roll(values, -1, axis=-2) - np.roll(values, 1, axis=-2)) / (
            2.0 * delta
        )

    expected = np.stack(
        [
            ddx(M[..., 4]) + ddy(M[..., 7]),
            ddx(M[..., 7]) + ddy(M[..., 5]),
            ddx(M[..., 9]) + ddy(M[..., 8]),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(ohm._build_source_2d(B, M, delta, c=1.0), expected)


# -*- reduced RHS (grad-rho correction) -*-


def test_reduced_rhs_2d_charge_gradient_sign_and_components():
    nx, ny = 9, 7
    delta = 0.25
    c = 2.0
    x = np.arange(nx)
    y = np.arange(ny)
    rho = np.sin(2.0 * np.pi * x / nx)[None, :]
    rho = np.broadcast_to(rho, (ny, nx)).copy()
    rho += 0.4 * np.cos(2.0 * np.pi * y[:, None] / ny)
    S = np.full((ny, nx, 3), 0.3)

    d_rho_dx, d_rho_dy = _periodic_gradient(rho, delta)
    rhs = ohm._build_reduced_rhs_2d(S, rho, delta, c)

    np.testing.assert_allclose(rhs[..., 0], S[..., 0] - c * c * d_rho_dx)
    np.testing.assert_allclose(rhs[..., 1], S[..., 1] - c * c * d_rho_dy)
    np.testing.assert_array_equal(rhs[..., 2], S[..., 2])

    constant_rhs = ohm._build_reduced_rhs_2d(S, np.ones_like(rho), delta, c)
    np.testing.assert_array_equal(constant_rhs, S)


# -*- Fourier verification (Ez) -*-


class TestEzFourier:
    """2D Ez Fourier verification against the scalar reduced operator.

    Manufactures ``E^z = E0 cos(k_x x + k_y y)`` with
    ``S^z = (Lambda - c^2 Laplacian) E^z``, i.e. the reduced
    source term.
    """

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ez_fourier_periodic(self, Nx, Ny, mx, my):
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly
        L_val = 0.5
        c = 1.0

        Ez0 = 0.8
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 2] = Ez0 * np.cos(kx * xx + ky * yy)

        eigenvalue = L_val + 4 * c * c / (delta * delta) * (
            np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
        )

        S = np.zeros((Ny, Nx, 3))
        S[..., 2] = eigenvalue * E_true[..., 2]

        E_solved = ohm.solve_ohm_2d(L_val * np.ones((Ny, Nx)), S, delta)
        rel_err = np.max(np.abs(E_solved[..., 2] - E_true[..., 2])) / np.max(
            np.abs(E_true[..., 2])
        )
        assert rel_err < 1e-10, f"Ez rel_err={rel_err:.2e}"


# -*- Fourier verification (Ex/Ey) -*-


class TestExEyFourier:
    """2D Ex / Ey Fourier verification against the scalar reduced operator.

    Both Ex and Ey now use the same scalar operator, so the test is
    identical for each component.
    """

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ex_ey_fourier_periodic(self, Nx, Ny, mx, my):
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly
        L_val = 0.5
        c = 1.0

        Ex0, Ey0 = 1.0, 0.5
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        eigenvalue = L_val + 4 * c * c / (delta * delta) * (
            np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
        )

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 0] = Ex0 * np.cos(kx * xx + ky * yy)
        E_true[..., 1] = Ey0 * np.cos(kx * xx + ky * yy)

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = eigenvalue * E_true[..., 0]
        S[..., 1] = eigenvalue * E_true[..., 1]

        E_solved = ohm.solve_ohm_2d(L_val * np.ones((Ny, Nx)), S, delta)
        rel_err_ex = np.max(np.abs(E_solved[..., 0] - E_true[..., 0])) / np.max(
            np.abs(E_true[..., 0])
        )
        rel_err_ey = np.max(np.abs(E_solved[..., 1] - E_true[..., 1])) / np.max(
            np.abs(E_true[..., 1])
        )
        assert rel_err_ex < 1e-10, f"Ex rel_err={rel_err_ex:.2e}"
        assert rel_err_ey < 1e-10, f"Ey rel_err={rel_err_ey:.2e}"


# -*- manufactured solution tests -*-


def test_manufactured_solution_fft_preconditioning():
    L, S, rho, expected, delta, c, _ = _manufactured_case(variable_lambda=True)

    S_reduced = ohm._build_reduced_rhs_2d(S, rho, delta, c)
    solved, info = ohm.solve_ohm_2d(
        L, S_reduced, delta, c=c, preconditioner="fft", rtol=1.0e-10, return_info=True
    )

    np.testing.assert_allclose(solved, expected, rtol=2.0e-9, atol=2.0e-10)
    assert info["status"] == (0, 0, 0)
    assert max(info["relative_residual"]) < 2.0e-9
    assert info["preconditioner"] == "fft"


def test_manufactured_solution_unpreconditioned_cg():
    L, S, rho, expected, delta, c, _ = _manufactured_case(variable_lambda=True)

    S_reduced = ohm._build_reduced_rhs_2d(S, rho, delta, c)
    solved, info = ohm.solve_ohm_2d(
        L,
        S_reduced,
        delta,
        c=c,
        preconditioner=None,
        rtol=1.0e-10,
        return_info=True,
    )

    np.testing.assert_allclose(solved, expected, rtol=2.0e-9, atol=2.0e-10)
    assert info["status"] == (0, 0, 0)
    assert max(info["relative_residual"]) < 2.0e-9
    assert info["preconditioner"] == "none"


def test_defaults_to_fft_preconditioning():
    L, S, rho, expected, delta, c, _ = _manufactured_case(variable_lambda=True)

    S_reduced = ohm._build_reduced_rhs_2d(S, rho, delta, c)
    solved, info = ohm.solve_ohm_2d(
        L, S_reduced, delta, c=c, rtol=1.0e-10, return_info=True
    )

    np.testing.assert_allclose(solved, expected, rtol=2.0e-9, atol=2.0e-10)
    assert info["preconditioner"] == "fft"


def test_cg_uses_relative_tolerance_for_small_rhs():
    L = np.ones((6, 8))
    S = np.full((*L.shape, 3), 1.0e-14)

    solved, info = ohm.solve_ohm_2d(L, S, 1.0, rtol=1.0e-10, return_info=True)

    np.testing.assert_allclose(solved, S, rtol=1.0e-10, atol=0.0)
    assert info["status"] == (0, 0, 0)
    assert max(info["relative_residual"]) < 1.0e-10


# -*- base / matrix reuse -*-


def test_reuses_base_and_matrix():
    L, S, rho, expected, delta, c, A = _manufactured_case(variable_lambda=True)
    ny, nx = L.shape
    laplacian = ohm._build_laplacian_2d(nx, ny, delta, c=c)
    S_reduced = ohm._build_reduced_rhs_2d(S, rho, delta, c)

    solved_base, _ = ohm.solve_ohm_2d(
        L, S_reduced, delta, c=c, base=laplacian, rtol=1.0e-10, return_info=True
    )
    solved_matrix, _ = ohm.solve_ohm_2d(
        L,
        S_reduced,
        delta,
        c=c,
        matrix=A,
        validate_matrix=False,
        rtol=1.0e-10,
        return_info=True,
    )

    np.testing.assert_allclose(solved_base, expected, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(solved_matrix, expected, rtol=2.0e-9, atol=2.0e-10)

    with pytest.raises(ValueError, match="matrix does not match"):
        ohm.solve_ohm_2d(2.0 * L, S_reduced, delta, c=c, matrix=A, rtol=1.0e-10)
    with pytest.raises(ValueError, match="base does not match"):
        ohm.solve_ohm_2d(L, S_reduced, delta, c=2.0 * c, base=laplacian, rtol=1.0e-10)

    transposed_S = np.transpose(S_reduced, (1, 0, 2))
    with pytest.raises(ValueError, match="matrix does not match"):
        ohm.solve_ohm_2d(L.T, transposed_S, delta, c=c, matrix=A, rtol=1.0e-10)

    corrupted = A.tolil(copy=True)
    row = nx + 1
    corrupted[row, row + 1] += 0.25
    corrupted[row, row + 3] -= 0.25
    with pytest.raises(ValueError, match="matrix does not match"):
        ohm.solve_ohm_2d(
            L, S_reduced, delta, c=c, matrix=corrupted.tocsr(), rtol=1.0e-10
        )


def test_reuses_reaction_only_matrix_when_c_is_zero():
    L = np.full((6, 8), 2.0)
    rng = np.random.default_rng(4)
    S = rng.normal(size=(*L.shape, 3))
    laplacian = ohm._build_laplacian_2d(8, 6, 1.0, c=0.0)
    matrix = ohm._assemble_ohm_matrix_2d(L, 1.0, c=0.0)

    solved_matrix = ohm.solve_ohm_2d(L, S, 1.0, c=0.0, matrix=matrix)
    solved_base = ohm.solve_ohm_2d(L, S, 1.0, c=0.0, base=laplacian)

    expected = S / L[..., None]
    np.testing.assert_allclose(solved_matrix, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(solved_base, expected, rtol=1.0e-13, atol=1.0e-13)


# -*- CG convergence on manufactured case -*-


def test_solve_ohm_2d_return_info_status_ok():
    """On a manufactured problem, CG converges (status 0)."""
    Nx, Ny = 10, 10
    delta = 1.0
    c = 1.0
    mx, my = 1, 1
    L_val = 0.5

    x = np.arange(Nx) * delta
    y = np.arange(Ny) * delta
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    kx = 2 * np.pi * mx / (Nx * delta)
    ky = 2 * np.pi * my / (Ny * delta)

    E_true = np.zeros((Ny, Nx, 3))
    E_true[..., 2] = 0.7 * np.cos(kx * xx + ky * yy)
    eig = L_val + 4 * c * c / (delta * delta) * (
        np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
    )
    S = np.zeros((Ny, Nx, 3))
    S[..., 2] = eig * E_true[..., 2]

    E_sol, info = ohm.solve_ohm_2d(
        L_val * np.ones((Ny, Nx)), S, delta, return_info=True
    )
    assert info["status"] == (0, 0, 0)
    rel_err = np.max(np.abs(E_sol - E_true)) / np.max(np.abs(E_true))
    assert rel_err < 1e-10


# -*- calc_e_ohm_2d smoke test -*-


class _MockRun:
    """Minimal picnix.Run stub for testing calc_e_ohm_1d / _2d."""

    def __init__(self, uf, um, config):
        self._uf = uf
        self._um = um
        self.config = config

    def read_at(self, prefix, step):
        return {"uf": self._uf, "um": self._um}


def test_calc_e_ohm_2d_dispatch():
    """calc_e_ohm_2d returns shape (Ny, Nx, 3) and finite values for Ny>1."""
    Nx, Ny = 12, 8
    Ns = 2
    delta = 0.2

    um = np.zeros((1, Ny, Nx, Ns, 14))
    um[..., 0, 0] = 1.0
    um[..., 1, 0] = 1.0
    um[..., 0, 1] = 0.1
    um[..., 1, 1] = 0.0

    uf = np.zeros((1, Ny, Nx, 6))
    uf[..., 3] = 0.5
    uf[..., 4] = 0.0
    uf[..., 5] = 0.0

    config = {
        "parameter": {
            "Ns": Ns,
            "Ny": Ny,
            "Nx": Nx,
            "delh": delta,
            "mime": 25.0,
            "wp": 1.0,
            "nppc": 100,
        }
    }
    run = _MockRun(uf, um, config)
    E_ohm = picnix.calc_e_ohm_2d(run, 0, c=1.0)
    assert E_ohm.shape == (Ny, Nx, 3)
    assert np.all(np.isfinite(E_ohm))


def test_calc_e_ohm_2d_explicit_qm_required_for_3_species():
    """For a 3-species run with no [[parameter.particle]], explicit qm is required."""
    Nx, Ny = 8, 8
    Ns = 3
    delta = 0.5

    um = np.zeros((1, Ny, Nx, Ns, 14))
    um[..., 0, 0] = 1.0
    um[..., 1, 0] = 1.0
    um[..., 2, 0] = 1.0

    uf = np.zeros((1, Ny, Nx, 6))
    uf[..., 3] = 0.5

    config = {
        "parameter": {
            "Ns": Ns,
            "Ny": Ny,
            "Nx": Nx,
            "delh": delta,
        }
    }
    run = _MockRun(uf, um, config)

    with pytest.raises(ValueError, match="Cannot infer qm"):
        picnix.calc_e_ohm_2d(run, 0, c=1.0)

    qm = np.array([-1.0, 0.04, 0.04])
    E_ohm = picnix.calc_e_ohm_2d(run, 0, c=1.0, qm_per_species=qm)
    assert E_ohm.shape == (Ny, Nx, 3)
    assert np.all(np.isfinite(E_ohm))


# -*- _resolve_qm priority order -*-


def test_resolve_qm_2d_explicit_arg_wins():
    """Explicit qm_per_species argument overrides everything else."""

    class _Run:
        qm = [-2.0, -2.0, 0.5]
        config = {"parameter": {"Ns": 3}}

    explicit = [-7.0, -7.0, 7.0]
    out = ohm._resolve_qm(_Run(), explicit)
    np.testing.assert_array_equal(out, explicit)


def test_resolve_qm_2d_profile_field_used_when_no_explicit():
    """run.qm is used when qm_per_species is not given."""

    class _Run:
        qm = [-1.0, -1.0, 0.01]
        config = {"parameter": {"Ns": 3, "mime": 100.0, "wp": 1.0, "nppc": 100}}

    out = ohm._resolve_qm(_Run(), None)
    np.testing.assert_array_equal(out, [-1.0, -1.0, 0.01])


def test_resolve_qm_2d_falls_back_to_config_when_no_profile_qm():
    """Old profiles (no qm) fall back to qm_per_species_from_config."""

    class _Run:
        qm = None
        config = {
            "parameter": {
                "Ns": 2,
                "mime": 25.0,
                "wp": 1.0,
                "nppc": 100,
            }
        }

    out = ohm._resolve_qm(_Run(), None)
    np.testing.assert_allclose(out, [-1.0, 1.0 / 25.0])


def test_resolve_qm_2d_handles_missing_qm_attribute():
    """If run has no qm attribute at all, fall back to config inference."""

    class _Run:
        config = {
            "parameter": {
                "Ns": 2,
                "mime": 25.0,
                "wp": 1.0,
                "nppc": 100,
            }
        }

    out = ohm._resolve_qm(_Run(), None)
    np.testing.assert_allclose(out, [-1.0, 1.0 / 25.0])


def test_calc_e_ohm_2d_uses_profile_qm_for_3_species():
    """A 3-species run with run.qm set works without an explicit override."""

    Nx, Ny = 8, 8
    Ns = 3
    delta = 0.5

    um = np.zeros((1, Ny, Nx, Ns, 14))
    um[..., 0, 0] = 1.0
    um[..., 1, 0] = 1.0
    um[..., 2, 0] = 1.0

    uf = np.zeros((1, Ny, Nx, 6))
    uf[..., 3] = 0.5

    config = {
        "parameter": {
            "Ns": Ns,
            "Ny": Ny,
            "Nx": Nx,
            "delh": delta,
        }
    }
    run = _MockRun(uf, um, config)
    run.qm = [-1.0, -1.0, 0.01]

    E_ohm = picnix.calc_e_ohm_2d(run, 0, c=1.0)
    assert E_ohm.shape == (Ny, Nx, 3)
    assert np.all(np.isfinite(E_ohm))


# -*- 3-species transform + solve -*-


def test_transform_moments_three_species_consistent_with_2d_solve():
    """End-to-end: build um for 3 species, transform, solve, check Fourier."""
    Ns = 3
    Nx, Ny = 8, 8
    delta = 1.0
    c = 1.0
    m = 1

    qm = np.array([-1.0, 0.04, 0.04])

    um = np.zeros((Ny, Nx, Ns, 14))
    um[..., 0, 0] = 1.0
    um[..., 1, 0] = 1.0 / qm[1] ** 2
    um[..., 2, 0] = 1.0 / qm[2] ** 2

    M = ohm.transform_moments(um, qm)
    L = M[..., 0]
    assert np.allclose(L, 3.0)

    x = np.arange(Nx) * delta
    y = np.arange(Ny) * delta
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    kx = 2 * np.pi * m / (Nx * delta)
    ky = 2 * np.pi * m / (Ny * delta)

    Ez0 = 0.6
    E_true = np.zeros((Ny, Nx, 3))
    E_true[..., 2] = Ez0 * np.cos(kx * xx + ky * yy)
    eig = 3.0 + 4 * c * c / (delta * delta) * (
        np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
    )
    S = np.zeros((Ny, Nx, 3))
    S[..., 2] = eig * E_true[..., 2]

    E_sol = ohm.solve_ohm_2d(L, S, delta)
    rel_err = np.max(np.abs(E_sol[..., 2] - E_true[..., 2])) / np.max(
        np.abs(E_true[..., 2])
    )
    assert rel_err < 1e-10, f"3-species Ez rel_err={rel_err:.2e}"
