#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import picnix
from picnix import ohm


# -*- shape and argument validation -*-


def test_solve_ohm_2d_shape_validation():
    """solve_ohm_2d rejects invalid input shapes."""
    with pytest.raises(ValueError, match="L must be 2D"):
        ohm.solve_ohm_2d(np.ones(8), np.zeros((8, 8, 3)), 1.0)
    with pytest.raises(ValueError, match=r"S must have shape"):
        ohm.solve_ohm_2d(np.ones((8, 8)), np.zeros((8, 8, 2)), 1.0)
    with pytest.raises(ValueError, match=r"matching L"):
        ohm.solve_ohm_2d(np.ones((8, 8)), np.zeros((7, 8, 3)), 1.0)


def test_build_source_term_2d_uses_ny_nx_axes_for_pressure_gradients():
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
    np.testing.assert_allclose(ohm._build_source_term_2d(B, M, delta), expected)


# -*- matrix symmetry and base shape validation -*-


@pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
def test_assembled_matrices_are_symmetric(Nx, Ny):
    """A1 and A2 should be symmetric for the 2D periodic system."""
    c2_dx2 = c2_dx4 = 1.0

    x = np.arange(Nx)
    y = np.arange(Ny)
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    L = 1.0 + 0.1 * np.cos(2 * np.pi * xx / Nx) + 0.05 * np.sin(2 * np.pi * yy / Ny)

    A_1, A_2 = ohm._build_ohm_matrices_2d(Nx, Ny, L, c2_dx2, c2_dx4)

    asym1 = (A_1 - A_1.T).tocoo()
    asym2 = (A_2 - A_2.T).tocoo()
    if asym1.nnz:
        assert np.max(np.abs(asym1.data)) < 1e-14
    if asym2.nnz:
        assert np.max(np.abs(asym2.data)) < 1e-14


# -*- Fourier verification (Ez) -*-


class TestEzFourier:
    """2D Ez Fourier verification with periodic boundary conditions.

    Manufactures ``E^z = E0 cos(k_x x + k_y y)`` with
    ``k_x = 2 pi m_x / (N_x Delta)``, ``k_y = 2 pi m_y / (N_y Delta)`` and
    the source ``S^z = A_{zz} E^z`` where

    .. math::

        A_{zz} = \\Lambda + 4 \\frac{c^2}{\\Delta^2}
        \\left[\\sin^2\\left(\\frac{k_x \\Delta}{2}\\right) +
              \\sin^2\\left(\\frac{k_y \\Delta}{2}\\right)\\right].
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
        c2 = c * c

        Ez0 = 0.8
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 2] = Ez0 * np.cos(kx * xx + ky * yy)

        eigenvalue = L_val + 4 * c2 / (delta * delta) * (
            np.sin(kx * delta / 2) ** 2 + np.sin(ky * delta / 2) ** 2
        )

        S = np.zeros((Ny, Nx, 3))
        S[..., 2] = eigenvalue * E_true[..., 2]

        E_solved = ohm.solve_ohm_2d(L_val * np.ones((Ny, Nx)), S, delta)
        rel_err = np.max(np.abs(E_solved[..., 2] - E_true[..., 2])) / np.max(
            np.abs(E_true[..., 2])
        )
        assert rel_err < 1e-10, f"Ez rel_err={rel_err:.2e}"


# -*- Fourier verification (Ex/Ey coupled) -*-


class TestExEyFourier:
    """2D Ex/Ey coupled Fourier verification with periodic boundary conditions.

    Manufactures ``E^x = E^x_0 cos(k_x x + k_y y)`` and
    ``E^y = E^y_0 cos(k_x x + k_y y)`` with the source

    .. math::

        \\begin{pmatrix} S^x \\\\ S^y \\end{pmatrix} =
        \\begin{pmatrix} A_{xx} & A_{xy} \\\\
                          A_{xy} & A_{yy} \\end{pmatrix}
        \\begin{pmatrix} E^x \\\\ E^y \\end{pmatrix},

    with ``A_{xx}``, ``A_{yy}``, ``A_{xy}`` as in the 2D doc.
    """

    @pytest.mark.parametrize("Nx,Ny", [(8, 8), (12, 16), (16, 12)])
    @pytest.mark.parametrize("mx,my", [(1, 1), (2, 3), (3, 2)])
    def test_ex_ey_coupled_fourier_periodic(self, Nx, Ny, mx, my):
        delta = 1.0
        Lx = Nx * delta
        Ly = Ny * delta

        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly
        L_val = 0.5
        c = 1.0
        c2 = c * c
        c2_dx2 = c2 / (delta * delta)

        Ex0, Ey0 = 1.0, 0.5
        x = np.arange(Nx) * delta
        y = np.arange(Ny) * delta
        xx = np.broadcast_to(x, (Ny, Nx))
        yy = np.broadcast_to(y[:, None], (Ny, Nx))

        E_true = np.zeros((Ny, Nx, 3))
        E_true[..., 0] = Ex0 * np.cos(kx * xx + ky * yy)
        E_true[..., 1] = Ey0 * np.cos(kx * xx + ky * yy)

        A_xx = L_val + 4 * c2_dx2 * np.sin(ky * delta / 2) ** 2
        A_yy = L_val + 4 * c2_dx2 * np.sin(kx * delta / 2) ** 2
        A_xy = -c2_dx2 * np.sin(kx * delta) * np.sin(ky * delta)

        S = np.zeros((Ny, Nx, 3))
        S[..., 0] = A_xx * E_true[..., 0] + A_xy * E_true[..., 1]
        S[..., 1] = A_xy * E_true[..., 0] + A_yy * E_true[..., 1]

        E_solved = ohm.solve_ohm_2d(L_val * np.ones((Ny, Nx)), S, delta)
        rel_err_ex = np.max(np.abs(E_solved[..., 0] - E_true[..., 0])) / np.max(
            np.abs(E_true[..., 0])
        )
        rel_err_ey = np.max(np.abs(E_solved[..., 1] - E_true[..., 1])) / np.max(
            np.abs(E_true[..., 1])
        )
        assert rel_err_ex < 1e-10, f"Ex rel_err={rel_err_ex:.2e}"
        assert rel_err_ey < 1e-10, f"Ey rel_err={rel_err_ey:.2e}"


# -*- precomputed base equivalence -*-


def test_solve_ohm_2d_default_vs_precomputed_base():
    """Default path matches the path with precomputed base matrices."""
    Nx, Ny = 10, 8
    delta = 1.0
    c = 1.0

    x = np.arange(Nx)
    y = np.arange(Ny)
    xx = np.broadcast_to(x, (Ny, Nx))
    yy = np.broadcast_to(y[:, None], (Ny, Nx))
    L = 0.5 + 0.1 * np.cos(2 * np.pi * xx / Nx) + 0.05 * np.sin(2 * np.pi * yy / Ny)

    rng = np.random.default_rng(123)
    S = rng.normal(size=(Ny, Nx, 3))

    base1, base2 = ohm.build_ohm_bases_from_grid(Nx, Ny, c, delta)

    E_default = ohm.solve_ohm_2d(L, S, delta, c=c)
    E_precomp = ohm.solve_ohm_2d(L, S, delta, c=c, base1=base1, base2=base2)

    np.testing.assert_allclose(E_default, E_precomp, rtol=1e-11, atol=1e-12)


def test_solve_ohm_2d_rejects_bad_precomputed_base_shape():
    """solve_ohm_2d validates supplied base-matrix shapes."""
    Nx, Ny = 10, 8
    delta = 1.0
    L = np.ones((Ny, Nx))
    S = np.zeros((Ny, Nx, 3))

    base1, base2 = ohm.build_ohm_bases_from_grid(Nx, Ny, c=1.0, delta=delta)

    with pytest.raises(ValueError, match="base shape"):
        ohm.solve_ohm_2d(L, S, delta, base1=base1[:-1, :-1], base2=base2)

    with pytest.raises(ValueError, match="base shape"):
        ohm.solve_ohm_2d(L, S, delta, base1=base1, base2=base2[:-1, :-1])


# -*- general species support (3-species synthetic) -*-


def test_transform_moments_three_species_consistent_with_2d_solve():
    """End-to-end: build um for 3 species, transform, solve, check Fourier.

    Synthetic 3-species setup with qm = [-1, +0.04, +0.04] (one electron,
    two ion species). Construct um such that the transformed Lambda is
    spatially constant, then drive a manufactured Ez solution.
    """
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
    assert info["status_1"] == 0
    assert info["status_2"] == 0
    rel_err = np.max(np.abs(E_sol - E_true)) / np.max(np.abs(E_true))
    assert rel_err < 1e-10


# -*- calc_e_ohm_2d smoke test with a mock run -*-


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

    # config is intentionally ambiguous (no [[parameter.particle]]); without
    # the profile qm field, calc_e_ohm_2d would raise.
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
