#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import picnix
from picnix import ohm


# -*- shape and argument validation -*-


def test_solve_ohm_1d_shape_validation():
    """solve_ohm_1d rejects invalid input shapes."""
    with pytest.raises(ValueError, match="L must be 1D"):
        ohm.solve_ohm_1d(np.ones((4, 4)), np.zeros((4, 3)), 1.0)
    with pytest.raises(ValueError, match=r"S must have shape \(Nx, 3\)"):
        ohm.solve_ohm_1d(np.ones(8), np.zeros((8, 2)), 1.0)
    with pytest.raises(ValueError, match=r"S must have shape \(Nx, 3\)"):
        ohm.solve_ohm_1d(np.ones(8), np.zeros((7, 3)), 1.0)


# -*- Fourier verification (1D) -*-


@pytest.mark.parametrize("N", [8, 16, 32, 64])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_solve_ohm_1d_fourier_yz(N, m):
    """1D periodic Fourier verification for the E_y and E_z components.

    ``E^alpha = E0 * cos(k_x * x)`` (alpha in {y, z}) is an eigenfunction
    of the 1D operator ``(Lambda - c^2 d^2/dx^2)`` with eigenvalue

    .. math::

        \\lambda(k_x) = \\Lambda + \\frac{4 c^2}{\\Delta^2} \\sin^2\\left(\\frac{k_x \\Delta}{2}\\right),
        \\quad k_x = \\frac{2 \\pi m}{N \\Delta}.
    """
    delta = 1.0
    c = 1.0
    E0 = 0.8
    L_val = 0.5

    x = np.arange(N) * delta
    kx = 2 * np.pi * m / (N * delta)
    L = np.full(N, L_val)
    lam = L_val + 4 * c * c / (delta * delta) * np.sin(np.pi * m / N) ** 2

    for alpha in (1, 2):
        E_true = np.zeros((N, 3))
        E_true[:, alpha] = E0 * np.cos(kx * x)
        S = np.zeros((N, 3))
        S[:, alpha] = lam * E_true[:, alpha]

        E_sol = ohm.solve_ohm_1d(L, S, delta)
        rel_err = np.max(np.abs(E_sol[:, alpha] - E_true[:, alpha])) / np.max(
            np.abs(E_true[:, alpha])
        )
        assert rel_err < 1e-10, f"alpha={alpha} rel_err={rel_err:.2e}"


@pytest.mark.parametrize("N", [8, 16, 32, 64])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_solve_ohm_1d_fourier_x(N, m):
    """1D periodic Fourier verification for the E_x component.

    All three components now use the same operator
    ``(Lambda - c^2 d^2/dx^2) E = S``.
    """
    delta = 1.0
    c = 1.0
    E0 = 0.8
    L_val = 0.5

    x = np.arange(N) * delta
    kx = 2 * np.pi * m / (N * delta)
    L = np.full(N, L_val)
    lam = L_val + 4 * c * c / (delta * delta) * np.sin(np.pi * m / N) ** 2

    E_true = np.zeros((N, 3))
    E_true[:, 0] = E0 * np.cos(kx * x)
    S = np.zeros((N, 3))
    S[:, 0] = lam * E_true[:, 0]

    E_sol = ohm.solve_ohm_1d(L, S, delta)
    rel_err = np.max(np.abs(E_sol[:, 0] - E_true[:, 0])) / np.max(np.abs(E_true[:, 0]))
    assert rel_err < 1e-10, f"alpha=0 rel_err={rel_err:.2e}"


# -*- precomputed base equivalence -*-


def test_solve_ohm_1d_deterministic_across_calls():
    """Two calls with the same arguments return the same E."""
    N = 24
    delta = 0.5
    c = 2.0

    L = 0.3 + 0.05 * np.cos(2 * np.pi * np.arange(N) / N)
    S = np.random.default_rng(0).normal(size=(N, 3))

    E_first = ohm.solve_ohm_1d(L, S, delta, c=c)
    E_second = ohm.solve_ohm_1d(L, S, delta, c=c)

    np.testing.assert_allclose(E_first, E_second, rtol=1e-12, atol=1e-12)


def test_solve_ohm_1d_c_scaling():
    """Operator scales with c^2; the manufactured-solution error is stable."""
    for c in [0.5, 1.0, 2.0]:
        N = 16
        delta = 1.0
        m = 2
        E0 = 0.5
        L_val = 0.4

        x = np.arange(N) * delta
        kx = 2 * np.pi * m / (N * delta)
        L = np.full(N, L_val)
        lam = L_val + 4 * c * c / (delta * delta) * np.sin(np.pi * m / N) ** 2

        E_true = np.zeros((N, 3))
        E_true[:, 1] = E0 * np.cos(kx * x)
        S = np.zeros((N, 3))
        S[:, 1] = lam * E_true[:, 1]

        E_sol = ohm.solve_ohm_1d(L, S, delta, c=c)
        rel_err = np.max(np.abs(E_sol[:, 1] - E_true[:, 1])) / np.max(
            np.abs(E_true[:, 1])
        )
        assert rel_err < 1e-10, f"c={c} rel_err={rel_err:.2e}"


# -*- general species support -*-


def test_transform_moments_general_species():
    """transform_moments correctly sums per-species contributions for Ns > 2."""
    Ns = 3
    shape = (10, Ns, 14)
    um = np.zeros(shape)
    um[..., 0] = 1.0
    um[..., 1] = 2.0
    um[..., 2] = 3.0
    um[..., 3] = 4.0
    um[..., 5] = 5.0
    um[..., 6] = 6.0
    um[..., 7] = 7.0
    um[..., 8] = 0.5
    um[..., 9] = 1.5
    um[..., 10] = 2.5
    um[..., 11] = 3.5
    um[..., 12] = 4.5
    um[..., 13] = 5.5

    qm = np.array([-1.0, 0.5, 0.1])
    qm2 = qm**2

    M = ohm.transform_moments(um, qm)

    expected_Lambda = (um[..., 0] * qm2).sum(axis=-1)
    expected_Gamma_x = (um[..., 1] * qm2).sum(axis=-1)
    expected_Pxx = (um[..., 5] * qm).sum(axis=-1)
    expected_Pzz = (um[..., 7] * qm).sum(axis=-1)

    np.testing.assert_allclose(M[..., 0], expected_Lambda)
    np.testing.assert_allclose(M[..., 1], expected_Gamma_x)
    np.testing.assert_allclose(M[..., 4], expected_Pxx)
    np.testing.assert_allclose(M[..., 6], expected_Pzz)


# -*- CG status reporting -*-


def test_solve_ohm_1d_return_info_status_ok():
    """On a manufactured problem, CG converges (status 0)."""
    N = 16
    delta = 1.0
    m = 1
    E0 = 0.5
    L_val = 0.5

    x = np.arange(N) * delta
    kx = 2 * np.pi * m / (N * delta)
    L = np.full(N, L_val)
    lam = L_val + 4 / (delta * delta) * np.sin(np.pi * m / N) ** 2

    E_true = np.zeros((N, 3))
    E_true[:, 2] = E0 * np.cos(kx * x)
    S = np.zeros((N, 3))
    S[:, 2] = lam * E_true[:, 2]

    E_sol, info = ohm.solve_ohm_1d(L, S, delta, return_info=True)
    assert all(s == 0 for s in info["status"]), f"statuses={info['status']}"
    rel_err = np.max(np.abs(E_sol - E_true)) / np.max(np.abs(E_true))
    assert rel_err < 1e-10


# -*- module export sanity -*-


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


# -*- calc_e_ohm_1d smoke test with a mock run -*-


class _MockRun:
    """Minimal picnix.Run stub for testing calc_e_ohm_1d / _2d."""

    def __init__(self, uf, um, config):
        self._uf = uf
        self._um = um
        self.config = config

    def read_at(self, prefix, step):
        return {"uf": self._uf, "um": self._um}


def test_calc_e_ohm_1d_dispatch():
    """calc_e_ohm_1d returns shape (Nx, 3) and finite values for a 1D run.

    Builds a 1D picnix-shaped snapshot (nc=1, ny=1, nx=N) and verifies
    the result has shape (Nx, 3) and finite values.
    """
    Nx = 32
    Ns = 2
    delta = 0.1

    um = np.zeros((1, 1, Nx, Ns, 14))
    um[..., 0, 0] = 1.0
    um[..., 1, 0] = 1.0
    um[..., 0, 1] = 0.1
    um[..., 1, 1] = 0.0

    uf = np.zeros((1, 1, Nx, 6))
    uf[..., 3] = 0.5
    uf[..., 4] = 0.0
    uf[..., 5] = 0.0

    config = {
        "parameter": {
            "Ns": Ns,
            "Ny": 1,
            "Nx": Nx,
            "delh": delta,
            "mime": 25.0,
            "wp": 1.0,
            "nppc": 100,
        }
    }
    run = _MockRun(uf, um, config)
    E_ohm = picnix.calc_e_ohm_1d(run, 0, c=1.0)
    assert E_ohm.shape == (Nx, 3)
    assert np.all(np.isfinite(E_ohm))


# -*- _resolve_qm priority order -*-


def test_resolve_qm_explicit_arg_wins():
    """Explicit qm_per_species argument overrides everything else."""

    class _Run:
        qm = [-2.0, -2.0, 0.5]
        config = {"parameter": {"Ns": 3}}

    explicit = [-7.0, -7.0, 7.0]
    out = ohm._resolve_qm(_Run(), explicit)
    np.testing.assert_array_equal(out, explicit)


def test_resolve_qm_profile_field_used_when_no_explicit():
    """run.qm is used when qm_per_species is not given."""

    class _Run:
        qm = [-1.0, -1.0, 0.01]
        config = {"parameter": {"Ns": 3, "mime": 100.0, "wp": 1.0, "nppc": 100}}

    out = ohm._resolve_qm(_Run(), None)
    np.testing.assert_array_equal(out, [-1.0, -1.0, 0.01])


def test_resolve_qm_falls_back_to_config_when_no_profile_qm():
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


def test_resolve_qm_handles_missing_qm_attribute():
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
