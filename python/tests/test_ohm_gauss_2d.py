import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

import picnix
from picnix import ohm


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

    A = ohm.assemble_ohm_gauss_matrix_2d(L, delta, c=c)
    reduced_rhs = np.empty_like(E)
    for component in range(3):
        reduced_rhs[..., component] = (A @ E[..., component].ravel()).reshape(L.shape)
    d_rho_dx, d_rho_dy = _periodic_gradient(rho, delta)
    S = reduced_rhs.copy()
    S[..., 0] += c * c * d_rho_dx
    S[..., 1] += c * c * d_rho_dy
    return L, S, rho, E, delta, c, A


def test_gauss_reduced_public_api_exported():
    for name in [
        "solve_ohm_2d_gauss_reduced",
        "build_ohm_gauss_base_2d",
        "assemble_ohm_gauss_matrix_2d",
        "build_ohm_gauss_preconditioner_2d",
    ]:
        assert hasattr(picnix, name)


def test_gauss_reduced_shape_and_value_validation():
    L = np.ones((6, 8))
    S = np.zeros((6, 8, 3))
    rho = np.zeros((6, 8))

    with pytest.raises(ValueError, match="L must be 2D"):
        ohm.solve_ohm_2d_gauss_reduced(np.ones(8), S, rho, 1.0)
    with pytest.raises(ValueError, match="S must have shape"):
        ohm.solve_ohm_2d_gauss_reduced(L, S[..., :2], rho, 1.0)
    with pytest.raises(ValueError, match="rho must have shape"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho[:-1], 1.0)

    for bad_L in [np.full_like(L, np.nan), np.zeros_like(L), -np.ones_like(L)]:
        with pytest.raises(ValueError, match="L must"):
            ohm.solve_ohm_2d_gauss_reduced(bad_L, S, rho, 1.0)

    bad_S = S.copy()
    bad_S[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="S must contain only finite"):
        ohm.solve_ohm_2d_gauss_reduced(L, bad_S, rho, 1.0)

    bad_rho = rho.copy()
    bad_rho[0, 0] = np.nan
    with pytest.raises(ValueError, match="rho must contain only finite"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, bad_rho, 1.0)

    with pytest.raises(ValueError, match="min_lambda"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, 1.0, min_lambda=1.0)
    with pytest.raises(ValueError, match="at least 3"):
        ohm.build_ohm_gauss_base_2d(2, 6, 1.0)
    with pytest.raises(ValueError, match="at least 3"):
        ohm.solve_ohm_2d_gauss_reduced(
            np.ones((3, 2)), np.zeros((3, 2, 3)), np.zeros((3, 2)), 1.0, solver="fft"
        )


def test_gauss_reduced_solver_and_preconditioner_validation():
    L = np.ones((6, 8))
    S = np.zeros((6, 8, 3))
    rho = np.zeros((6, 8))

    with pytest.raises(ValueError, match="unknown solver"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, 1.0, solver="invalid")
    with pytest.raises(ValueError, match="unknown preconditioner"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, 1.0, preconditioner="invalid")
    with pytest.raises(TypeError, match="LinearOperator"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, 1.0, preconditioner=np.eye(L.size))
    with pytest.raises(ValueError, match="does not accept"):
        ohm.solve_ohm_2d_gauss_reduced(
            L, S, rho, 1.0, solver="fft", preconditioner="amg"
        )

    matrix = ohm.assemble_ohm_gauss_matrix_2d(L, 1.0)
    base = ohm.build_ohm_gauss_base_2d(8, 6, 1.0)
    with pytest.raises(ValueError, match="does not use base or matrix"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, 1.0, solver="fft", base=base)
    with pytest.raises(ValueError, match="does not use a sparse matrix"):
        ohm.build_ohm_gauss_preconditioner_2d(L, 1.0, kind="fft", matrix=matrix)

    matrix.data[0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, 1.0, matrix=matrix)


def test_gauss_reduced_charge_gradient_sign_units_and_components():
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
    rhs = ohm._build_gauss_reduced_rhs_2d(S, rho, delta, c)

    np.testing.assert_allclose(rhs[..., 0], S[..., 0] - c * c * d_rho_dx)
    np.testing.assert_allclose(rhs[..., 1], S[..., 1] - c * c * d_rho_dy)
    np.testing.assert_array_equal(rhs[..., 2], S[..., 2])

    constant_rhs = ohm._build_gauss_reduced_rhs_2d(S, np.ones_like(rho), delta, c)
    np.testing.assert_array_equal(constant_rhs, S)


@pytest.mark.parametrize("solver", ["cg", "fft", "splu"])
def test_gauss_reduced_constant_lambda_manufactured_solution(solver):
    L, S, rho, expected, delta, c, _ = _manufactured_case(variable_lambda=False)

    solved, info = ohm.solve_ohm_2d_gauss_reduced(
        L, S, rho, delta, c=c, solver=solver, rtol=1.0e-11, return_info=True
    )

    np.testing.assert_allclose(solved, expected, rtol=2.0e-10, atol=2.0e-11)
    assert info["status"] == (0, 0, 0)
    assert max(info["relative_residual"]) < 2.0e-10
    assert info["solver"] == solver


def test_gauss_reduced_cg_uses_relative_tolerance_for_small_rhs():
    L = np.ones((6, 8))
    rho = np.zeros_like(L)
    S = np.full((*L.shape, 3), 1.0e-14)

    solved, info = ohm.solve_ohm_2d_gauss_reduced(
        L, S, rho, 1.0, rtol=1.0e-10, return_info=True
    )

    np.testing.assert_allclose(solved, S, rtol=1.0e-10, atol=0.0)
    assert info["status"] == (0, 0, 0)
    assert max(info["relative_residual"]) < 1.0e-10


def test_gauss_reduced_fft_rejects_variable_lambda():
    L, S, rho, _, delta, c, _ = _manufactured_case(variable_lambda=True)

    with pytest.raises(ValueError, match="requires L to be constant"):
        ohm.solve_ohm_2d_gauss_reduced(L, S, rho, delta, c=c, solver="fft")


@pytest.mark.parametrize("preconditioner", ["fft", None, "amg"])
def test_gauss_reduced_variable_lambda_cg_and_pcg(preconditioner):
    L, S, rho, expected, delta, c, _ = _manufactured_case(variable_lambda=True)

    solved, info = ohm.solve_ohm_2d_gauss_reduced(
        L,
        S,
        rho,
        delta,
        c=c,
        preconditioner=preconditioner,
        rtol=1.0e-10,
        return_info=True,
    )

    np.testing.assert_allclose(solved, expected, rtol=2.0e-9, atol=2.0e-10)
    assert info["status"] == (0, 0, 0)
    assert max(info["relative_residual"]) < 2.0e-9
    expected_name = "none" if preconditioner is None else preconditioner
    assert info["preconditioner"] == expected_name


def test_gauss_reduced_defaults_to_fft_preconditioning():
    L, S, rho, expected, delta, c, _ = _manufactured_case(variable_lambda=True)

    solved, info = ohm.solve_ohm_2d_gauss_reduced(
        L, S, rho, delta, c=c, rtol=1.0e-10, return_info=True
    )

    np.testing.assert_allclose(solved, expected, rtol=2.0e-9, atol=2.0e-10)
    assert info["preconditioner"] == "fft"


def test_gauss_reduced_reuses_base_matrix_and_external_preconditioner():
    L, S, rho, expected, delta, c, A = _manufactured_case(variable_lambda=True)
    ny, nx = L.shape
    base = ohm.build_ohm_gauss_base_2d(nx, ny, delta, c=c)
    external = ohm.build_ohm_gauss_preconditioner_2d(L, delta, c=c, kind="fft")
    assert isinstance(external, LinearOperator)

    solved_base = ohm.solve_ohm_2d_gauss_reduced(
        L, S, rho, delta, c=c, base=base, rtol=1.0e-10
    )
    solved_matrix, info = ohm.solve_ohm_2d_gauss_reduced(
        L,
        S,
        rho,
        delta,
        c=c,
        matrix=A,
        preconditioner=external,
        rtol=1.0e-10,
        return_info=True,
    )

    np.testing.assert_allclose(solved_base, expected, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(solved_matrix, expected, rtol=2.0e-9, atol=2.0e-10)
    assert info["preconditioner"] == "external"

    with pytest.raises(ValueError, match="matrix does not match"):
        ohm.solve_ohm_2d_gauss_reduced(
            2.0 * L, S, rho, delta, c=c, matrix=A, rtol=1.0e-10
        )
    with pytest.raises(ValueError, match="base does not match"):
        ohm.solve_ohm_2d_gauss_reduced(
            L, S, rho, delta, c=2.0 * c, base=base, rtol=1.0e-10
        )

    transposed_S = np.transpose(S, (1, 0, 2))
    with pytest.raises(ValueError, match="matrix does not match"):
        ohm.solve_ohm_2d_gauss_reduced(
            L.T, transposed_S, rho.T, delta, c=c, matrix=A, rtol=1.0e-10
        )
    with pytest.raises(ValueError, match="base does not match"):
        ohm.solve_ohm_2d_gauss_reduced(
            L.T, transposed_S, rho.T, delta, c=c, base=base, rtol=1.0e-10
        )

    corrupted = A.tolil(copy=True)
    row = nx + 1
    corrupted[row, row + 1] += 0.25
    corrupted[row, row + 3] -= 0.25
    with pytest.raises(ValueError, match="matrix does not match"):
        ohm.solve_ohm_2d_gauss_reduced(
            L, S, rho, delta, c=c, matrix=corrupted.tocsr(), rtol=1.0e-10
        )


def test_gauss_reduced_reuses_reaction_only_matrix_when_c_is_zero():
    L = np.full((6, 8), 2.0)
    S = np.random.default_rng(4).normal(size=(*L.shape, 3))
    rho = np.random.default_rng(5).normal(size=L.shape)
    base = ohm.build_ohm_gauss_base_2d(8, 6, 1.0, c=0.0)
    matrix = ohm.assemble_ohm_gauss_matrix_2d(L, 1.0, c=0.0)

    solved_matrix = ohm.solve_ohm_2d_gauss_reduced(
        L, S, rho, 1.0, c=0.0, matrix=matrix, solver="splu"
    )
    solved_base = ohm.solve_ohm_2d_gauss_reduced(
        L, S, rho, 1.0, c=0.0, base=base, solver="splu"
    )

    expected = S / L[..., None]
    np.testing.assert_allclose(solved_matrix, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(solved_base, expected, rtol=1.0e-13, atol=1.0e-13)
