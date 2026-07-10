# Ohm's law solver

`picnix` provides a reference curl-curl solver and a Gauss-law-reduced solver for reconstructing the electric field from particle moments, the magnetic field, and charge density.  
The reference formulation solves:

```math
    (\Lambda + c^2 \nabla \times \nabla \times) \boldsymbol{E}
    = -\frac{\boldsymbol{\Gamma}}{c} \times \boldsymbol{B}
    + \nabla \cdot \boldsymbol{\Pi},
```

with periodic boundary conditions in all spatial directions.
The transformed moments are obtained from the per-species moment data (``um``) via

```math
\begin{aligned}
& \Lambda = \sum_s (q_s/m_s)^2 \int f_s\, d\boldsymbol{v}, \\
& \boldsymbol{\Gamma} = \sum_s (q_s/m_s)^2 \int \boldsymbol{v} f_s\, d\boldsymbol{v}, \\
& \boldsymbol{\Pi}    = \sum_s (q_s/m_s) \int \boldsymbol{v}\boldsymbol{v}\, f_s\, d\boldsymbol{v}.
\end{aligned}
```

which are given in Lorentz-Heaviside units.

## 1D

In 1D ($\partial/\partial y = \partial/\partial z = 0$), the curl-curl identity

```math
\nabla \times \nabla \times \boldsymbol{E} = -\nabla^2 \boldsymbol{E} + \nabla (\nabla \cdot \boldsymbol{E})
```

reduces, component by component, to

```math
\begin{aligned}
(\nabla \times \nabla \times \boldsymbol{E})_x &= -\partial_x^2 E_x + \partial_x^2 E_x = 0, \\
(\nabla \times \nabla \times \boldsymbol{E})_y &= -\partial_x^2 E_y, \\
(\nabla \times \nabla \times \boldsymbol{E})_z &= -\partial_x^2 E_z.
\end{aligned}
```

The 1D Ohm's law then becomes

```math
\begin{aligned}
\Lambda \, E_x &= -\frac{1}{c}(\Gamma^y B^z - \Gamma^z B^y) + \partial_x \Pi^{xx}, \\
(\Lambda - c^2 \partial_x^2) \, E_y &= -\frac{1}{c}(\Gamma^z B^x - \Gamma^x B^z) + \partial_x \Pi^{xy}, \\
(\Lambda - c^2 \partial_x^2) \, E_z &= -\frac{1}{c}(\Gamma^x B^y - \Gamma^y B^x) + \partial_x \Pi^{xz}.
\end{aligned}
```

The equation for the $x$ component has no spatial derivative and can be easily solved as
$E_x = S_x / \Lambda$. The $y$ and $z$ components are identical and decoupled equations.

Discretizing $\partial_x^2$ by the second-order central difference on a uniform grid of spacing $\Delta$ with periodic boundary conditions, the Fourier mode

```math
e^{i k_x x}, \quad k_x = \frac{2 \pi m}{N \Delta}, \quad m = 0, \ldots, N-1
```

is an eigenfunction of the operator $(\Lambda - c^2 \partial_x^2)$ with eigenvalue

```math
\lambda(k_x) = \Lambda + \frac{4 c^2}{\Delta^2} \sin^2\!\left(\frac{k_x \Delta}{2}\right),
```

assuming a constant $\Lambda$. The $y$ and $z$ components can be solved by the CG method.

## 2D

In 2D in the x-y plane ($\partial/\partial z = 0$), the Ohm's law then becomes

```math
\begin{aligned}
\bigl(\Lambda - c^2 \partial_y^2\bigr) E^x + c^2 \partial_x \partial_y E^y
&= -\frac{1}{c}(\Gamma^y B^z - \Gamma^z B^y)
+ \partial_x \Pi^{xx} + \partial_y \Pi^{yx} \\
c^2 \partial_x \partial_y E^x + \bigl(\Lambda - c^2 \partial_x^2\bigr) E^y
&= -\frac{1}{c}(\Gamma^z B^x - \Gamma^x B^z)
+ \partial_x \Pi^{xy} + \partial_y \Pi^{yy} \\
\bigl(\Lambda - c^2 (\partial_x^2 + \partial_y^2)\bigr) E^z
&= -\frac{1}{c}(\Gamma^x B^y - \Gamma^y B^x)
+ \partial_x \Pi^{xz} + \partial_y \Pi^{yz}.
\end{aligned}
```

The $x$ and $y$ components are coupled whereas the $z$ component is decoupled.
Both are solved with the CG method.

If we assume a constant $\Lambda$ and the periodic boundary conditions, the Fourier eigenvalues with the second-order central difference are given by

```math
\begin{aligned}
A_{xx} &= \Lambda + 4 \frac{c^2}{\Delta^2} \sin^2\!\left(\frac{k_y \Delta}{2}\right), \\
A_{yy} &= \Lambda + 4 \frac{c^2}{\Delta^2} \sin^2\!\left(\frac{k_x \Delta}{2}\right), \\
A_{xy} &= -\frac{c^2}{\Delta^2} \sin(k_x \Delta) \sin(k_y \Delta), \\
A_{zz} &= \Lambda + 4 \frac{c^2}{\Delta^2}\!\left[\sin^2\!\left(\frac{k_x \Delta}{2}\right) + \sin^2\!\left(\frac{k_y \Delta}{2}\right)\right],
\end{aligned}
```

for the 2D mode $e^{i (k_x i \Delta + k_y j \Delta)}$. In matrix form, the Ohm's law can be written as

```math
\begin{pmatrix}
A_{xx} & A_{xy} & 0 \\
A_{xy} & A_{yy} & 0 \\
0      & 0      & A_{zz}
\end{pmatrix}
\begin{pmatrix} \tilde{E}^x \\ \tilde{E}^y \\ \tilde{E}^z\end{pmatrix}
=
\begin{pmatrix} \tilde{S}^x \\ \tilde{S}^y \\ \tilde{S}^z \end{pmatrix},
```
where $\tilde{A}$ is the Fourier amplitude of $A$.

## Gauss-law-reduced 2D solver

PIC-NIX uses Lorentz-Heaviside units, so Gauss's law is

```math
\nabla\cdot\boldsymbol E=\rho
```

without a $4\pi$ factor.  
Using

```math
\nabla\times\nabla\times\boldsymbol E
=\nabla(\nabla\cdot\boldsymbol E)-\nabla^2\boldsymbol E,
```

the generalized Ohm equation can be reduced to

```math
\left(\Lambda-c^2\nabla^2\right)\boldsymbol E
=\boldsymbol S-c^2\nabla\rho.
```

All three electric-field components therefore use the same scalar reaction-diffusion operator.  
In 2D, the charge-density correction is applied only to the in-plane components:

```math
\begin{aligned}
R_x &= S_x-c^2\partial_x\rho, \\
R_y &= S_y-c^2\partial_y\rho, \\
R_z &= S_z.
\end{aligned}
```

The implementation uses centered second-order periodic differences for $\nabla_h\rho$.  
For a uniform spacing $\Delta$, the shared sparse matrix is

```math
A=\operatorname{diag}(\Lambda)-c^2L_h,
```

where $L_h$ is the negative-semidefinite periodic five-point Laplacian.  
The matrix is symmetric positive definite when every value of $\Lambda$ is positive.  
The solver rejects non-finite and non-positive $\Lambda$ rather than silently applying a floor.  
The optional `min_lambda` argument can reject an explicitly chosen near-zero range.

### API and methods

The reduced low-level solver requires charge density explicitly:

```python
import picnix

E, info = picnix.solve_ohm_2d_gauss_reduced(
    Lambda,
    S,
    rho,
    delta,
    c=1.0,
    solver="cg",
    preconditioner="amg",
    return_info=True,
)
```

Available solver methods are:

- `cg`: production finite-difference CG for spatially varying positive $\Lambda$.  
- `fft`: exact direct FFT solve for constant $\Lambda$ and periodic boundaries.  
- `splu`: sparse-LU reference for small problems; fill-in makes it unsuitable for large grids.  

CG uses the FFT constant-coefficient preconditioner by default.  
Pass `None` for an unpreconditioned CG baseline, `"amg"` for scalar AMG, `"fft"` explicitly, or an external SciPy `LinearOperator`.  
The AMG hierarchy or FFT `LinearOperator` is built once and reused for all three component solves.  
Callers performing repeated solves can also reuse a matrix built by `assemble_ohm_gauss_matrix_2d`, a base built by `build_ohm_gauss_base_2d`, or a preconditioner built by `build_ohm_gauss_preconditioner_2d`.  
A complete matrix is valid only for the same $\Lambda$, grid shape, $\Delta$, and $c$; by default the solver compares it with the complete expected sparse finite-difference operator.  
The Lambda-independent base can be reused when the grid shape, $\Delta$, and $c$ are unchanged.
Trusted performance-sensitive callers may set `validate_matrix=False` after validating a reused matrix once; true residuals are still evaluated against the requested finite-difference equation.

With `return_info=True`, diagnostics contain component-wise convergence statuses, iteration counts, final true relative residuals, and solver/preconditioner names.  
The reported residual is

```math
\frac{\lVert R_\alpha-AE_\alpha\rVert_2}
{\max(\lVert R_\alpha\rVert_2,\epsilon)}.
```

### FFT discretization

The direct FFT method solves the same finite-difference equation as CG; it does not substitute the continuous spectral Laplacian.  
For constant $\Lambda=\Lambda_0$, its denominator is

```math
\Lambda_0+\frac{4c^2}{\Delta^2}
\left[
\sin^2\left(\frac{k_x\Delta}{2}\right)
+\sin^2\left(\frac{k_y\Delta}{2}\right)
\right].
```

For variable $\Lambda$, the FFT option is available only as a constant-coefficient PCG preconditioner.  
It uses the arithmetic mean of $\Lambda$ by default, or the positive coefficient supplied through `fft_lambda`.  
It never approximates the variable-coefficient operator itself as diagonal in Fourier space.

### Charge-density caveats

The reduced solver deliberately accepts `rho` as a low-level input.  
PIC-NIX field diagnostics do not write `uj` or a standalone charge-density array, although charge density can be reconstructed from raw moments as

```python
rho = np.sum(um[..., :, 0] * qm, axis=-1)
```

where `qm` is the exact per-species charge-to-mass ratio.  
No automatic run wrapper is provided yet because diagnostic decimation changes the grid spacing, raw field output may retain staggering, legacy profiles may not contain exact `qm`, and float32 HDF5 conversion can amplify cancellation error in nearly neutral plasmas.  
Charge-density noise, field/moment time centering, and compatibility between discrete divergence and gradient operators remain physical interpretation concerns.  
Solving the reduced equation does not by itself guarantee a small Gauss-law residual for inconsistent input data.

## From a picnix run

For a single snapshot, `calc_e_ohm_1d` and `calc_e_ohm_2d` read the field and moment data, infer per-species `q_s/m_s` from the config, build the source term, and call the corresponding curl-curl solver.  
The Gauss-law-reduced solver currently uses only the explicit low-level interface described above.

```python
import picnix

run = picnix.Run("data/profile.msgpack")
step = run.get_step("field")[-1]
E_ohm = picnix.calc_e_ohm_1d(run, step, c=1.0)  # or calc_e_ohm_2d
```

The return value has shape `(Nx, 3)` (1D) or `(Ny, Nx, 3)` (2D).

## Limitations

- Periodic boundary conditions only.
- Uniform grid (`dx == dy == delta`) assumed.
- Each 2D grid axis must contain at least three points.
- 1D and 2D only. 3D solver not provided yet.
