# Ohm's law solver  

`picnix` solves the reduced generalized Ohm's law to reconstruct the electric
field from particle moments, the magnetic field, and charge density:

```math
    (\Lambda - c^2 \nabla^2) \boldsymbol{E}
    = -\frac{\boldsymbol{\Gamma}}{c} \times \boldsymbol{B}
    + \nabla \cdot \boldsymbol{\Pi} - c^2 \nabla \rho,
```

with periodic boundary conditions in all spatial directions.  
All three electric-field components use the same scalar reaction-diffusion
operator.  The charge-density gradient term comes from applying the
curl-curl identity $\nabla\times\nabla\times\boldsymbol{E}
= \nabla(\nabla\cdot\boldsymbol{E})-\nabla^2\boldsymbol{E}$ and
substituting Gauss's law $\nabla\cdot\boldsymbol{E}=\rho$.

The transformed moments are obtained from the per-species moment data
(``um``) via

```math
\begin{aligned}
& \Lambda = \sum_s (q_s/m_s)^2 \int f_s\, d\boldsymbol{v}, \\
& \boldsymbol{\Gamma} = \sum_s (q_s/m_s)^2 \int \boldsymbol{v} f_s\, d\boldsymbol{v}, \\
& \boldsymbol{\Pi}    = \sum_s (q_s/m_s) \int \boldsymbol{v}\boldsymbol{v}\, f_s\, d\boldsymbol{v}.
\end{aligned}
```

Charge density is reconstructed directly from raw moments as
``rho = sum_s (q_s/m_s) * um_{s,0}``.

## 1D  

In 1D ($\partial/\partial y = \partial/\partial z = 0$) the reduced
operator becomes

```math
(\Lambda - c^2 \partial_x^2) \, \boldsymbol{E}
= \boldsymbol{S} - c^2 (\partial_x \rho, 0, 0),
```

with the original source

```math
\begin{aligned}
S_x &= -\frac{1}{c}(\Gamma^y B^z - \Gamma^z B^y) + \partial_x \Pi^{xx}, \\
S_y &= -\frac{1}{c}(\Gamma^z B^x - \Gamma^x B^z) + \partial_x \Pi^{xy}, \\
S_z &= -\frac{1}{c}(\Gamma^x B^y - \Gamma^y B^x) + \partial_x \Pi^{xz}.
\end{aligned}
```

All three components are solved identically by the same circulant system
$(\Lambda - c^2 d^2/dx^2) E^\alpha = S^\alpha - c^2 \delta_{\alpha x} \partial_x \rho$
discretized on a uniform grid of spacing $\Delta$.  

For a constant $\Lambda$, the Fourier eigenvalue is

```math
\lambda(k_x) = \Lambda + \frac{4 c^2}{\Delta^2} \sin^2\!\left(\frac{k_x \Delta}{2}\right),
```

with $k_x = 2 \pi m / (N \Delta)$, $m = 0, \ldots, N-1$.

## 2D  

In 2D in the x-y plane ($\partial/\partial z = 0$)

```math
(\Lambda - c^2 \nabla^2) \, \boldsymbol{E}
= \boldsymbol{S} - c^2 (\partial_x \rho, \partial_y \rho, 0).
```

The three components are decoupled and share the same finite-difference
operator.  The in-plane components receive the charge-density gradient
correction; $E_z$ does not.

For a constant $\Lambda$, the Fourier eigenvalue is

```math
\lambda(\boldsymbol{k})
= \Lambda + \frac{4 c^2}{\Delta^2}
  \!\left[
    \sin^2\!\left(\frac{k_x \Delta}{2}\right)
   +\sin^2\!\left(\frac{k_y \Delta}{2}\right)
  \right],
```

for the 2D mode $e^{i (k_x i \Delta + k_y j \Delta)}$.

## Solver  

The low-level solvers expect a **pre-reduced** source term

```math
\boldsymbol{S}_{\text{reduced}} = \boldsymbol{S}_{\text{original}} - c^2 \nabla \rho
```

and solve

```math
(\Lambda - c^2 \nabla^2) \boldsymbol{E} = \boldsymbol{S}_{\text{reduced}}.
```

The charge-density gradient is computed inside `_build_source_1d` / `_build_source_2d`;
callers get the reduced source directly.  The high-level helpers
`calc_e_ohm_1d` and `calc_e_ohm_2d` handle this automatically via
`transform_moments`.

The default solver uses conjugate gradient.  Pass an FFT preconditioner
via the `M` argument (build with `_build_fft_preconditioner_2d`) or
pass `M=None` for unpreconditioned CG.

### API

```python
import picnix
from picnix import ohm

# Low-level (pre-reduced source):
E = picnix.solve_ohm_2d(Lambda, S_reduced, delta, c=1.0)

# or with FFT-preconditioned CG and info:
M_prec = ohm._build_fft_preconditioner_2d(Lambda.shape, delta, c, float(np.mean(Lambda)))
E, info = picnix.solve_ohm_2d(
    Lambda, S_reduced, delta, c=1.0,
    M=M_prec, rtol=1e-12, return_info=True
)
```

With `return_info=True`, diagnostics contain component-wise convergence
statuses, iteration counts, final true relative residuals, and
preconditioner name.  The reported residual is

```math
\frac{\lVert R_\alpha - A E_\alpha \rVert_2}
{\max(\lVert R_\alpha \rVert_2, \epsilon)}.
```

## Charge-density computation  

PIC-NIX field diagnostics do not write ``uj`` or a standalone
charge-density array; ``rho`` is returned by ``transform_moments(um, qm)``
alongside Lambda, Gamma, and Pi.  For direct use without the high-level
helpers, compute it manually:

```python
rho = np.sum(um[..., :, 0] * qm, axis=-1)
```

No automatic run wrapper is provided at the low level because diagnostic
decimation changes the grid spacing, raw field output may retain
staggering, legacy profiles may not contain exact ``qm``, and float32
HDF5 conversion can amplify cancellation error in nearly neutral
plasmas.  Charge-density noise, field/moment time centering, and
compatibility between discrete divergence and gradient operators remain
physical interpretation concerns.  Solving the reduced equation does not
by itself guarantee a small Gauss-law residual for inconsistent input
data.

## From a picnix run  

For a single snapshot, ``calc_e_ohm_1d`` and ``calc_e_ohm_2d`` read the
field and moment data, use ``run.qm`` (resolved at init time from profile
metadata or config), and build the reduced source automatically via
``transform_moments``:

```python
import picnix

run = picnix.Run("data/profile.msgpack")
step = run.get_step("field")[-1]
E_ohm = picnix.calc_e_ohm_1d(run, step, c=1.0)  # or calc_e_ohm_2d
```

The return value has shape ``(Nx, 3)`` (1D) or ``(Ny, Nx, 3)`` (2D).  

If the profile does not carry per-species ``qm`` and cannot be inferred
from the config, ``Run`` leaves ``qm=None``; use the ``picnix-ohm-compare``
CLI with ``--qm`` to provide it explicitly when needed.

## Limitations  

- Periodic boundary conditions only.  
- Uniform grid (``dx == dy == delta``) assumed.  
- Each grid axis must contain at least three points.  
- 1D and 2D only. 3D solver not provided yet.
