# Ohm's law solver

A generalized Ohm's law solver is provided in `picnix` to reconstruct the electric field from the particle moment data and the magnetic field. It solve the following equation:

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

## From a picnix run

For a single snapshot, ``calc_e_ohm_1d`` and ``calc_e_ohm_2d`` read
the field and moment data, infer per-species ``q_s/m_s`` from the
config, build the source term, and call the corresponding solver.

```python
import picnix

run = picnix.Run("data/profile.msgpack")
step = run.get_step("field")[-1]
E_ohm = picnix.calc_e_ohm_1d(run, step, c=1.0)  # or calc_e_ohm_2d
```

The return value has shape ``(Nx, 3)`` (1D) or ``(Ny, Nx, 3)`` (2D).

## Per-species q/m

``qm_per_species_from_config`` resolves ``q_s/m_s`` from the picnix
config in two ways:

1. ``[[parameter.particle]]`` array of per-species ``qm`` values
   (e.g., ``beam/twostream``, ``foot/``)
2. Top-level ``mime``/``nppc``/``wp`` keys, which imply a 2-species
   electron-ion pair (e.g., ``anisotropy``, ``heatflux``)

For multi-species cases that do not fit either pattern (such as the
3-species ``heatflux`` run, which has a core electron, a beam electron,
and an ion), pass ``qm`` explicitly:

```python
E_ohm = picnix.calc_e_ohm_2d(run, step, c=1.0, qm_per_species=[-1.0, -1.0, 0.01])
```

## Limitations

- Periodic boundary conditions only.
- Uniform grid (``dx == dy == delta``) assumed.
- 1D and 2D only. 3D solver not provided yet.

