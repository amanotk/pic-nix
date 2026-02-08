# Wave Tool Documentation

## Generalized Ohm's Law

The generalized Ohm's law for collisionless plasmas is given by:

```math
\boldsymbol{E} + \boldsymbol{v} \times \boldsymbol{B} = \frac{1}{ne} \left( \boldsymbol{J} \times \boldsymbol{B} - \nabla \cdot \boldsymbol{P}_e \right)
```

## Definitions

### Λ (Lambda)

```math
\Lambda = \int f \, d^3v
```

### Γ (Gamma)

```math
\Gamma = \int \boldsymbol{v} f \, d^3v
```

### Π (Pi)

```math
\Pi = \int \boldsymbol{v} \boldsymbol{v} f \, d^3v
```

## Moment Quantities

The moment quantities are defined as:

```math
\begin{aligned}
n &= \int f \, d^3v \\
\boldsymbol{u} &= \frac{1}{n} \int \boldsymbol{v} f \, d^3v \\
P_{ij} &= m \int (v_i - u_i)(v_j - u_j) f \, d^3v
\end{aligned}
```

## Transformed Moments

The transformed moment equations in matrix form are:

```math
\mathbf{M} = \begin{pmatrix}
M_{11} & M_{12} & M_{13} \\
M_{21} & M_{22} & M_{23} \\
M_{31} & M_{32} & M_{33}
\end{pmatrix}
```

## Vector Notation

The wave vector is denoted as:

```math
\boldsymbol{k} = k_x \hat{\boldsymbol{x}} + k_y \hat{\boldsymbol{y}} + k_z \hat{\boldsymbol{z}}
```

with magnitude:

```math
k = |\boldsymbol{k}| = \sqrt{k_x^2 + k_y^2 + k_z^2}
```
