# Equations and Units

PIC-NIX uses rationalized Lorentz-Heaviside electromagnetic units. Equivalently,
the factors of $4\pi$ used in Gaussian CGS equations are absorbed into the
definitions of charge and the electromagnetic fields. The code does not impose
a single normalization for time, length, mass, or density; each simulation must
choose a consistent set of input quantities.  

## Relativistic Vlasov equation

For species $s$, PIC-NIX evolves a distribution in position
$\boldsymbol{x}$ and four-velocity $\boldsymbol{u} = \gamma\boldsymbol{v}$:  

```math
\frac{\partial f_s}{\partial t}
+ \frac{\boldsymbol{u}}{\gamma}\cdot\nabla f_s
+ \frac{q_s}{m_s}
\left(
  \boldsymbol{E}
  + \frac{\boldsymbol{u}}{\gamma c}\times\boldsymbol{B}
\right)\cdot\frac{\partial f_s}{\partial\boldsymbol{u}}
= 0,
```

where  

```math
\gamma = \sqrt{1 + \frac{|\boldsymbol{u}|^2}{c^2}}.
```

The speed of light $c$ is the `parameter.cc` value in an example configuration.  

## Maxwell equations

The field update follows:  

```math
\begin{aligned}
\frac{\partial\boldsymbol{B}}{\partial t}
  &= -c\,\nabla\times\boldsymbol{E}, \\
\frac{\partial\boldsymbol{E}}{\partial t}
  &= c\,\nabla\times\boldsymbol{B} - \boldsymbol{J}, \\
\nabla\cdot\boldsymbol{E} &= \rho, \\
\nabla\cdot\boldsymbol{B} &= 0.
\end{aligned}
```

In particular, Gauss's law contains no factor of $4\pi$.  

## Characteristic frequencies

With these conventions, the non-relativistic plasma and signed cyclotron
frequencies are  

```math
\omega_{ps} = \sqrt{\frac{n_s q_s^2}{m_s}},
\qquad
\Omega_{cs} = \frac{q_s B}{m_s c}.
```

Examples often select parameters so one frequency, mass, or length scale is
unity. Those choices are example-specific rather than global code units. Check
the example's README, `config.toml`, and initializer before interpreting its
dimensional scales.  
