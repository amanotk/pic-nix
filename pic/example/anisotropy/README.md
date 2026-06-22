# Temperature Anisotropy Driven Instability

This setup models the temperature anisotropy driven instability in a magnetized plasma  
with periodic boundary conditions in all three directions.  
The system initially consists of bi-Maxwellian electrons and ions.

## Physical Parameters

The following parameters should be defined in the configuration file:

* `cc` : speed of light $c$
* `wp` : electron plasma frequency $\omega_{pe}$
* `mime` : ion-to-electron mass ratio $m_i/m_e$
* `theta` : polar angle of the ambient magnetic field with respect to the
  x-axis $\theta$
* `phi` : azimuthal angle of the ambient magnetic field with respect to the
  x-axis $\phi$
* `sigma` : electron cyclotron-to-plasma frequency squared
  $\sigma = \Omega_{ce}^2/\omega_{pe}^2$
* `betae_para` : electron parallel plasma beta $\beta_{e,\parallel}$
* `betae_perp` : electron perpendicular plasma beta $\beta_{e,\perp}$
* `betai_para` : ion parallel plasma beta $\beta_{i,\parallel}$
* `betai_perp` : ion perpendicular plasma beta $\beta_{i,\perp}$
* `nppc` : number of particles per cell per species
* `delt` : time step $\Delta t$ in units of $\omega_{pe}^{-1}$
* `delh` : grid spacing $\Delta h$ in units of $c/\omega_{pe}$

The electron charge-to-mass ratio is assumed to be unity $|e|/m_e = 1$, which gives the ambient magnetic field $B_0 = c \sqrt{\sigma}$.
The three components of the ambient magnetic field are then given by
```math
\begin{aligned}
B_{0,x} &= B_0 \cos \theta \\
B_{0,y} &= B_0 \sin \theta \cos \phi \\
B_{0,z} &= B_0 \sin \theta \sin \phi
\end{aligned}
```
The thermal velocities are related to the plasma beta via
```math
v_{th, s, \parallel} = v_{A,s} \sqrt{\frac{\beta_{s, \parallel}}{2}}, \qquad
v_{th, s, \perp} = v_{A,s} \sqrt{\frac{\beta_{s, \perp}}{2}},
```
where $v_{A,s}/c = B_0 / \sqrt{n_0 m_s}$ is the Alfvén speed defined for species $s$ with the number density $n_0$.

## Scenarios

The default configuration file `config.toml` provides an example for the whistler instability driven by electron temperature anisotropy with $\beta_{e,\perp} = 3.0$ and $\beta_{e,\parallel} = 1.0$. The ions are isotropic with $\beta_{i,\perp} = \beta_{i,\parallel} = 1.0$. See, Gary and Wang (1996) for more details.

## References

* Gary, S. P., & Wang, J. (1996). Whistler instability: Electron temperature
  anisotropy in the solar wind. *Journal of Geophysical Research*,
  *101*(A5), 10749–10758. https://doi.org/10.1029/96JA00354
